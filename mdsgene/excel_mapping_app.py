# excel_mapping_app.py
import json
import sys
import os
import traceback
import re
from pathlib import Path

from typing import Dict, Optional, Set, Callable

from cache_manager import CacheManager
from excel_writer import ExcelWriter
from gemini_processor import GeminiProcessor
from mapping_item import MappingItem, QuestionInfo
from pmid_extractor import PmidExtractor


# --- Configuration ---
# PDF File Path (Update as needed)
PDF_FILEPATH = Path(r"./.pdf_docs/ando2012-22991136.pdf")  # Example from gemini_request_sample.txt

# Output Excel Path
OUTPUT_EXCEL_PATH = Path("./.tables/Patient_Data_Gemini_Output.xlsx")

# Gemini Model Name (Update if needed, ensure it matches your access)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")  # Or "gemini-1.5-pro" etc.

# Define which fields are considered "common" (apply to whole study) vs "patient-specific"
# Currently, all fields are processed per patient. Set to empty list if not used.
COMMON_FIELDS = ["pmid", "author, year", "comments_study"]

# --- NEW CONFIGURATION ---
CACHE_DIR = Path(os.getenv("GEMINI_CACHE_DIR", ".gemini_cache"))  # Directory for cache files

# Create the cache directory if it doesn't exist
if not CACHE_DIR.exists():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created at: {CACHE_DIR}")
    except Exception as e:
        print(f"Error creating cache directory: {e}", file=sys.stderr)
        sys.exit(1)

PMID_CACHE_PATH = CACHE_DIR / "pmid_cache.json"

def load_pmid_cache() -> Dict[str, dict]:
    if PMID_CACHE_PATH.exists():
        with open(PMID_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_pmid_cache(cache: Dict[str, dict]):
    with open(PMID_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


class ExcelMappingApp:
    """Orchestrates PDF processing, Gemini querying/formatting, and Excel writing."""

    # --- ИЗМЕНЕНО: __init__ с поддержкой documents и document_id ---
    def __init__(self, pdf_filepath: Path, model_name: str, document_id=None, documents=None):
        """Initializes application state."""
        self.pdf_filepath = pdf_filepath
        self.model_name = model_name
        self.gemini_processor: Optional[GeminiProcessor] = None
        self.external_pmid: Optional[str] = None  # Будет установлен в run
        self.document_id = document_id
        self.documents = documents
        self.log_messages = []  # Для хранения сообщений логов
        # CacheManager будет инициализирован локально в run
        self.log("ExcelMappingApp initialized.")

    def log(self, message, level="INFO", add_to_steps=True, error=False):
        """
        Заменяет стандартный вывод print на добавление в steps.

        Args:
            message: Сообщение для логирования
            level: Уровень логирования (INFO, WARNING, ERROR)
            add_to_steps: Добавлять ли сообщение в steps
            error: Выводить ли в stderr вместо stdout
        """
        # Всегда выводим в консоль для обратной совместимости
        if error:
            print(message, file=sys.stderr)
        else:
            print(message)

        # Сохраняем сообщение в локальном списке
        self.log_messages.append({"level": level, "message": message})

        # Добавляем в steps, если требуется и есть доступ к documents
        if add_to_steps and self.documents and self.document_id and self.document_id in self.documents:
            # Создаем запись для steps, если её еще нет
            if "steps" not in self.documents[self.document_id]:
                self.documents[self.document_id]["steps"] = []

            # Добавляем сообщение как шаг обработки
            import datetime
            # Генерируем уникальный ID для шага
            step_id = f"log_{len(self.documents[self.document_id].get('steps', []))}"
            # Определяем статус на основе уровня логирования
            status = "success" if level == "INFO" else "error" if level == "ERROR" else "warning"

            self.documents[self.document_id]["steps"].append({
                # Атрибуты для фронтенда
                "id": step_id,
                "title": message,
                "answer": "",
                "status": status,
                # Оригинальные атрибуты для обратной совместимости
                "type": "log",
                "level": level,
                "message": message,
                "timestamp": str(datetime.datetime.now())
            })

    # --- Custom Field Processor Implementations ---
    def _default_processor(
        self,
        item: MappingItem,
        raw_answer: str | None = None,
    ) -> dict[str, str]:
        """
        Default processor: Uses the Gemini formatter for the raw answer.
        Now expects raw_answer to be a dict: {"answer": ..., "evidence": ...}
        """
        if self.gemini_processor is None:
            self.log("ERROR: Gemini processor not initialized in _default_processor.", level="ERROR", error=True)
            return {item.field: "-99_PROCESSOR_ERROR"}
        try:
            # --- NEW: Expect dict with 'answer' and 'evidence' ---
            if isinstance(raw_answer, dict):
                formatted_answer = self.gemini_processor.format_answer(raw_answer.get("answer"), item.response_convertion_strategy)
                evidence = raw_answer.get("evidence", "")
            else:
                formatted_answer = self.gemini_processor.format_answer(raw_answer, item.response_convertion_strategy)
                evidence = ""
            return {
                item.field: formatted_answer,
                f"{item.field}_evidence": evidence
            }
        except Exception as e:
            self.log(f"ERROR in default_processor formatting field '{item.field}': {e}", level="ERROR", error=True)
            return {item.field: "-99_FORMATTING_ERROR", f"{item.field}_evidence": ""}

    def _motor_symptom_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        """
        Processor for parsing MOTOR symptoms list like "symptom_name:yes/no;..."
        Generates columns like "motor_symptomname_sympt".
        Parses the RAW answer directly.
        """
        results: Dict[str, str] = {}
        prefix = "motor"  # Prefix for motor symptom columns
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported",
                                                            "information not found"]:
            return results  # Return empty map if no symptoms reported
        symptoms = re.split(r'[;\n]', raw_answer)  # Split by semicolon or newline

        for symptom_entry in symptoms:
            symptom_entry = symptom_entry.strip()
            if not symptom_entry:
                continue
            parts = symptom_entry.split(":", 1)  # Split only on the first colon
            if len(parts) == 2:
                symptom_name_raw = parts[0].strip()
                if not symptom_name_raw or symptom_name_raw.lower() == "none":
                    continue
                symptom_name = symptom_name_raw.lower()
                symptom_name = re.sub(r'[\s\(\)-]+', '_', symptom_name)
                symptom_name = re.sub(r'_+', '_', symptom_name)
                symptom_name = re.sub(r'[^\w]+', '', symptom_name)
                symptom_name = symptom_name.strip('_')
                if not symptom_name: 
                    self.log(f"  -> Motor Symptom WARNING: Name '{symptom_name_raw}' became empty after sanitizing.", level="WARNING")
                    continue
                presence = parts[1].strip().lower()
                column_name = f"{prefix}_{symptom_name}_sympt"
                if presence in ["yes", "present", "positive", "true", "1", "y"]:
                    value = "yes"
                elif presence in ["no", "absent", "negative", "false", "0", "n"]:
                    value = "no"
                else:
                    self.log(f"  -> Motor Symptom WARNING: Unknown presence value '{presence}' for symptom '{symptom_name_raw}'. Setting column '{column_name}' to -99.", level="WARNING")
                    value = "-99"
                results[column_name] = value
            elif len(parts) == 1 and len(symptoms) == 1 and parts[0].lower() in ["none", "n/a", "not applicable", "none reported"]:
                break  # No symptoms to add
        return results

    def _non_motor_symptom_processor(
        self,
        raw_answer: Optional[str],
        item: MappingItem
    ) -> dict[str, str]:
        """
        Processor for parsing NON-MOTOR symptoms list like "symptom_name:yes/no;..."
        Generates columns like "nms_symptomname_sympt".
        Parses the RAW answer directly.
        """
        results: Dict[str, str] = {}
        prefix = "nms"  # Prefix for non-motor symptom columns
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported",
                                                            "information not found"]:
            self.log(f"  NMS Processor: No symptoms found or raw answer empty/negative ('{raw_answer}').")
            return results

        self.log(f"  NMS Processor: Processing raw answer: '{raw_answer[:100]}...'")
        symptoms = re.split(r'[;\n]', raw_answer)

        for symptom_entry in symptoms:
            symptom_entry = symptom_entry.strip()
            if not symptom_entry:
                continue

            parts = symptom_entry.split(":", 1)
            if len(parts) == 2:
                symptom_name_raw = parts[0].strip()
                if not symptom_name_raw or symptom_name_raw.lower() == "none":
                    continue
                symptom_name = symptom_name_raw.lower()
                symptom_name = re.sub(r'[\s\(\)-]+', '_', symptom_name)
                symptom_name = re.sub(r'_+', '_', symptom_name)
                symptom_name = re.sub(r'[^\w]+', '', symptom_name)
                symptom_name = symptom_name.strip('_')
                if not symptom_name: 
                    self.log(f"  -> NMS WARNING: Name '{symptom_name_raw}' became empty after sanitizing.", level="WARNING")
                    continue
                presence = parts[1].strip().lower()
                column_name = f"{prefix}_{symptom_name}_sympt"
                if presence in ["yes", "present", "positive", "true", "1", "y"]:
                    value = "yes"
                elif presence in ["no", "absent", "negative", "false", "0", "n"]:
                    value = "no"
                else:
                    self.log(f"  -> NMS WARNING: Unknown presence value '{presence}' for symptom '{symptom_name_raw}'. Setting column '{column_name}' to -99.", level="WARNING")
                    value = "-99"
                results[column_name] = value
            elif len(parts) == 1 and len(symptoms) == 1 and parts[0].lower() in ["none", "n/a", "not applicable", "none reported"]:
                break
        return results

    def _symptom_list_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        results: Dict[str, str] = {}
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported", "information not found"]:
            return results
        symptoms = re.split(r'[;\n]', raw_answer)
        found_structured = False
        for symptom_entry in symptoms:
            symptom_entry = symptom_entry.strip()
            if not symptom_entry:
                continue
            parts = symptom_entry.split(":", 1)
            if len(parts) == 2:
                symptom_name_raw = parts[0].strip()
                if not symptom_name_raw:
                    continue
                symptom_name = symptom_name_raw.lower()
                symptom_name = re.sub(r'\s+', '_', symptom_name)
                symptom_name = re.sub(r'[^a-z0-9_]+', '', symptom_name)
                if not symptom_name: 
                    self.log(f"  -> Symptom WARNING: Name '{symptom_name_raw}' became empty after sanitizing.", level="WARNING")
                    continue
                presence = parts[1].strip().lower()
                column_name = f"{symptom_name}_sympt"
                if presence in ["yes", "present", "positive", "true", "1", "y"]:
                    value = "yes"
                elif presence in ["no", "absent", "negative", "false", "0", "n"]:
                    value = "no"
                else:
                    self.log(f"  -> Symptom WARNING: Unknown presence value '{presence}' for symptom '{symptom_name_raw}'. Setting column '{column_name}' to -99.", level="WARNING")
                    value = "-99"
                results[column_name] = value
                found_structured = True
            elif len(parts) == 1 and len(symptoms) == 1 and parts[0].lower() in ["none", "n/a", "not applicable",
                                                                                 "none reported"]:
                found_structured = True
                break
            else:
                self.log(f"  -> Symptom WARNING: Could not parse entry like 'name:value': '{symptom_entry}'", level="WARNING")
        if not found_structured and raw_answer.strip(): 
            self.log(f"  -> Symptom WARNING: Parsing yielded no structured results for: '{raw_answer[:50]}...'. Storing raw in '{item.field}'.", level="WARNING")
        results[item.field] = "-99"
        return results

    def _hpo_symptom_processor(
        self,
        item: MappingItem,
        raw_answer: str | None = None,
    ) -> dict[str, str]:
        """
        Processor for parsing HPO coded symptoms like "headache_HP:0002315:yes;..."
        Generates columns like "HP_0002315". Parses the RAW answer directly.
        """
        results: Dict[str, str] = {}
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported", "information not found"]:
            return results
        hpo_pattern = re.compile(
            r"(HP[:_ ]?\s?\d{7})[\s:\-\(\)]+(yes|no|present|absent|positive|negative|true|false|1|0)", re.IGNORECASE)
        found_structured = False
        matches = hpo_pattern.findall(raw_answer)
        if not matches and raw_answer.strip().lower() in ["none", "n/a", "not applicable", "none reported"]:
            found_structured = True
        for match in matches:
            hpo_id_raw = match[0]
            presence = match[1].lower()
            hpo_digits = re.sub(r'[^0-9]', '', hpo_id_raw)
            if len(hpo_digits) == 7:
                hpo_id_normalized = "HP_" + hpo_digits
            else:
                self.log(f"  -> HPO WARNING: Invalid HPO ID format found: '{hpo_id_raw}'. Skipping.", level="WARNING")
                continue
            column_name = hpo_id_normalized
            value = "yes" if presence in ["yes", "present", "positive", "true", "1", "y"] else "no"
            results[column_name] = value
            found_structured = True
        if not found_structured and raw_answer.strip(): 
            self.log(f"  -> HPO WARNING: HPO parsing yielded no structured results for: '{raw_answer[:50]}...'. Storing raw in '{item.field}'.", level="WARNING")
        results[item.field] = "-99"
        return results

    def create_mapping_data(self, mapping_json_path: Path = None) -> list['MappingItem']:
        """
        Загружает список MappingItem из JSON-файла.
        Каждый элемент JSON должен содержать:
        - field
        - question
        - mapped_excel_column
        - response_convertion_strategy
        - custom_processor_key (или None)
        """
        import json

        # Determine the path to the mapping JSON file
        if mapping_json_path is None:
            mapping_json_path = Path(os.getenv("MAPPING_JSON_PATH", ".questions/mapping_data.json"))
        if not mapping_json_path.exists():
            raise FileNotFoundError(f"Mapping JSON file not found: {mapping_json_path}")

        # Processor map
        processor_map: Dict[str, Callable[[Optional[str], 'MappingItem'], Dict[str, str]]] = {
            "motor_symptoms": self._motor_symptom_processor,
            "non_motor_symptoms": self._non_motor_symptom_processor,
            "hpo_phenotypes": getattr(self, "_hpo_symptom_processor", None),
            # Add more processors as needed
        }

        mapping_data: list['MappingItem'] = []
        with open(mapping_json_path, "r", encoding="utf-8") as f:
            items = json.load(f)
            for entry in items:
                processor_key = entry.get("custom_processor")
                processor = processor_map.get(processor_key) if processor_key else None
                mapping_data.append(
                    MappingItem(
                        field=entry["field"],
                        question=entry["question"],
                        mapped_excel_column=entry["mapped_excel_column"],
                        response_convertion_strategy=entry["response_convertion_strategy"],
                        custom_processor=processor
                    )
                )
        self.log(f"Loaded {len(mapping_data)} mapping items from JSON.")
        return mapping_data

    def run(self):
        """Runs the main application workflow using Gemini with PMID-based Caching."""
        if not self.documents or not self.document_id:
            self.log("WARNING: documents или document_id не переданы. Обновление steps невозможно.", level="WARNING", add_to_steps=False)

        self.log(f"Starting Excel Mapping Application run for: {self.pdf_filepath.name}")

        try:
            self.gemini_processor = GeminiProcessor(
                pdf_filepath=self.pdf_filepath,
                model_name=self.model_name
            )
            if self.gemini_processor.pdf_parts is None:
                raise ValueError("Failed to load PDF into Gemini parts.")
            self.log("Gemini Processor initialized successfully.")
        except Exception as e:
            self.log(f"CRITICAL ERROR: Failed to initialize Gemini Processor: {e}", level="ERROR", error=True)
            traceback.print_exc()
            return

        self.log("\nExtracting publication details for PMID...")
        pmid_cache = load_pmid_cache()
        pdf_name = self.pdf_filepath.name

        if pdf_name in pmid_cache:
            pub_details = pmid_cache[pdf_name]
            self.log(f"Loaded publication details for PMID from cache: {pub_details}")
        else:
            try:
                if hasattr(self.gemini_processor, 'extract_publication_details'):
                    pub_details = self.gemini_processor.extract_publication_details()
                else:
                    self.log(
                        "WARNING: GeminiProcessor does not have 'extract_publication_details' method. Skipping PMID extraction.",
                        level="WARNING", error=True)
                    pub_details = {}

                pmid_cache[pdf_name] = pub_details
                save_pmid_cache(pmid_cache)
            except Exception as e:
                self.log(f"  ERROR extracting publication details or PMID: {e}", level="ERROR", error=True)
                traceback.print_exc()
                self.external_pmid = None

        self.external_pmid = PmidExtractor.get_pmid(
            title=pub_details.get("title"),
            author=pub_details.get("first_author_lastname", ""),
            year=pub_details.get("year", "")
        )
        if self.external_pmid:
            self.log(f"  Successfully extracted PMID: {self.external_pmid}")
        else:
            self.log("  WARNING: Could not extract PMID.", level="WARNING")

        if self.external_pmid:
            safe_pmid = re.sub(r'[^\w\-]+', '_', self.external_pmid)
            cache_identifier = f"pmid_{safe_pmid}"
        else:
            pdf_stem = self.pdf_filepath.stem
            safe_stem = re.sub(r'[^\w\-]+', '_', pdf_stem)
            cache_identifier = f"file_{safe_stem}"
            self.log(f"  WARNING: Using PDF filename stem for cache identifier: {cache_identifier}", level="WARNING")

        cache_filepath = CACHE_DIR / f"{cache_identifier}.cache.json"
        raw_answer_cache_filepath = CACHE_DIR / f"{cache_identifier}_raw.json"

        try:
            cache = CacheManager(cache_filepath)
            self.log(f"CacheManager initialized with cache file: {cache_filepath}")
        except Exception as e:
            self.log(f"ERROR initializing CacheManager for {cache_filepath}: {e}", level="ERROR", error=True)
            traceback.print_exc()
            return

        try:
            processed_data_cache = CacheManager(raw_answer_cache_filepath)
            self.log(f"CacheManager initialized with raw answer cache file: {raw_answer_cache_filepath}")
        except Exception as e:
            self.log(f"ERROR initializing CacheManager for {raw_answer_cache_filepath}: {e}", level="ERROR", error=True)
            traceback.print_exc()
            return

        all_mapping_items = self.create_mapping_data()
        if not all_mapping_items:
            self.log("ERROR: No mapping items loaded. Cannot proceed.", level="ERROR", error=True)
            return
        self.log(f"\nLoaded {len(all_mapping_items)} field definitions.")
        mapping_item_dict: Dict[str, MappingItem] = {item.field: item for item in all_mapping_items}

        patient_identifiers = []
        cache_key_patients = "__patient_identifiers_list_v1__"

        self.log("\nGetting patient identifiers (checking cache)...")
        cached_data_str = cache.get(cache_key_patients)
        loaded_from_cache = False

        if cached_data_str is not None:
            self.log("  Cache HIT for patient identifiers.")
            try:
                patient_identifiers_parsed = json.loads(cached_data_str)
                if isinstance(patient_identifiers_parsed, list):
                    patient_identifiers = patient_identifiers_parsed
                    self.log(f"  Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                    loaded_from_cache = True
                else:
                    self.log("  WARNING: Cached data for patient identifiers is not a list. Re-fetching.", level="WARNING", error=True)
            except json.JSONDecodeError:
                self.log("  ERROR: Failed to parse cached patient identifiers JSON. Re-fetching.", level="ERROR", error=True)

        if not loaded_from_cache:
            self.log("  Cache MISS or invalid cache data. Querying Gemini API for patient identifiers...")
            try:
                if hasattr(self.gemini_processor, 'get_patient_identifiers'):
                    fetched_identifiers = self.gemini_processor.get_patient_identifiers()
                else:
                    self.log(
                        "ERROR: GeminiProcessor does not have 'get_patient_identifiers' method. Cannot fetch identifiers.",
                        level="ERROR", error=True)
                    fetched_identifiers = []

                self.log(f"  Successfully fetched {len(fetched_identifiers)} identifiers via Gemini.")

                try:
                    patient_identifiers_json_str = json.dumps(fetched_identifiers)
                    cache.put(cache_key_patients, patient_identifiers_json_str)
                    self.log("  Stored fetched patient identifiers in cache.")
                    patient_identifiers = fetched_identifiers
                except TypeError as json_err:
                    self.log(f"  ERROR: Could not serialize fetched patient identifiers to JSON: {json_err}. Not caching.",
                          level="ERROR", error=True)
                    patient_identifiers = fetched_identifiers
                except Exception as cache_err:
                    self.log(f"  ERROR: Could not save patient identifiers to cache: {cache_err}", level="ERROR", error=True)
                    patient_identifiers = fetched_identifiers

            except Exception as e:
                self.log(f"ERROR getting patient identifiers via Gemini: {e}", level="ERROR", error=True)
                traceback.print_exc()
                patient_identifiers = []

        self.log(f"--> Proceeding with {len(patient_identifiers)} patient identifiers.")

        list_of_patient_question_sets: list[list[QuestionInfo]] = []
        if patient_identifiers:
            self.log(f"\nGenerating questions for {len(patient_identifiers)} unique patients...")
            for entry in patient_identifiers:
                patient_id = entry.get("patient")
                family_id = entry.get("family")
                one_patient_questions: list[QuestionInfo] = []
                for item in all_mapping_items:
                    context_prefix = f"Regarding patient '{patient_id}'"
                    if family_id:
                        context_prefix += f" from family '{family_id}'"
                    specific_query = f"{context_prefix}: {item.question}"
                    try:
                        q_info = QuestionInfo(
                            field=item.field, query=specific_query,
                            response_convertion_strategy=item.response_convertion_strategy,
                            family_id=family_id, patient_id=patient_id
                        )
                        one_patient_questions.append(q_info)
                    except NameError:
                        self.log("ERROR: 'QuestionInfo' class not found. Please ensure it's defined in 'mapping_item.py'.",
                              level="ERROR", error=True)
                        return
                list_of_patient_question_sets.append(one_patient_questions)

            self.log(f"Generated question sets for {len(list_of_patient_question_sets)} patients.")
        else:
            self.log("\nNo patient identifiers found or loaded. Proceeding without patient-specific questions.")

        all_patient_data_rows: list[Dict[str, str]] = []
        collected_headers: Set[str] = {"pmid", "family_id", "individual_id"}

        if not list_of_patient_question_sets:
            self.log("\nNo patient question sets generated. Cannot process patients.")
        else:
            self.log(f"\nProcessing {len(list_of_patient_question_sets)} identified patient sets...")
            patient_num = 0
            for patient_question_set in list_of_patient_question_sets:
                patient_num += 1
                if not patient_question_set:
                    self.log(f"\n=== Skipping Patient Set {patient_num} (Empty Question Set) ===")
                    continue

                current_patient_id = patient_question_set[0].patient_id or "UnknownPatient"
                current_family_id = patient_question_set[0].family_id
                self.log(
                    f"\n=== Processing Patient Set {patient_num} (Patient: '{current_patient_id}', Family: '{current_family_id or 'N/A'}') ===")

                patient_results: Dict[str, str] = {}
                patient_results["pmid"] = self.external_pmid if self.external_pmid else "-99"
                patient_results["family_id"] = current_family_id or "-99"
                patient_results["individual_id"] = current_patient_id

                for q_obj in patient_question_set:
                    current_item = mapping_item_dict.get(q_obj.field)
                    if not current_item:
                        self.log(f"WARNING: No MappingItem found for field '{q_obj.field}'. Skipping query.", level="WARNING")
                        continue

                    self.log(f"--- Querying for field: {current_item.field} (Patient: {current_patient_id}) ---")
                    query_text = q_obj.query

                    cached_answer = cache.get(query_text)

                    raw_answer: Optional[str]

                    if cached_answer is not None:
                        raw_answer = cached_answer
                    else:
                        try:
                            if hasattr(self.gemini_processor, 'answer_question'):
                                raw_answer = self.gemini_processor.answer_question(query_text, with_evidence=True)
                            else:
                                self.log("ERROR: GeminiProcessor does not have 'answer_question' method. Cannot answer.",
                                      level="ERROR", error=True)
                                raw_answer = {"answer": "-99_API_METHOD_MISSING", "evidence": ""}

                            answer_to_cache = raw_answer if raw_answer is not None else ""
                            cache.put(query_text, answer_to_cache)

                        except Exception as api_err:
                            self.log(f"ERROR during Gemini API call for field '{current_item.field}': {api_err}",
                                  level="ERROR", error=True)
                            raw_answer = {"answer": "-99_API_ERROR", "evidence": ""}

                    processor_id = (current_item.custom_processor.__name__
                                    if current_item.custom_processor
                                    else self._default_processor.__name__)
                    cache_key_processed = f"{processor_id}_{raw_answer}"
                    cached_processed = processed_data_cache.get(cache_key_processed)

                    if cached_processed is not None:
                        try:
                            processed_data_single = json.loads(cached_processed)
                        except json.JSONDecodeError as json_err:
                            self.log(
                                f"WARN: Failed to load processed data from cache key '{cache_key_processed}': {json_err}. Reprocessing.",
                                level="WARNING", error=True)
                            cached_processed = None

                    if cached_processed is None:
                        processor = current_item.custom_processor or self._default_processor
                        try:
                            processed_data_single = processor(raw_answer, current_item)
                            try:
                                json_string = json.dumps(processed_data_single)
                                processed_data_cache.put(cache_key_processed, json_string)
                                cached_processed = processed_data_cache.get(cache_key_processed)
                            except TypeError as json_err:
                                self.log(
                                    f"ERROR: Processor '{processor_id}' for field '{current_item.field}' returned non-JSON-serializable data: {json_err}",
                                    level="ERROR", error=True)
                                processed_data_single = {
                                    current_item.field: "PROCESSING_ERROR: Non-serializable output"}

                        except Exception as proc_err:
                            self.log(
                                f"ERROR processing field '{current_item.field}' with processor '{processor_id}' for cache key '{cache_key_processed}': {proc_err}",
                                level="ERROR", error=True)
                            traceback.print_exc()
                            processed_data_single = {current_item.field: f"PROCESSING_ERROR: {proc_err}"}

                    processed_data_single = json.loads(cached_processed)

                    if self.documents and self.document_id:
                        step_id = f"proc_{patient_num}_{list(patient_question_set).index(q_obj)}"
                        title = f"{current_item.field}: {current_patient_id}"
                        answer = processed_data_single.get(current_item.field, "")
                        status = "success" if answer else "error"

                        step = {
                            "id": step_id,
                            "title": title,
                            "answer": answer,
                            "status": status,
                            "type": "processing",
                            "patient_index": patient_num - 1,
                            "question_index": list(patient_question_set).index(q_obj),
                            "field": current_item.field,
                            "question": q_obj.query,
                            "context": raw_answer,
                            "patient_id": current_patient_id,
                            "family_id": current_family_id
                        }
                        self.documents[self.document_id]["steps"].append(step)

                    patient_results.update(processed_data_single)

                    for key in processed_data_single.keys():
                        collected_headers.add(key)

                    if current_item.field not in processed_data_single and current_item.field not in ["motor_symptoms",
                                                                                               "non_motor_symptoms"]:
                        collected_headers.add(current_item.field)

                all_patient_data_rows.append(patient_results)

        self.log("\nDetermining final header order...")
        final_header_order: list[str] = ["pmid", "family_id", "individual_id"]
        for field in COMMON_FIELDS:
            if field != 'pmid' and field in collected_headers and field not in final_header_order:
                final_header_order.append(field)

        defined_fields = sorted([
            item.field for item in all_mapping_items
            if item.field not in final_header_order
               and item.field not in ["motor_symptoms", "non_motor_symptoms"]
               and item.field in collected_headers
        ])
        final_header_order.extend(defined_fields)

        other_headers = sorted([h for h in collected_headers if h not in final_header_order])
        final_header_order.extend(other_headers)

        if not all_patient_data_rows:
            self.log("\nNo data rows were generated. Excel file will not be created.")
        else:
            self.log(f"\nProcessing complete. Found {len(final_header_order)} total headers.")
            self.log(f"Writing {len(all_patient_data_rows)} rows to Excel file: {OUTPUT_EXCEL_PATH}")
            try:
                ExcelWriter.write_all_data(
                    filepath=OUTPUT_EXCEL_PATH,
                    headers=final_header_order,
                    all_data_rows=all_patient_data_rows
                )
                self.log(f"Successfully wrote data to {OUTPUT_EXCEL_PATH}")
            except Exception as write_err:
                self.log(f"ERROR: Failed to write Excel file: {write_err}", level="ERROR", error=True)
                traceback.print_exc()

        if cache:
            cache.save_cache()

        if processed_data_cache:
            processed_data_cache.save_cache()

        self.log("\nApplication finished.")


if __name__ == "__main__":
    print(f"Script starting execution at {Path(__file__).parent.resolve()}")
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable not set. The application might fail.")

    if not PDF_FILEPATH.exists() or not PDF_FILEPATH.is_file():
        print(f"CRITICAL ERROR: Input PDF file not found or invalid: '{PDF_FILEPATH}'", file=sys.stderr)
        sys.exit(1)

    print(f"Creating ExcelMappingApp for PDF: {PDF_FILEPATH}")
    app = ExcelMappingApp(pdf_filepath=PDF_FILEPATH, model_name=GEMINI_MODEL_NAME)
    app.run()
