# excel_mapping_app.py
import sys
import os
import traceback # For detailed error printing
from pathlib import Path
import re
import json # <--- ДОБАВЛЕНО для кэширования patient_identifiers
from typing import List, Dict, Optional, Set, Tuple, Callable # Added Callable


# --- Local Imports ---
# Use the new Gemini Processor
# Ensure gemini_processor.py is in the same directory or Python path
try:
    from gemini_processor import GeminiProcessor
except ImportError:
     print("ERROR: Could not import 'gemini_processor'. Ensure 'gemini_processor.py' exists.", file=sys.stderr)
     sys.exit(1)
# MappingItem structures
# Ensure mapping_item.py is in the same directory or Python path
try:
    from mapping_item import MappingItem, QuestionInfo
except ImportError:
     print("ERROR: Could not import 'mapping_item'. Ensure 'mapping_item.py' exists.", file=sys.stderr)
     sys.exit(1)
# Excel writing utility (assuming excel_writer.py exists and has write_all_data)
try:
    from excel_writer import ExcelWriter
except ImportError:
    print("ERROR: excel_writer.py not found. Please ensure it exists and contains the 'ExcelWriter' class with a 'write_all_data' static method.", file=sys.stderr)
    sys.exit(1)

# PMID extractor for PubMed
try:
    from pmid_extractor import PmidExtractor
except ImportError:
    print("ERROR: Could not import 'pmid_extractor'. Ensure 'pmid_extractor.py' exists in the same directory.", file=sys.stderr)
    sys.exit(1)

# Cache Manager (assuming the version accepting cache_filepath in init)
try:
    from cache_manager import CacheManager
except ImportError:
    print("ERROR: Could not import 'CacheManager'. Ensure 'cache_manager.py' exists in the same directory.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
# PDF File Path (Update as needed)
PDF_FILEPATH = Path(r"./pdf_docs/ando2012-22991136.pdf") # Example from gemini_request_sample.txt

# Output Excel Path
OUTPUT_EXCEL_PATH = Path("./tables/Patient_Data_Gemini_Output.xlsx")

# Gemini Model Name (Update if needed, ensure it matches your access)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Or "gemini-1.5-pro" etc.

# Define which fields are considered "common" (apply to whole study) vs "patient-specific"
# Currently, all fields are processed per patient. Set to empty list if not used.
COMMON_FIELDS = ["pmid", "author, year", "comments_study"]

# --- NEW CONFIGURATION ---
CACHE_DIR = Path("./.gemini_cache") # Directory for cache files

# create cache directory if it doesn't exist
if not CACHE_DIR.exists():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created at: {CACHE_DIR}")
    except Exception as e:
        print(f"Error creating cache directory: {e}", file=sys.stderr)
        sys.exit(1)

class ExcelMappingApp:
    """Orchestrates PDF processing, Gemini querying/formatting, and Excel writing."""

    # --- ИЗМЕНЕНО: Упрощенный __init__ ---
    def __init__(self, pdf_filepath: Path, model_name: str):
        """Initializes application state."""
        self.pdf_filepath = pdf_filepath
        self.model_name = model_name
        self.gemini_processor: Optional[GeminiProcessor] = None
        self.external_pmid: Optional[str] = None # Будет установлен в run
        # CacheManager будет инициализирован локально в run
        print("ExcelMappingApp initialized.")

    # --- Custom Field Processor Implementations ---
    # ... (все ваши процессоры _default_processor, _motor_symptom_processor, _non_motor_symptom_processor и т.д. остаются ЗДЕСЬ без изменений) ...
    def _default_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        """ Default processor: Uses the Gemini formatter for the raw answer. """
        # --- ДОБАВЛЕНА ПРОВЕРКА ---
        if self.gemini_processor is None:
             print("ERROR: Gemini processor not initialized in _default_processor.", file=sys.stderr)
             return {item.field: "-99_PROCESSOR_ERROR"}
        # --- КОНЕЦ ПРОВЕРКИ ---
        # Обратите внимание: format_answer теперь должен быть методом GeminiProcessor
        try:
            formatted_answer = self.gemini_processor.format_answer(raw_answer, item.response_convertion_strategy)
            return {item.field: formatted_answer}
        except Exception as e:
             print(f"ERROR in default_processor formatting field '{item.field}': {e}", file=sys.stderr)
             # traceback.print_exc() # Раскомментируйте для детальной ошибки
             return {item.field: "-99_FORMATTING_ERROR"}

    # --- NEW: Motor Symptom Processor ---
    def _motor_symptom_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        """
        Processor for parsing MOTOR symptoms list like "symptom_name:yes/no;..."
        Generates columns like "motor_symptomname_sympt".
        Parses the RAW answer directly.
        """
        results: Dict[str, str] = {}
        prefix = "motor" # Prefix for motor symptom columns
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported", "information not found"]:
             # print(f"  Motor Symptom Processor: No symptoms found or raw answer empty/negative ('{raw_answer}').")
             return results # Return empty map if no symptoms reported
        # print(f"  Motor Symptom Processor: Processing raw answer: '{raw_answer[:100]}...'")
        symptoms = re.split(r'[;\n]', raw_answer) # Split by semicolon or newline
        found_structured = False

        for symptom_entry in symptoms:
            symptom_entry = symptom_entry.strip()
            if not symptom_entry: continue
            parts = symptom_entry.split(":", 1); # Split only on the first colon
            if len(parts) == 2:
                symptom_name_raw = parts[0].strip();
                if not symptom_name_raw or symptom_name_raw.lower() == "none": continue
                symptom_name = symptom_name_raw.lower();
                symptom_name = re.sub(r'[\s\(\)-]+', '_', symptom_name); symptom_name = re.sub(r'_+', '_', symptom_name)
                symptom_name = re.sub(r'[^\w]+', '', symptom_name); symptom_name = symptom_name.strip('_')
                if not symptom_name: print(f"  -> Motor Symptom WARNING: Name '{symptom_name_raw}' became empty after sanitizing."); continue
                presence = parts[1].strip().lower()
                column_name = f"{prefix}_{symptom_name}_sympt"
                if presence in ["yes", "present", "positive", "true", "1", "y"]: value = "yes"
                elif presence in ["no", "absent", "negative", "false", "0", "n"]: value = "no"
                else: print(f"  -> Motor Symptom WARNING: Unknown presence value '{presence}' for symptom '{symptom_name_raw}'. Setting column '{column_name}' to -99."); value = "-99"
                results[column_name] = value; found_structured = True
            elif len(parts) == 1 and len(symptoms) == 1 and parts[0].lower() in ["none", "n/a", "not applicable", "none reported"]:
                 # print(f"  Motor Symptom Processor: Raw answer indicates absence of symptoms ('{parts[0]}').")
                 found_structured = True; break # No symptoms to add
        # print(f"  Motor Symptom Processor: Generated {len(results)} columns.")
        return results

    # --- NEW: Non-Motor Symptom Processor ---
    def _non_motor_symptom_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        """
        Processor for parsing NON-MOTOR symptoms list like "symptom_name:yes/no;..."
        Generates columns like "nms_symptomname_sympt".
        Parses the RAW answer directly.
        """
        results: Dict[str, str] = {}
        prefix = "nms" # Prefix for non-motor symptom columns
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported", "information not found"]:
             print(f"  NMS Processor: No symptoms found or raw answer empty/negative ('{raw_answer}').")
             return results

        print(f"  NMS Processor: Processing raw answer: '{raw_answer[:100]}...'")
        symptoms = re.split(r'[;\n]', raw_answer)
        found_structured = False

        for symptom_entry in symptoms:
            symptom_entry = symptom_entry.strip()
            if not symptom_entry: continue

            parts = symptom_entry.split(":", 1)
            if len(parts) == 2:
                symptom_name_raw = parts[0].strip()
                if not symptom_name_raw or symptom_name_raw.lower() == "none": continue
                symptom_name = symptom_name_raw.lower()
                symptom_name = re.sub(r'[\s\(\)-]+', '_', symptom_name); symptom_name = re.sub(r'_+', '_', symptom_name)
                symptom_name = re.sub(r'[^\w]+', '', symptom_name); symptom_name = symptom_name.strip('_')
                if not symptom_name: print(f"  -> NMS WARNING: Name '{symptom_name_raw}' became empty after sanitizing."); continue
                presence = parts[1].strip().lower()
                column_name = f"{prefix}_{symptom_name}_sympt"
                if presence in ["yes", "present", "positive", "true", "1", "y"]: value = "yes"
                elif presence in ["no", "absent", "negative", "false", "0", "n"]: value = "no"
                else: print(f"  -> NMS WARNING: Unknown presence value '{presence}' for symptom '{symptom_name_raw}'. Setting column '{column_name}' to -99."); value = "-99"
                results[column_name] = value; found_structured = True
            elif len(parts) == 1 and len(symptoms) == 1 and parts[0].lower() in ["none", "n/a", "not applicable", "none reported"]:
                 # print(f"  NMS Processor: Raw answer indicates absence of symptoms ('{parts[0]}').")
                 found_structured = True; break
        # print(f"  NMS Processor: Generated {len(results)} columns.")
        return results

    def _symptom_list_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        # (код без изменений)
        results: Dict[str, str] = {}
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported", "information not found"]: return results
        symptoms = re.split(r'[;\n]', raw_answer); found_structured = False
        for symptom_entry in symptoms:
            symptom_entry = symptom_entry.strip()
            if not symptom_entry: continue
            parts = symptom_entry.split(":", 1)
            if len(parts) == 2:
                symptom_name_raw = parts[0].strip();
                if not symptom_name_raw: continue
                symptom_name = symptom_name_raw.lower(); symptom_name = re.sub(r'\s+', '_', symptom_name); symptom_name = re.sub(r'[^a-z0-9_]+', '', symptom_name)
                if not symptom_name: print(f"  -> Symptom WARNING: Name '{symptom_name_raw}' became empty after sanitizing."); continue
                presence = parts[1].strip().lower()
                column_name = f"{symptom_name}_sympt"
                if presence in ["yes", "present", "positive", "true", "1", "y"]: value = "yes"
                elif presence in ["no", "absent", "negative", "false", "0", "n"]: value = "no"
                else: print(f"  -> Symptom WARNING: Unknown presence value '{presence}' for symptom '{symptom_name_raw}'. Setting column '{column_name}' to -99."); value = "-99"
                results[column_name] = value; found_structured = True
            elif len(parts) == 1 and len(symptoms) == 1 and parts[0].lower() in ["none", "n/a", "not applicable", "none reported"]: found_structured = True; break
            else: print(f"  -> Symptom WARNING: Could not parse entry like 'name:value': '{symptom_entry}'")
        if not found_structured and raw_answer.strip(): print(f"  -> Symptom WARNING: Parsing yielded no structured results for: '{raw_answer[:50]}...'. Storing raw in '{item.field}'."); results[item.field] = "-99"
        return results

    def _hpo_symptom_processor(self, raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
        """
        Processor for parsing HPO coded symptoms like "headache_HP:0002315:yes;..."
        Generates columns like "HP_0002315". Parses the RAW answer directly.
        """
        # (Implementation remains the same as previous version)
        results: Dict[str, str] = {}
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported", "information not found"]: return results
        hpo_pattern = re.compile(r"(HP[:_ ]?\s?\d{7})[\s:\-\(\)]+(yes|no|present|absent|positive|negative|true|false|1|0)", re.IGNORECASE)
        found_structured = False; matches = hpo_pattern.findall(raw_answer)
        if not matches and raw_answer.strip().lower() in ["none", "n/a", "not applicable", "none reported"]: found_structured = True
        for match in matches:
            hpo_id_raw = match[0]; presence = match[1].lower(); hpo_digits = re.sub(r'[^0-9]', '', hpo_id_raw)
            if len(hpo_digits) == 7: hpo_id_normalized = "HP_" + hpo_digits
            else: print(f"  -> HPO WARNING: Invalid HPO ID format found: '{hpo_id_raw}'. Skipping."); continue
            column_name = hpo_id_normalized; value = "yes" if presence in ["yes", "present", "positive", "true", "1", "y"] else "no"; results[column_name] = value; found_structured = True
        if not found_structured and raw_answer.strip(): print(f"  -> HPO WARNING: HPO parsing yielded no structured results for: '{raw_answer[:50]}...'. Storing raw in '{item.field}'."); results[item.field] = "-99"
        return results

    # --- Field and Question Definitions ---
    def create_mapping_data(self) -> List[MappingItem]:
        """Creates the list of fields, questions, strategies, and custom processors."""
        mapping_data: List[MappingItem] = []

        # --- Assign Processors ---
        # Assign the NEW processors here
        processor_map: Dict[str, Callable[[Optional[str], MappingItem], Dict[str, str]]] = {
            "motor_symptoms": self._motor_symptom_processor,
            "non_motor_symptoms": self._non_motor_symptom_processor,
            # Add other custom processors if needed:
            # "hpo_phenotypes": self._hpo_symptom_processor,
        }

        # --- Define Mapping Items (Populated from user input) ---

        # General study information (Keep these)
        mapping_data.append(MappingItem(
            field="author, year", # PMID is handled externally now
            question="Who is the first author and what is the publication year?",
            mapped_excel_column="Guessed: Author_year",
            response_convertion_strategy="Enter the last name of the first author, followed by a comma and the four-digit publication year (e.g., 'Smith, 2018'). If not found, enter -99.",
            custom_processor=processor_map.get("author, year")
        ))

        mapping_data.append(MappingItem(
            field="comments_study",
            question="Any general comments about the study?",
            mapped_excel_column="Guessed: Study Comments",
            response_convertion_strategy="Enter any overarching comments regarding the study. For example, if the authors were contacted for missing information, record the date of contact and any relevant response in this field. If none, enter -99.",
            custom_processor=processor_map.get("comments_study")
        ))

        # Family information
        mapping_data.append(MappingItem(
            field="family_id",
            question="What is the family ID?",
            mapped_excel_column="Guessed: Family ID",
            response_convertion_strategy="Enter the family identifier (family ID) exactly as reported in the publication. If the publication does not specify a family ID (e.g., sporadic case), assign a unique family ID (such as 'Family 1') for each distinct family. **Note:** The combination of PMID, family_ID, and individual_ID must be unique for each patient. If not applicable or found, enter -99.",
            custom_processor=processor_map.get("family_id")
        ))

        mapping_data.append(MappingItem(
            field="individual_id",
            question="What is the individual ID within the family?",
            mapped_excel_column="Guessed: Individual ID",
            response_convertion_strategy="Enter the individual's identifier within the family as given in the publication (e.g., II-1, Patient 3). If not provided, assign an individual ID (e.g., '1' or 'I-1') ensuring it is unique within the family. If not found, enter -99.",
            custom_processor=processor_map.get("individual_id")
        ))

        mapping_data.append(MappingItem(
            field="consanguinity",
            question="Are the parents consanguineous?",
            mapped_excel_column="Guessed: Consanguinity",
            response_convertion_strategy="Enter 'yes' if the patient's parents are consanguineous (blood relatives), 'no' if they are not consanguineous. If the publication does not provide this information, enter -99 (unknown).",
            custom_processor=processor_map.get("consanguinity")
        ))

        mapping_data.append(MappingItem(
            field="family_history",
            question="Is there a family history of the disease?",
            mapped_excel_column="Guessed: Family History",
            response_convertion_strategy="Enter 'yes' if the individual has a known family history of the disease (i.e., other affected family members), 'no' if the individual is a sporadic case with no known affected relatives. If this information is not stated, enter -99.",
            custom_processor=processor_map.get("family_history")
        ))

        mapping_data.append(MappingItem(
            field="num_wildtype_affected",
            question="Number of mutation-negative but affected family members?",
            mapped_excel_column="Guessed: #Wildtype Affected",
            response_convertion_strategy="Enter the number of mutation-negative (wild-type) family members who are affected by the disease (phenocopies). If none or not applicable, enter 0. If the publication does not report this, enter -99.",
            custom_processor=processor_map.get("num_wildtype_affected")
        ))

        mapping_data.append(MappingItem(
            field="num_wildtype_unaffected",
            question="Number of mutation-negative and unaffected family members?",
            mapped_excel_column="Guessed: #Wildtype Unaffected",
            response_convertion_strategy="Enter the number of mutation-negative (wild-type) family members who are unaffected (no disease signs). If none or not applicable, enter 0. If this information is not provided, enter -99.",
            custom_processor=processor_map.get("num_wildtype_unaffected")
        ))

        # Patient demographic data
        mapping_data.append(MappingItem(
            field="sex",
            question="What is the sex of the individual?",
            mapped_excel_column="Guessed: Sex",
            response_convertion_strategy="Enter 'M' for male or 'F' for female, as reported in the publication. If the individual's sex is not reported, enter -99.",
            custom_processor=processor_map.get("sex")
        ))

        mapping_data.append(MappingItem(
            field="aao",
            question="What was the age at onset of symptoms? (Enter -99 if unknown)",
            mapped_excel_column="Guessed: AAO",
            response_convertion_strategy="Enter the age at onset (AAO) of symptoms, in years, as reported in the publication. If the age at onset is not given, enter -99.",
            custom_processor=processor_map.get("aao")
        ))

        mapping_data.append(MappingItem(
            field="age",
            question="What is the current age or age at last examination?",
            mapped_excel_column="Guessed: Age",
            response_convertion_strategy="Enter the age at last clinical evaluation or the age at death, as reported. If the patient was alive at last follow-up, use the age at last examination; if the patient is deceased, use the age at death. If this age is not reported, enter -99.",
            custom_processor=processor_map.get("age")
        ))

        mapping_data.append(MappingItem(
            field="alive",
            question="Was the individual alive at last follow-up? (Y/N)",
            mapped_excel_column="Guessed: Alive",
            response_convertion_strategy="Enter 'yes' if the individual was alive at the last reported follow-up, 'no' if the individual was deceased by the last follow-up. If the survival status is not mentioned in the article, enter -99.",
            custom_processor=processor_map.get("alive")
        ))

        mapping_data.append(MappingItem(
            field="disease_duration",
            question="What was the disease duration in years?",
            mapped_excel_column="Guessed: Disease Duration",
            response_convertion_strategy="Enter the disease duration in years, defined as the time from symptom onset to the last evaluation or death, as reported in the publication. If the duration is not explicitly reported, you may calculate it from the age at onset and age at last exam (if both available); if it cannot be determined, enter -99.",
            custom_processor=processor_map.get("disease_duration")
        ))

        # Clinical signs and symptoms
        mapping_data.append(MappingItem(
            field="parkinsonism_sympt",
            question="Did the patient have parkinsonism?",
            mapped_excel_column="Guessed: Parkinsonism",
            response_convertion_strategy="Enter 'yes' if parkinsonism (the presence of Parkinson’s disease–like motor features) is reported for the individual, 'no' if the article explicitly states that parkinsonism is absent. If not mentioned (for a non-Parkinson disease case), enter -99. **Note:** For patients in a Parkinsonism (PARK) study, parkinsonism is inherently present by inclusion criteria.",
            custom_processor=processor_map.get("parkinsonism_sympt")
        ))

        mapping_data.append(MappingItem(
            field="NMS_park_sympt",
            question="Any non-motor symptoms associated with parkinsonism?",
            mapped_excel_column="Guessed: NMS",
            response_convertion_strategy="Enter 'yes' if at least one non-motor symptom associated with Parkinsonism is reported. If the publication states that no non-motor symptoms (NMS) were present, enter 'no'. If only some NMS are noted as absent but it’s unclear if all were assessed, classify NMS_park_sympt as -99.",
            custom_processor=processor_map.get("NMS_park_sympt")
        ))

        mapping_data.append(MappingItem(
            field="autonomic_sympt",
            question="Any autonomic symptoms present?",
            mapped_excel_column="Guessed: Autonomic",
            response_convertion_strategy="Enter 'yes' if any autonomic symptom occurred, 'no' if the paper explicitly states that **all** autonomic symptoms were absent. (The absence of one or two specific autonomic symptoms does not qualify as 'no' unless all common autonomic symptoms are reported absent.) If the presence of autonomic symptoms is not addressed or only partially addressed, enter -99.",
            custom_processor=processor_map.get("autonomic_sympt")
        ))

        mapping_data.append(MappingItem(
            field="depression_sympt",
            question="Was depression present?",
            mapped_excel_column="Guessed: Depression",
            response_convertion_strategy="Enter 'yes' if depression or depressive symptoms are reported as present in the individual. Enter 'no' if the article states the individual had no depression. If a depression rating scale is provided without a clear statement of presence/absence, use the recommended cut-off for that scale to infer whether depression is present ('yes') or not ('no'). If no information on depression is available, enter -99.",
            custom_processor=processor_map.get("depression_sympt")
        ))

        mapping_data.append(MappingItem(
            field="dementia_sympt",
            question="Did the patient have dementia or significant cognitive impairment?",
            mapped_excel_column="Guessed: Dementia",
            response_convertion_strategy="Enter 'yes' if dementia or significant cognitive impairment is reported, 'no' if the article explicitly notes the absence of dementia/cognitive decline. If cognitive status is not mentioned, enter -99.",
            custom_processor=processor_map.get("dementia_sympt")
        ))

        # --- NEW: Add items to trigger the symptom processors ---
        mapping_data.append(MappingItem(
            field="motor_symptoms", # This field name triggers the processor
            question="List all *motor* symptoms observed in the patient and state if they were present (yes/no). Format as 'Symptom Name: yes/no', separated by semicolons or newlines. Examples: Rigidity: yes; Tremor: no; Bradykinesia: yes.",
            mapped_excel_column="Dynamic Motor Symptom Columns", # Placeholder, actual columns generated by processor
            response_convertion_strategy="Provide a list of motor symptoms and their status (e.g., 'Rigidity: yes; Tremor: no'). If none are mentioned or present, state 'None'.",
            custom_processor=processor_map.get("motor_symptoms") # Assign the motor processor
        ))
        mapping_data.append(MappingItem(
            field="non_motor_symptoms", # This field name triggers the processor
            question="List all *non-motor* symptoms (like cognitive, autonomic, psychiatric, sleep, sensory) observed in the patient and state if they were present (yes/no). Format as 'Symptom Name: yes/no', separated by semicolons or newlines. Examples: Depression: yes; Anosmia: no; RBD: yes; Constipation: yes.",
            mapped_excel_column="Dynamic NMS Columns", # Placeholder
            response_convertion_strategy="Provide a list of non-motor symptoms and their status (e.g., 'Depression: yes; Anosmia: no'). If none are mentioned or present, state 'None'.",
            custom_processor=processor_map.get("non_motor_symptoms") # Assign the NMS processor
        ))

        # --- Genetic info (standard) ---
        mapping_data.append(MappingItem(
            field="gene1",
            question="What is the first gene with a pathogenic variant found in the patient?",
            mapped_excel_column="Guessed: Gene1",
            response_convertion_strategy="Enter the official HGNC-approved gene symbol for the first pathogenic variant reported. If none found or not applicable, enter -99.",
            custom_processor=processor_map.get("gene1")
        ))

        mapping_data.append(MappingItem(
            field="gene2",
            question="What is the second gene with a pathogenic variant found in the patient (if any)?",
            mapped_excel_column="Guessed: Gene2",
            response_convertion_strategy="If applicable, enter the official HGNC-approved gene symbol for the second pathogenic variant. Leave blank or enter -99 if not applicable.",
             custom_processor=processor_map.get("gene2")
       ))

        mapping_data.append(MappingItem(
            field="gene3",
            question="What is the third gene with a pathogenic variant found in the patient (if any)?",
            mapped_excel_column="Guessed: Gene3",
            response_convertion_strategy="If applicable, enter the official HGNC-approved gene symbol for the third pathogenic variant. Leave blank or enter -99 if not applicable.",
             custom_processor=processor_map.get("gene3")
       ))

        mapping_data.append(MappingItem(
            field="mut1_c",
            question="What is the DNA-level mutation? (Primary mutation)",
            mapped_excel_column="Guessed: Mut1 cDNA",
            response_convertion_strategy="Enter the nucleotide change of the first (primary) mutation as reported in the article, using HGVS coding DNA nomenclature (e.g., c.511C>T). Include the transcript reference or genomic build if provided. If the notation is given in a non-standard format, record it as provided (it will be harmonized later). If no mutation is reported, enter -99.",
            custom_processor=processor_map.get("mut1_c")
        ))

        mapping_data.append(MappingItem(
            field="mut1_p",
            question="What is the protein-level effect of the mutation?",
            mapped_excel_column="Guessed: Mut1 Protein",
            response_convertion_strategy="Enter the protein change (amino acid level) corresponding to the first mutation, using HGVS protein nomenclature and the three-letter amino acid code (e.g., p.Gln171* for a glutamine to stop codon change). Use the notation as given in the manuscript (it will be adjusted to HGVS format if needed). If no mutation reported or applicable, enter -99.",
            custom_processor=processor_map.get("mut1_p")
        ))

        mapping_data.append(MappingItem(
            field="mut1_zygosity",
            question="What is the zygosity of the first mutation?",
            mapped_excel_column="Guessed: Mut1 Zygosity",
            response_convertion_strategy="Enter the zygosity of the first mutation as reported: use 'heterozygous' if one allele is mutated, 'homozygous' if both alleles carry the mutation, or 'hemizygous' if the mutation is on an X-chromosome in a male. If zygosity is not stated, infer from context (e.g., in an autosomal recessive gene, a single reported mutation likely implies a homozygous state if no second mutation is mentioned) or enter -99 if uncertain.",
            custom_processor=processor_map.get("mut1_zygosity")
        ))

        mapping_data.append(MappingItem(
            field="mut2_c",
            question="Second mutation (cDNA), if any?",
            mapped_excel_column="Guessed: Mut2 cDNA",
            response_convertion_strategy="If the individual carries a second pathogenic mutation (e.g., in autosomal recessive cases or compound heterozygosity), enter the nucleotide change for the second mutation (e.g., c.1234G>A) as reported. If there is no second mutation reported, leave this blank or enter -99.",
            custom_processor=processor_map.get("mut2_c")
        ))

        mapping_data.append(MappingItem(
            field="mut2_p",
            question="Second mutation protein change?",
            mapped_excel_column="Guessed: Mut2 Protein",
            response_convertion_strategy="Enter the protein-level change for the second mutation (e.g., p.Arg412His) using the three-letter amino acid code, if a second mutation is present. If no second mutation, leave blank or enter -99.",
            custom_processor=processor_map.get("mut2_p")
        ))

        mapping_data.append(MappingItem(
            field="mut2_zygosity",
            question="Zygosity of second mutation?",
            mapped_excel_column="Guessed: Mut2 Zygosity",
            response_convertion_strategy="Enter the zygosity of the second mutation (usually 'heterozygous' for a compound heterozygous case). If not applicable or not stated, leave blank or -99.",
            custom_processor=processor_map.get("mut2_zygosity")
        ))

        mapping_data.append(MappingItem(
            field="mut3_c",
            question="Third mutation (cDNA), if any?",
            mapped_excel_column="Guessed: Mut3 cDNA",
            response_convertion_strategy="If a third mutation is reported (in rare cases, e.g., tri-allelic combinations), enter the cDNA notation for that mutation. If not applicable, leave blank or enter -99.",
            custom_processor=processor_map.get("mut3_c")
        ))

        mapping_data.append(MappingItem(
            field="mut3_p",
            question="Third mutation protein change?",
            mapped_excel_column="Guessed: Mut3 Protein",
            response_convertion_strategy="Enter the protein change for the third mutation if applicable, or leave blank/enter -99 if no third mutation.",
            custom_processor=processor_map.get("mut3_p")
        ))

        mapping_data.append(MappingItem(
            field="mut3_zygosity",
            question="Zygosity of third mutation?",
            mapped_excel_column="Guessed: Mut3 Zygosity",
            response_convertion_strategy="Enter the zygosity of the third mutation if applicable. If not applicable, leave blank or enter -99.",
            custom_processor=processor_map.get("mut3_zygosity")
        ))

        mapping_data.append(MappingItem(
            field="genome_build",
            question="Which genome build/transcript was used?",
            mapped_excel_column="Guessed: Genome Build",
            response_convertion_strategy="Enter any reference genome build or transcript ID mentioned in the publication that was used for reporting the mutation (e.g., hg19, GRCh37, NM_000041.3). Include terms like 'build', 'hg', 'GRCh', or transcript accession if provided. If no genome build or reference sequence is indicated, enter -99.",
            custom_processor=processor_map.get("genome_build")
        ))

        mapping_data.append(MappingItem(
            field="physical_location",
            question="What is the physical genomic location of the mutation?",
            mapped_excel_column="Guessed: Physical Location",
            response_convertion_strategy="Enter the chromosomal coordinate of the mutation if it is provided or can be unequivocally determined from the publication, in the format 'chromosome:position' (e.g., 12:123456). **Note:** The physical location refers to the plus strand; for deletions/duplications/insertions, use the coordinate of the base immediately 5′ to the variant’s start. If the genomic position is not given or cannot be derived, enter -99.",
            custom_processor=processor_map.get("physical_location")
        ))

        # --- Decision and comments (standard) ---
        mapping_data.append(MappingItem(
            field="mdsgene_decision",
            question="Include this patient in MDSGene?",
            mapped_excel_column="Guessed: MDSGene Decision",
            response_convertion_strategy="Set mdsgene_decision to IN for all eligible patients who meet inclusion criteria and are included in MDSGene. If a patient is not included (e.g., does not fulfill criteria or is listed only for reference), set mdsgene_decision to EX. For excluded patients, ensure to provide the reason for exclusion in the patient’s comments. If unclear, enter -99.",
             custom_processor=processor_map.get("mdsgene_decision")
       ))

        mapping_data.append(MappingItem(
            field="comments_pat",
            question="Any additional comments about this patient?",
            mapped_excel_column="Guessed: Patient Comments",
            response_convertion_strategy="Enter any patient-specific comments or clarifications here. If the patient is excluded (mdsgene_decision = EX), include the reason for exclusion in this field so that the rationale is clear upon review. Otherwise, use this field for any notable details about the individual not captured by other variables. If none, enter -99.",
            custom_processor=processor_map.get("comments_pat")
        ))

        # --- Initial Symptoms (standard) ---
        mapping_data.append(MappingItem(
            field="initial_sympt1",
            question="What was the initial symptom of the patient?",
            mapped_excel_column="Guessed: initial_sympt1",
            response_convertion_strategy="Enter the primary initial symptom as explicitly reported in the publication (e.g., rigidity, tremor). If not reported clearly, enter -99.",
            custom_processor=processor_map.get("initial_sympt1")
        ))

        mapping_data.append(MappingItem(
            field="initial_sympt2",
            question="What was the second initial symptom of the patient (if applicable)?",
            mapped_excel_column="Guessed: initial_sympt2",
            response_convertion_strategy="If a second initial symptom is explicitly reported, enter it here (use symptom categories identical to '_sympt' variables, e.g., 'rigidity_sympt'). Leave blank or -99 if not applicable.",
            custom_processor=processor_map.get("initial_sympt2")
        ))

        mapping_data.append(MappingItem(
            field="initial_sympt3",
            question="What was the third initial symptom of the patient (if applicable)?",
            mapped_excel_column="Guessed: initial_sympt3",
            response_convertion_strategy="If a third initial symptom is explicitly reported, enter it here (use symptom categories identical to '_sympt' variables, e.g., 'rigidity_sympt'). Leave blank or -99 if not applicable.",
            custom_processor=processor_map.get("initial_sympt3")
        ))

        # --- End of user-provided MappingItems ---

        print(f"Created {len(mapping_data)} mapping items.")
        # Optional: Check if all processors were correctly assigned or defaulted
        # for item in mapping_data:
        #     processor_name = item.custom_processor.__name__ if item.custom_processor else "_default_processor (used)"
        #     print(f"  - Field '{item.field}' -> Processor: {processor_name}")
        return mapping_data

    # --- Main Execution Logic (remains the same as previous version) ---
    # --- Main Execution Logic (Modified) ---
    def run(self):
        """Runs the main application workflow using Gemini with PMID-based Caching."""
        print(f"Starting Excel Mapping Application run for: {self.pdf_filepath.name}")

        # --- File Path Check ---
        # Перенесено в __main__ или внешний вызов

        # --- Инициализация Gemini Processor ---
        # (Перенесено в __init__, но добавим проверку)
        try:
            self.gemini_processor = GeminiProcessor(
                pdf_filepath=self.pdf_filepath,
                model_name=self.model_name
            )
            if self.gemini_processor.pdf_parts is None:
                 raise ValueError("Failed to load PDF into Gemini parts.")
            print("Gemini Processor initialized successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize Gemini Processor: {e}", file=sys.stderr)
            traceback.print_exc()
            return # Не можем продолжать без процессора

        # --- Шаг 1: Извлечение деталей публикации и PMID ---
        print("\nExtracting publication details for PMID...")
        try:
            # !!! Убедитесь, что у GeminiProcessor есть этот метод !!!
            # Если нет, возможно, его нужно реализовать или использовать другой подход
            if hasattr(self.gemini_processor, 'extract_publication_details'):
                pub_details = self.gemini_processor.extract_publication_details()
            else:
                 print("WARNING: GeminiProcessor does not have 'extract_publication_details' method. Skipping PMID extraction.", file=sys.stderr)
                 pub_details = {} # Пустой словарь, чтобы не было ошибки ниже

             # Извлекаем PMID с помощью PmidExtractor
            self.external_pmid = PmidExtractor.get_pmid(
                title=pub_details.get("title"),
                author=pub_details.get("first_author_lastname", ""),
                year=pub_details.get("year", "")
            )
            if self.external_pmid:
                 print(f"  Successfully extracted PMID: {self.external_pmid}")
            else:
                 print("  WARNING: Could not extract PMID.")
        except Exception as e:
             print(f"  ERROR extracting publication details or PMID: {e}", file=sys.stderr)
             traceback.print_exc() # Добавим traceback для отладки ошибки извлечения
             self.external_pmid = None # Убедимся, что None в случае ошибки

        # --- Шаг 2: Определение идентификатора и пути для кэша ---
        if self.external_pmid:
             # Используем PMID, если он есть (очищаем от потенциально невалидных символов)
             safe_pmid = re.sub(r'[^\w\-]+', '_', self.external_pmid)
             cache_identifier = f"pmid_{safe_pmid}"
        else:
             # Используем имя файла PDF как запасной вариант
             pdf_stem = self.pdf_filepath.stem
             safe_stem = re.sub(r'[^\w\-]+', '_', pdf_stem) # Очищаем имя файла
             cache_identifier = f"file_{safe_stem}"
             print(f"  WARNING: Using PDF filename stem for cache identifier: {cache_identifier}")

        # Формируем полный путь к файлу кэша в поддиректории
        cache_filepath = CACHE_DIR / f"{cache_identifier}.cache.json"

        # --- Шаг 3: Инициализация CacheManager ---
        # Теперь CacheManager инициализируется здесь, используя определенный путь
        # Предполагаем, что CacheManager принимает путь в __init__ и загружает кэш
        try:
            cache = CacheManager(cache_filepath)
        except Exception as e:
            print(f"ERROR initializing CacheManager for {cache_filepath}: {e}", file=sys.stderr)
            traceback.print_exc()
            return # Не можем продолжать без кэша

        # --- Load Mapping Definitions ---
        all_mapping_items = self.create_mapping_data()
        if not all_mapping_items:
            print("ERROR: No mapping items loaded. Cannot proceed.", file=sys.stderr)
            return
        print(f"\nLoaded {len(all_mapping_items)} field definitions.")
        mapping_item_dict: Dict[str, MappingItem] = {item.field: item for item in all_mapping_items}

        # --- Get Patient Identifiers (с проверкой кэша) ---
        patient_identifiers = [] # Значение по умолчанию
        # Уникальный ключ для кэширования списка пациентов
        cache_key_patients = "__patient_identifiers_list_v1__"

        print("\nGetting patient identifiers (checking cache)...")
        cached_data_str = cache.get(cache_key_patients)
        loaded_from_cache = False

        if cached_data_str is not None:
            print("  Cache HIT for patient identifiers.")
            try:
                # Пытаемся распарсить JSON строку из кэша
                patient_identifiers_parsed = json.loads(cached_data_str)
                # Проверяем, что это действительно список
                if isinstance(patient_identifiers_parsed, list):
                    patient_identifiers = patient_identifiers_parsed # Используем данные из кэша
                    print(f"  Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                    loaded_from_cache = True
                else:
                    print("  WARNING: Cached data for patient identifiers is not a list. Re-fetching.", file=sys.stderr)
                    # Оставляем patient_identifiers пустым списком
            except json.JSONDecodeError:
                print("  ERROR: Failed to parse cached patient identifiers JSON. Re-fetching.", file=sys.stderr)
                # Оставляем patient_identifiers пустым списком

        # Если из кэша не загрузили (miss или ошибка парсинга), вызываем API
        if not loaded_from_cache:
            print("  Cache MISS or invalid cache data. Querying Gemini API for patient identifiers...")
            try:
                 # --- Вызов API ---
                 # !!! Убедитесь, что у GeminiProcessor есть этот метод !!!
                 if hasattr(self.gemini_processor, 'get_patient_identifiers'):
                     fetched_identifiers = self.gemini_processor.get_patient_identifiers()
                 else:
                      print("ERROR: GeminiProcessor does not have 'get_patient_identifiers' method. Cannot fetch identifiers.", file=sys.stderr)
                      fetched_identifiers = [] # Пустой список, если метода нет

                 print(f"  Successfully fetched {len(fetched_identifiers)} identifiers via Gemini.")

                 # --- Сохранение в кэш ---
                 try:
                     patient_identifiers_json_str = json.dumps(fetched_identifiers)
                     cache.put(cache_key_patients, patient_identifiers_json_str)
                     print("  Stored fetched patient identifiers in cache.")
                     patient_identifiers = fetched_identifiers # Используем свежие данные
                 except TypeError as json_err:
                     print(f"  ERROR: Could not serialize fetched patient identifiers to JSON: {json_err}. Not caching.", file=sys.stderr)
                     patient_identifiers = fetched_identifiers # Используем свежие данные, но не кэшируем
                 except Exception as cache_err:
                      print(f"  ERROR: Could not save patient identifiers to cache: {cache_err}", file=sys.stderr)
                      patient_identifiers = fetched_identifiers # Используем свежие данные

            except Exception as e:
                    # Обработка ошибки вызова self.gemini_processor.get_patient_identifiers()
                print(f"ERROR getting patient identifiers via Gemini: {e}", file=sys.stderr)
                traceback.print_exc()
                patient_identifiers = [] # Оставляем пустым списком при ошибке API

        # На этом этапе patient_identifiers содержит либо данные из кэша, либо свежие данные от API,
        # либо пустой список в случае ошибок.
        print(f"--> Proceeding with {len(patient_identifiers)} patient identifiers.")

        # --- Generate Question Sets ---
        list_of_patient_question_sets: List[List[QuestionInfo]] = []
        if patient_identifiers:
             print(f"\nGenerating questions for {len(patient_identifiers)} unique patients...")
             for entry in patient_identifiers:
                 patient_id = entry.get("patient")
                 family_id = entry.get("family")
                 one_patient_questions: List[QuestionInfo] = []
                 for item in all_mapping_items: # Includes the new symptom questions
                     context_prefix = f"Regarding patient '{patient_id}'"
                     if family_id:
                         context_prefix += f" from family '{family_id}'"
                     specific_query = f"{context_prefix}: {item.question}"
                     # --- ВАЖНО: Убедитесь, что QuestionInfo существует в mapping_item.py ---
                     try:
                         q_info = QuestionInfo(
                             field=item.field, query=specific_query,
                             response_convertion_strategy=item.response_convertion_strategy,
                             family_id=family_id, patient_id=patient_id
                         )
                         one_patient_questions.append(q_info)
                     except NameError:
                          print("ERROR: 'QuestionInfo' class not found. Please ensure it's defined in 'mapping_item.py'.", file=sys.stderr)
                          return # Не можем продолжать без QuestionInfo
                 list_of_patient_question_sets.append(one_patient_questions)

             print(f"Generated question sets for {len(list_of_patient_question_sets)} patients.")
        else:
             print("\nNo patient identifiers found or loaded. Proceeding without patient-specific questions.")


        # --- Data Collection & Header Management ---
        all_patient_data_rows: List[Dict[str, str]] = []
        # Используем PMID из self.external_pmid для заголовка
        collected_headers: Set[str] = { "pmid", "family_id", "individual_id" } # Добавим основные идентификаторы сразу

        # --- Process Patients ---
        if not list_of_patient_question_sets:
            print("\nNo patient question sets generated. Cannot process patients.")
        else:
            print(f"\nProcessing {len(list_of_patient_question_sets)} identified patient sets...")
            patient_num = 0
            for patient_question_set in list_of_patient_question_sets: # Contains symptom questions now
                patient_num += 1
                if not patient_question_set:
                    print(f"\n=== Skipping Patient Set {patient_num} (Empty Question Set) ===")
                    continue

                # Получаем ID из первого объекта QuestionInfo (они должны быть одинаковы для одного набора)
                current_patient_id = patient_question_set[0].patient_id or "UnknownPatient"
                current_family_id = patient_question_set[0].family_id
                print(f"\n=== Processing Patient Set {patient_num} (Patient: '{current_patient_id}', Family: '{current_family_id or 'N/A'}') ===")

                patient_results: Dict[str, str] = {}
                # Используем извлеченный или заданный PMID
                patient_results["pmid"] = self.external_pmid if self.external_pmid else "-99"
                patient_results["family_id"] = current_family_id or "-99"
                patient_results["individual_id"] = current_patient_id
                # Заголовки уже добавлены выше

                for q_obj in patient_question_set: # Process all questions including symptom ones
                    current_item = mapping_item_dict.get(q_obj.field)
                    if not current_item:
                        print(f"WARNING: No MappingItem found for field '{q_obj.field}'. Skipping query.")
                        continue

                    print(f"--- Querying for field: {current_item.field} (Patient: {current_patient_id}) ---")
                    query_text = q_obj.query

                    # --- ПРОВЕРКА КЭША через CacheManager ---
                    cached_answer = cache.get(query_text) # Попытка получить из кэша

                    raw_answer : Optional[str] # Объявляем тип для ясности

                    if cached_answer is not None:
                        raw_answer = cached_answer
                        # Уменьшим количество вывода в консоль для ускорения
                        # print(f"    Cache HIT. Using cached answer: {raw_answer[:60] if raw_answer else 'None'}...")
                    else:
                        # --- Запрос к API, если в кэше нет ---
                        # print(f"    Cache MISS. Querying Gemini API...")
                        try:
                            # !!! Убедитесь, что у GeminiProcessor есть этот метод !!!
                            if hasattr(self.gemini_processor, 'answer_question'):
                                 raw_answer = self.gemini_processor.answer_question(query_text)
                            else:
                                 print(f"ERROR: GeminiProcessor does not have 'answer_question' method. Cannot answer.", file=sys.stderr)
                                 raw_answer = "-99_API_METHOD_MISSING"

                            # print(f"    Raw Gemini Answer: {raw_answer[:100] if raw_answer else 'None'}...")
                            # --- СОХРАНИТЬ НОВЫЙ ОТВЕТ В КЭШ ---
                            answer_to_cache = raw_answer if raw_answer is not None else ""
                            cache.put(query_text, answer_to_cache)

                        except Exception as api_err:
                             print(f"ERROR during Gemini API call for field '{current_item.field}': {api_err}", file=sys.stderr)
                             # traceback.print_exc() # Раскомментируйте для детальной ошибки
                             raw_answer = "-99_API_ERROR" # Значение при ошибке API

                    # --- Обработка ответа ---
                    processor = current_item.custom_processor or self._default_processor
                    try:
                         processed_data = processor(raw_answer, current_item)
                    except Exception as proc_err:
                         print(f"ERROR processing field '{current_item.field}' with processor '{processor.__name__}': {proc_err}", file=sys.stderr)
                         traceback.print_exc()
                         processed_data = {current_item.field: f"PROCESSING_ERROR: {proc_err}"}

                    # print(f"    Processed Data: {processed_data}")
                    patient_results.update(processed_data) # Merge the results

                    # Add all generated keys (dynamic columns) to headers
                    for key in processed_data.keys():
                        collected_headers.add(key) # Просто добавляем в set

                    # Ensure the base field name from the mapping item is also added if not dynamic and not already covered
                    if current_item.field not in processed_data and current_item.field not in ["motor_symptoms", "non_motor_symptoms"]: # Не добавляем "виртуальные" поля
                         collected_headers.add(current_item.field)


                    # print("--- Finished query ---") # Уменьшим вывод
                all_patient_data_rows.append(patient_results)

        # --- Define Final Header Order ---
        print("\nDetermining final header order...")
        # Базовые заголовки в начале
        final_header_order: List[str] = ["pmid", "family_id", "individual_id"]
        # Добавляем общие поля (если они есть и еще не добавлены)
        for field in COMMON_FIELDS:
             if field != 'pmid' and field in collected_headers and field not in final_header_order:
                final_header_order.append(field)

        # Добавляем остальные поля из mapping_items (не динамические), которых еще нет, отсортированные
        defined_fields = sorted([
            item.field for item in all_mapping_items
            if item.field not in final_header_order
            and item.field not in ["motor_symptoms", "non_motor_symptoms"] # Исключаем "виртуальные" поля
            and item.field in collected_headers # Только те, что реально были собраны
        ])
        final_header_order.extend(defined_fields)

        # Добавляем все остальные собранные заголовки (вероятно, динамические симптомы), отсортированные
        other_headers = sorted([h for h in collected_headers if h not in final_header_order])
        final_header_order.extend(other_headers)

        # --- Write ALL data to Excel AFTER processing ---
        if not all_patient_data_rows:
            print("\nNo data rows were generated. Excel file will not be created.")
        else:
            print(f"\nProcessing complete. Found {len(final_header_order)} total headers.")
            # print(f"Final Headers Order: {final_header_order}") # Можно раскомментировать для отладки
            print(f"Writing {len(all_patient_data_rows)} rows to Excel file: {OUTPUT_EXCEL_PATH}")
            try:
                ExcelWriter.write_all_data(
                    filepath=OUTPUT_EXCEL_PATH,
                    headers=final_header_order,
                    all_data_rows=all_patient_data_rows
                )
                print(f"Successfully wrote data to {OUTPUT_EXCEL_PATH}")
            except Exception as write_err:
                 print(f"ERROR: Failed to write Excel file: {write_err}", file=sys.stderr)
                 traceback.print_exc()

        # --- СОХРАНЕНИЕ КЭША в конце ---
        if cache: # Убедимся, что кэш был инициализирован
             cache.save_cache() # CacheManager сам проверит, были ли обновления

        print("\nApplication finished.")


# --- Entry Point ---
if __name__ == "__main__":
    print(f"Script starting execution at {Path(__file__).parent.resolve()}") # Покажем рабочую директорию
    # Проверка API ключа
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable not set. The application might fail.")

    # Проверка существования PDF
    if not PDF_FILEPATH.exists() or not PDF_FILEPATH.is_file():
        print(f"CRITICAL ERROR: Input PDF file not found or invalid: '{PDF_FILEPATH}'", file=sys.stderr)
        sys.exit(1) # Выход, если PDF не найден

    # Удаление старого файла Excel (опционально)
    if OUTPUT_EXCEL_PATH.exists():
        print(f"Removing existing output file: {OUTPUT_EXCEL_PATH}")
        try:
            OUTPUT_EXCEL_PATH.unlink()
        except OSError as e:
            print(f"Could not remove existing file: {e}. Please close it if it's open before running.", file=sys.stderr)
            # Решаем, прерывать ли выполнение - пока не будем
            # sys.exit(1)

    # --- Создание экземпляра и запуск ---
    print(f"Creating ExcelMappingApp for PDF: {PDF_FILEPATH}")
    app = ExcelMappingApp(pdf_filepath=PDF_FILEPATH, model_name=GEMINI_MODEL_NAME)
    app.run()