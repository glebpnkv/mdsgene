import hashlib
import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from mdsgene.agents.base_agent import BaseAgent, CACHE_DIR
from mdsgene.cache_utils import load_formatted_result, save_formatted_result
from mdsgene.logging_config import configure_logging
from mdsgene.mapping_item import MappingItem, QuestionInfo
from mdsgene.pdf_uri_utils import resolve_pdf_uri

# Get a logger for this module
configure_logging()
logger = logging.getLogger(__name__)


# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    mapping_items: list[MappingItem]
    patient_identifiers: list[dict[str, str | None]]
    patient_questions: list[list[QuestionInfo]]
    patient_answers: list[dict[str, str]]
    messages: Annotated[list, add_messages]


class QuestionsProcessingAgent(BaseAgent[State]):
    """Agent for processing questions mapping data from PDFs."""

    def __init__(self, pmid: str = None):
        """
        Initialize the question processing agent.

        Args:
            pmid: PMID of the document (optional)
        """
        super().__init__("questions_processing", "patient_cache.json", pmid)
        self.patient_cache_key = "__patient_identifiers_list_v1__"
        self.questions_dir = Path(".questions")
        self.mapping_data_path = self.questions_dir / "mapping_data.json"
        self.vector_store_dir = Path("vector_store/faiss_index")

        # Create vector store directory if it doesn't exist
        if not self.vector_store_dir.exists():
            try:
                self.vector_store_dir.mkdir(parents=True, exist_ok=True)
                logging.info(f"Vector store directory created at: {self.vector_store_dir}")
            except Exception as e:
                logging.error(f"Error creating vector store directory: {e}")

    def generate_cache_identifier(self, query_text: str) -> str:
        """
        Generate a cache identifier based on the query text and PMID.

        Args:
            query_text: The query text to generate a cache identifier for

        Returns:
            A cache identifier string
        """
        # Include PMID in the cache identifier if available
        if self.pmid:
            # Create a hash of the PMID + query text to use as the cache identifier
            return hashlib.md5(f"{self.pmid}_{query_text}".encode('utf-8')).hexdigest()
        else:
            # Create a hash of just the query text if PMID is not available
            return hashlib.md5(query_text.encode('utf-8')).hexdigest()

    def load_from_cache(self, cache_identifier: str) -> dict[str, str] | None:
        """
        Load a cached answer from the cache file.

        Args:
            cache_identifier: The cache identifier to load

        Returns:
            A dictionary containing the cached answer and context if found, None otherwise
        """
        # Get the PMID-specific cache directory
        pmid_dir = CACHE_DIR / self.pmid if self.pmid else CACHE_DIR

        # Create the directory if it doesn't exist
        pmid_dir.mkdir(parents=True, exist_ok=True)

        # Construct the cache file path using the PMID directory
        raw_answer_cache_filepath = pmid_dir / f"{cache_identifier}_raw.json"

        # Check if the cache file exists
        if raw_answer_cache_filepath.exists():
            try:
                with open(raw_answer_cache_filepath, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    logging.info(f"Cache HIT for query (id: {cache_identifier[:8]}...)")
                    return cached_data
            except Exception as e:
                logging.error(f"ERROR loading from cache: {e}")

        return None

    def get_cached_questions(self, all_pmids: bool = False) -> list[dict[str, str]]:
        """
        Get cached questions for the current PMID or all PMIDs.

        Args:
            all_pmids: If True, retrieve questions from all PMID directories

        Returns:
            A list of dictionaries containing pmid, cache_id and question
        """
        questions = []

        if all_pmids:
            # Scan all PMID directories in the cache
            for pmid_dir in CACHE_DIR.glob("*"):
                if pmid_dir.is_dir():
                    pmid = pmid_dir.name
                    # Get questions for this PMID
                    pmid_questions = self._get_questions_for_pmid(pmid_dir, pmid)
                    questions.extend(pmid_questions)
        elif self.pmid:
            # Get the PMID-specific cache directory
            pmid_dir = CACHE_DIR / self.pmid

            # If the directory exists, get questions for this PMID
            if pmid_dir.exists():
                questions = self._get_questions_for_pmid(pmid_dir, self.pmid)

        return questions

    def _get_questions_for_pmid(self, pmid_dir: Path, pmid: str) -> list[dict[str, str]]:
        """
        Helper method to get questions for a specific PMID directory.

        Args:
            pmid_dir: Path to the PMID directory
            pmid: PMID string

        Returns:
            List of questions for this PMID
        """
        pmid_questions = []

        # Scan the directory for *_raw.json files
        for cache_file in pmid_dir.glob("*_raw.json"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)

                    # Only include files that have a question field
                    if "question" in cached_data:
                        cache_id = cache_file.stem.replace("_raw", "")
                        pmid_questions.append({
                            "pmid": pmid,
                            "cache_id": cache_id,
                            "question": cached_data["question"],
                            "raw_answer": cached_data.get("raw_answer", "")
                        })
            except Exception as e:
                logging.error(f"Error reading cache file {cache_file}: {e}")
                continue

        return pmid_questions

    def save_to_cache(
        self,
        cache_identifier: str,
        raw_answer: str,
        question: str = None,
        context: str = None
    ):
        """
        Save an answer to the cache file.

        Args:
            cache_identifier: The cache identifier to save
            raw_answer: The raw answer to save
            question: The original question (optional)
            context: The context used to generate the answer (optional)
        """
        # Get the PMID-specific cache directory
        pmid_dir = CACHE_DIR / self.pmid if self.pmid else CACHE_DIR

        # Create the directory if it doesn't exist
        pmid_dir.mkdir(parents=True, exist_ok=True)

        # Construct the cache file path using the PMID directory
        raw_answer_cache_filepath = pmid_dir / f"{cache_identifier}_raw.json"

        # Save the raw answer to the cache file
        try:
            cache_data = {"raw_answer": raw_answer}
            if question:
                cache_data["question"] = question
            if context:
                cache_data["context"] = context

            with open(raw_answer_cache_filepath, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            logging.info(f"Saved answer and context to cache (id: {cache_identifier[:8]}...)")
        except Exception as e:
            logging.error(f"ERROR saving to cache: {e}")

    def load_mapping_data(self, state: State) -> State:
        """Load mapping data from JSON file."""
        if not self.mapping_data_path.exists():
            error_msg = f"Mapping data file not found at {self.mapping_data_path}"
            logger.error(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        try:
            with open(self.mapping_data_path, "r", encoding="utf-8") as f:
                items = json.load(f)

            mapping_data = []
            active_items = 0
            skipped_items = 0

            for entry in items:
                # Only add items that are marked as active or don't have an active field
                if entry.get("active", False):  # Default to False if field not present
                    mapping_data.append(
                        MappingItem(
                            field=entry["field"],
                            question=entry["question"],
                            mapped_excel_column=entry["mapped_excel_column"],
                            response_convertion_strategy=entry["response_convertion_strategy"],
                            custom_processor=None,  # We don't have processors in this simplified version
                            query_processor=entry.get("query_processor", "gemini"),  # "gemini" as default value
                            active=True  # Explicitly set active to True since we're only including active items
                        )
                    )
                    active_items += 1
                else:
                    skipped_items += 1

            logger.info(
                f"Loaded {active_items} active mapping items from JSON (skipped {skipped_items} inactive items)."
            )

            return {
                **state,
                "mapping_items": mapping_data,
                "messages": state["messages"] + [
                    {
                        "role": "assistant",
                        "content": f"Loaded {active_items} active mapping items from JSON "
                                   f"(skipped {skipped_items} inactive items)."
                    }
                ]
            }
        except Exception as e:
            error_msg = f"Failed to load mapping data: {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

    def get_patient_identifiers(self, state: State) -> State:
        """Get patient identifiers from cache or throw an exception."""
        cache = self.load_cache()
        loaded_from_cache = False

        logger.info("\nGetting patient identifiers (checking cache)...")
        cached_data = cache.get(self.patient_cache_key)

        if cached_data is not None:
            logger.debug("Cache HIT for patient identifiers.")
            try:
                if isinstance(cached_data, list):
                    patient_identifiers = cached_data
                    logger.info(f"Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                    loaded_from_cache = True
                else:
                    logging.warning("WARNING: Cached data for patient identifiers is not a list.")
            except Exception:
                print("ERROR: Failed to parse cached patient identifiers.")

        if not loaded_from_cache:
            error_msg = ("Patient identifiers not found in cache. "
                         "This agent requires pre-cached patient identifiers.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Proceeding with {len(patient_identifiers)} patient identifiers.")

        return {
            **state,
            "patient_identifiers": patient_identifiers,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Found {len(patient_identifiers)} patient identifiers in cache."}
            ]
        }

    def generate_patient_questions(self, state: State) -> State:
        """Generate questions for each patient using mapping items."""
        mapping_items = state["mapping_items"]
        patient_identifiers = state["patient_identifiers"]

        if not mapping_items:
            error_msg = "No mapping items loaded. Cannot generate questions."
            logger.error(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        if not patient_identifiers:
            error_msg = "No patient identifiers found. Cannot generate questions."
            logger.error(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        list_of_patient_question_sets = []
        logger.info(f"\nGenerating questions for {len(patient_identifiers)} unique patients...")

        for entry in patient_identifiers:
            patient_id = entry.get("patient")
            family_id = entry.get("family")
            one_patient_questions = []

            for item in mapping_items:
                context_prefix = f"Regarding patient '{patient_id}'"
                if family_id:
                    context_prefix += f" from family '{family_id}'"
                specific_query = f"{context_prefix}: {item.question}"

                try:
                    q_info = QuestionInfo(
                        field=item.field,
                        query=specific_query,
                        response_convertion_strategy=item.response_convertion_strategy,
                        # Use "gemini" as a fallback if query_processor is None
                        query_processor=item.query_processor or "gemini",
                        family_id=family_id,
                        patient_id=patient_id
                    )
                    one_patient_questions.append(q_info)
                except Exception as e:
                    logger.error(f"Failed to create QuestionInfo: {e}")  # Error should be more informative now
                    continue

            list_of_patient_question_sets.append(one_patient_questions)

        logger.info(f"Generated question sets for {len(list_of_patient_question_sets)} patients.")

        return {
            **state,
            "patient_questions": list_of_patient_question_sets,
            "messages": state["messages"] + [
                {
                    "role": "assistant",
                    "content": f"Generated question sets for {len(list_of_patient_question_sets)} patients."
                }
            ]
        }

    def process_patient_questions(self, state: State) -> State:
        """Process questions for all patients using a single Gemini request."""
        pdf_filepath = state["pdf_filepath"]
        patient_questions = state["patient_questions"]
        pdf_uri = resolve_pdf_uri(Path(pdf_filepath))

        if not patient_questions:
            error_msg = "No patient question sets generated. Cannot process patients."
            logger.error(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        logger.info(f"\nProcessing {len(patient_questions)} identified patient sets...")

        batch_questions: list[tuple[str, str, str]] = []
        patient_family_map: dict[str, str | None] = {}

        for patient_question_set in patient_questions:
            if not patient_question_set:
                continue
            patient_id = patient_question_set[0].patient_id or "UnknownPatient"
            patient_family_map[patient_id] = patient_question_set[0].family_id
            for q_obj in patient_question_set:
                batch_questions.append((patient_id, q_obj.field, q_obj.query))

        if not batch_questions:
            error_msg = "No questions generated for patients."
            logger.error(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        prompt_lines = [
            "Based on the PDF, answer all questions grouped by patient and field in JSON format like:",
            "{",
            '  "<patient_id>": {',
            '    "<field>": "<answer>",',
            "    ...",
            "  },",
            "  ...",
            "}",
            "",
            "Questions:",
        ]
        for patient_id, field, question_text in batch_questions:
            prompt_lines.append(f"- {patient_id}::{field}::{question_text}")
        prompt = "\n".join(prompt_lines)

        cache_identifier = self.generate_cache_identifier(prompt)
        cached_data = self.load_from_cache(cache_identifier)
        if cached_data is not None:
            raw_answer = cached_data.get("raw_answer")
        else:
            logger.info("Calling Gemini with batch question prompt...")
            raw_answer = None
            try:
                if pdf_uri:
                    result = self.ai_processor_client.answer_question(
                        question=prompt,
                        processor_name="gemini",
                        pdf_uri=pdf_uri,
                    )
                else:
                    result = self.ai_processor_client.answer_question(
                        pdf_filepath,
                        prompt,
                        "gemini",
                    )
                if result:
                    raw_answer = result[0]
                    self.save_to_cache(cache_identifier, raw_answer, prompt, result[1])
            except Exception as err:
                logger.error(f"Error using AIProcessorClient: {err}")

        patient_results: list[dict[str, Any]] = []
        if raw_answer:
            try:
                text = raw_answer.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text).strip()
                else:
                    match = re.search(r"\{.*\}", text, re.DOTALL)
                    if match:
                        text = match.group(0)
                    else:
                        logger.error("JSON content not found in Gemini response")
                        text = ""

                logger.debug("Sanitized raw response before json.loads:")
                logger.debug(text)

                if not text or not text.strip():
                    logger.error("Formatted response is empty before json.loads")
                    parsed = None
                else:
                    parsed = json.loads(text)
                if isinstance(parsed, dict):
                    for pid, answers in parsed.items():
                        row: dict[str, Any] = {
                            "family_id": patient_family_map.get(pid) or "-99",
                            "individual_id": pid,
                        }
                        if isinstance(answers, dict):
                            for field, answer in answers.items():
                                row[field] = answer
                        patient_results.append(row)
                else:
                    logger.error("Parsed batch JSON is not a dictionary.")
            except Exception as parse_err:
                logger.error(f"ERROR parsing batch JSON: {parse_err}")
        else:
            logger.error("No raw answer returned from Gemini.")

        if patient_results:
            patient_results = self.batch_format_patient_results(
                patient_results, state["mapping_items"]
            )

        logger.info(f"Processed {len(patient_results)} patient data rows.")

        return {
            **state,
            "patient_answers": patient_results,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Processed {len(patient_results)} patient data rows."}
            ]
        }

    def batch_format_patient_results(
        self,
        patient_results: list[dict[str, Any]],
        mapping_items: list[MappingItem],
    ) -> list[dict[str, Any]]:
        """Format all patient answers in one Gemini request."""
        raw_values: dict[str, dict[str, Any]] = {}
        for row in patient_results:
            patient_id = row.get("individual_id")
            if not patient_id:
                continue
            for field, value in row.items():
                if field in ("individual_id", "family_id"):
                    continue
                raw_values.setdefault(patient_id, {})[field] = value

        if not raw_values:
            return patient_results

        field_rules: dict[str, str] = {}
        for item in mapping_items:
            field_rules[item.field] = item.response_convertion_strategy

        patient_ids = list(raw_values.keys())
        fields_in_results = {fld for patient in raw_values.values() for fld in patient.keys()}
        rules_lines = [
            f"Field: {fld}\nRule: {field_rules[fld]}" for fld in fields_in_results if fld in field_rules
        ]
        rules_block = "\n".join(rules_lines)

        formatted_data: dict[str, Any] = {}
        missing_ids = patient_ids
        if self.pmid:
            cached = load_formatted_result(self.pmid, patient_ids)
            if cached:
                formatted_data.update(cached)
                missing_ids = [pid for pid in patient_ids if pid not in cached]

        if missing_ids:
            strategy = (
                "Below are the raw patient answers and fields in JSON format. "
                "Format the values according to the following rules:\n"
                f"{rules_block}\n\n"
                "Return answer strictly in format:\n{\n  \"<patient_id>\": {\n    \"<field>\": \"<formatted>\"\n  }\n}"
            )

            batches = [missing_ids[i : i + 3] for i in range(0, len(missing_ids), 3)]
            for batch in batches:
                try:
                    subset_json = json.dumps({pid: raw_values[pid] for pid in batch}, ensure_ascii=False)
                    result = self.ai_processor_client.format_answer(
                        subset_json,
                        strategy,
                        "gemini",
                        pmid=self.pmid,
                    )
                    if not result:
                        logger.error("AIProcessorClient returned no result for batch formatting.")
                        continue

                    formatted_json_text = result[0]
                    if not formatted_json_text or not formatted_json_text.strip().startswith("{"):
                        logger.error("ERROR: Gemini formatting response is empty or not JSON:\n---")
                        logger.error(formatted_json_text)
                        continue

                    text = formatted_json_text.strip()
                    if text.startswith("```"):
                        text = re.sub(r"^```(?:json)?\s*", "", text)
                        text = re.sub(r"\s*```$", "", text).strip()
                    else:
                        match = re.search(r"\{.*\}", text, re.DOTALL)
                        if match:
                            text = match.group(0)

                    formatted_json_text = text
                    if not formatted_json_text or not formatted_json_text.strip():
                        logging.error("ERROR: Formatted response is empty before json.loads")
                        continue
                    try:
                        new_data = json.loads(formatted_json_text)
                        if isinstance(new_data, dict):
                            missing_from_batch = [pid for pid in batch if pid not in new_data]
                            if missing_from_batch:
                                logging.warning(
                                    f"WARNING: Server response missing {len(missing_from_batch)} "
                                    f"patients: {missing_from_batch}"
                                )
                            formatted_data.update(new_data)
                            if self.pmid:
                                save_formatted_result(
                                    self.pmid,
                                    subset_json,
                                    formatted_json_text,
                                    strategy,
                                    new_data,
                                )
                        else:
                            logger.error("Parsed batch JSON is not a dictionary.")
                    except Exception as parse_err:
                        print(f"ERROR parsing batch JSON: {parse_err}")
                except Exception as err:
                    logger.error(f"ERROR using AIProcessorClient for batch formatting: {err}")

        if formatted_data:
            for row in patient_results:
                pid = row.get("individual_id")
                patient_fields = formatted_data.get(pid, {}) if isinstance(formatted_data, dict) else {}
                for field in list(row.keys()):
                    if field in ("individual_id", "family_id"):
                        continue
                    formatted_val = patient_fields.get(field)
                    if formatted_val is None:
                        row[field] = "-99_FORMAT_ERROR"
                    else:
                        row[field] = formatted_val
        else:
            logger.error("Failed to format batch answers or parse result.")

        return patient_results

    def setup(self):
        """Set up the agent by building the graph."""
        nodes = {
            "load_mapping_data": self.load_mapping_data,
            "get_patient_identifiers": self.get_patient_identifiers,
            "generate_patient_questions": self.generate_patient_questions,
            "process_patient_questions": self.process_patient_questions
        }
        return self.build_graph(State, nodes)

    def print_results(self, final_state: State):
        """Print the results of running the agent."""
        logger.info(
            f"\n=== Results ===\n"
            f"PDF: {Path(final_state['pdf_filepath']).name}\n"
            f"Mapping Items: {len(final_state.get('mapping_items', []))} items\n"
            f"Patient Identifiers: {len(final_state.get('patient_identifiers', []))} patients\n"
            f"Patient Answers: {len(final_state.get('patient_answers', []))} rows"
        )

        # Call the base class method to print the conversation
        super().print_results(final_state)


def main():
    """Run the agent if this file is executed directly."""
    # Get PDF filepath from command line argument or use a default
    if len(sys.argv) > 1:
        pdf_filepath = sys.argv[1]
    else:
        # Prompt user for PDF filepath
        pdf_filepath = input("Enter the path to a PDF file: ")

    # Validate the PDF filepath
    if not Path(pdf_filepath).exists():
        logger.error(f"PDF file not found at {pdf_filepath}")
        sys.exit(1)

    logger.info(f"Processing PDF: {pdf_filepath}")

    # Initialize the agent
    agent = QuestionsProcessingAgent()
    agent.setup()

    # Initialize the state
    initial_state = {
        "pdf_filepath": pdf_filepath,
        "mapping_items": [],
        "patient_identifiers": [],
        "patient_questions": [],
        "patient_answers": [],
        "messages": [
            {"role": "user", "content": f"Process questions for patients in {Path(pdf_filepath).name}"}
        ]
    }

    try:
        # Run the agent
        final_state = agent.run(initial_state)

        # Display the results
        agent.print_results(final_state)

    except ValueError as e:
        # This will catch the exception thrown when patient identifiers are not found in cache
        logger.error(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ERROR: An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
