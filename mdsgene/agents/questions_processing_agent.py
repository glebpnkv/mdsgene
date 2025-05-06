import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from ai.agents.base_agent import BaseAgent, CACHE_DIR
from ai.mapping_item import MappingItem, QuestionInfo
from ai.document_processor import DocumentProcessor

# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    mapping_items: List[MappingItem]
    patient_identifiers: List[Dict[str, Optional[str]]]
    patient_questions: List[List[QuestionInfo]]
    patient_answers: List[Dict[str, str]]
    messages: Annotated[list, add_messages]

class QuestionsProcessingAgent(BaseAgent[State]):
    """Agent for processing questions mapping data from PDFs."""

    def __init__(self, pmid: str = None):
        """
        Initialize the questions processing agent.

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
                print(f"Vector store directory created at: {self.vector_store_dir}")
            except Exception as e:
                print(f"Error creating vector store directory: {e}")

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

    def load_from_cache(self, cache_identifier: str) -> Optional[Dict[str, str]]:
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
                    print(f"  Cache HIT for query (id: {cache_identifier[:8]}...)")
                    return cached_data
            except Exception as e:
                print(f"  ERROR loading from cache: {e}")

        return None

    def get_cached_questions(self, all_pmids: bool = False) -> List[Dict[str, str]]:
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

    def _get_questions_for_pmid(self, pmid_dir: Path, pmid: str) -> List[Dict[str, str]]:
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
                print(f"Error reading cache file {cache_file}: {e}")
                continue

        return pmid_questions

    def save_to_cache(self, cache_identifier: str, raw_answer: str, question: str = None, context: str = None):
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
            print(f"  Saved answer and context to cache (id: {cache_identifier[:8]}...)")
        except Exception as e:
            print(f"  ERROR saving to cache: {e}")


    def load_mapping_data(self, state: State) -> State:
        """Load mapping data from JSON file."""
        if not self.mapping_data_path.exists():
            error_msg = f"ERROR: Mapping data file not found at {self.mapping_data_path}"
            print(error_msg)
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

            print(f"Loaded {active_items} active mapping items from JSON (skipped {skipped_items} inactive items).")

            return {
                **state,
                "mapping_items": mapping_data,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Loaded {active_items} active mapping items from JSON (skipped {skipped_items} inactive items)."}
                ]
            }
        except Exception as e:
            error_msg = f"ERROR: Failed to load mapping data: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
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

        print("\nGetting patient identifiers (checking cache)...")
        cached_data = cache.get(self.patient_cache_key)

        if cached_data is not None:
            print("  Cache HIT for patient identifiers.")
            try:
                if isinstance(cached_data, list):
                    patient_identifiers = cached_data
                    print(f"  Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                    loaded_from_cache = True
                else:
                    print("  WARNING: Cached data for patient identifiers is not a list.")
            except Exception:
                print("  ERROR: Failed to parse cached patient identifiers.")

        if not loaded_from_cache:
            error_msg = "ERROR: Patient identifiers not found in cache. This agent requires pre-cached patient identifiers."
            print(error_msg)
            raise ValueError(error_msg)

        print(f"--> Proceeding with {len(patient_identifiers)} patient identifiers.")

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
            error_msg = "ERROR: No mapping items loaded. Cannot generate questions."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        if not patient_identifiers:
            error_msg = "ERROR: No patient identifiers found. Cannot generate questions."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        list_of_patient_question_sets = []
        print(f"\nGenerating questions for {len(patient_identifiers)} unique patients...")

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
                        query_processor=item.query_processor or "gemini", # Use "gemini" as fallback if query_processor is None
                        family_id=family_id,
                        patient_id=patient_id
                    )
                    one_patient_questions.append(q_info)
                except Exception as e:
                    print(f"ERROR: Failed to create QuestionInfo: {e}") # Error should be more informative now
                    continue

            list_of_patient_question_sets.append(one_patient_questions)

        print(f"Generated question sets for {len(list_of_patient_question_sets)} patients.")

        return {
            **state,
            "patient_questions": list_of_patient_question_sets,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Generated question sets for {len(list_of_patient_question_sets)} patients."}
            ]
        }

    def process_patient_questions(self, state: State) -> State:
        """Process questions for each patient and get answers."""
        pdf_filepath = state["pdf_filepath"]
        patient_questions = state["patient_questions"]

        if not patient_questions:
            error_msg = "ERROR: No patient question sets generated. Cannot process patients."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        # Process each patient's questions
        all_patient_data_rows = []
        print(f"\nProcessing {len(patient_questions)} identified patient sets...")

        patient_num = 0
        for patient_question_set in patient_questions:
            patient_num += 1
            if not patient_question_set:
                print(f"\n=== Skipping Patient Set {patient_num} (Empty Question Set) ===")
                continue

            current_patient_id = patient_question_set[0].patient_id or "UnknownPatient"
            current_family_id = patient_question_set[0].family_id
            print(f"\n=== Processing Patient Set {patient_num} (Patient: '{current_patient_id}', Family: '{current_family_id or 'N/A'}') ===")

            patient_results = {}
            patient_results["family_id"] = current_family_id or "-99"
            patient_results["individual_id"] = current_patient_id

            for q_obj in patient_question_set:
                print(f"--- Querying for field: {q_obj.field} (Patient: {current_patient_id}) ---")
                query_text = q_obj.query

                # Generate a cache identifier for this query
                cache_identifier = self.generate_cache_identifier(query_text)

                # Try to load from cache first
                cached_data = self.load_from_cache(cache_identifier)
                if cached_data is not None:
                    raw_answer = cached_data.get("raw_answer")
                    context = cached_data.get("context")
                    print(f"  Using cached answer: {raw_answer[:50]}...")
                    print(f"  Using cached context: {context[:50] if context else 'None'}...")
                else:
                    # Use AIProcessorClient to get answer
                    raw_answer = None
                    context = None
                    try:
                        print(f"  Using AIProcessorClient for query: {query_text[:50]}...")
                        processor_name = getattr(q_obj, "query_processor", "gemini")  # Use query_processor if available, default to "gemini"
                        result = self.ai_processor_client.answer_question(pdf_filepath, query_text, processor_name)
                        if result:
                            raw_answer, context = result
                            print(f"  AIProcessorClient found answer: {raw_answer[:50]}...")
                            print(f"  AIProcessorClient context: {context[:50] if context else 'None'}...")
                            # Save to cache if we got an answer
                            self.save_to_cache(cache_identifier, raw_answer, query_text, context)
                        else:
                            print(f"  AIProcessorClient returned no answer.")
                    except Exception as processor_err:
                        print(f"  ERROR using AIProcessorClient: {processor_err}")
                        raw_answer = None
                        context = None

                # Format the answer if we got one
                if raw_answer:
                    try:
                        # Generate a cache identifier for the formatting request
                        format_cache_identifier = self.generate_cache_identifier(f"format_{raw_answer}_{q_obj.response_convertion_strategy}")

                        # Try to load formatted answer from cache first
                        cached_format_data = self.load_from_cache(format_cache_identifier)

                        if cached_format_data is not None:
                            formatted_answer = cached_format_data.get("raw_answer")
                            format_context = cached_format_data.get("context")
                            print(f"  Using cached formatted answer: {formatted_answer}")
                            print(f"  Using cached format context: {format_context[:50] if format_context else 'None'}...")
                        else:
                            # Use AIProcessorClient for formatting
                            formatted_answer = None
                            format_context = None
                            try:
                                print(f"  Using AIProcessorClient for formatting...")
                                processor_name = getattr(q_obj, "query_processor", "gemini")  # Use query_processor if available, default to "gemini"
                                format_result = self.ai_processor_client.format_answer(raw_answer, q_obj.response_convertion_strategy, processor_name)
                                if format_result:
                                    formatted_answer, format_context = format_result
                                    print(f"  AIProcessorClient formatted answer: {formatted_answer}")
                                    print(f"  AIProcessorClient format context: {format_context[:50] if format_context else 'None'}...")
                                    # Save formatted answer to cache
                                    format_question = f"Format answer for: {query_text}"
                                    self.save_to_cache(format_cache_identifier, formatted_answer, format_question, format_context)
                                else:
                                    print(f"  AIProcessorClient returned no formatted answer.")
                            except Exception as format_err:
                                print(f"  ERROR using AIProcessorClient for formatting: {format_err}")
                                formatted_answer = None
                                format_context = None

                            # If no processor could format the answer, use a default
                            if formatted_answer is None:
                                formatted_answer = "-99_FORMAT_ERROR"
                                print(f"  No processor could format the answer. Using default: {formatted_answer}")

                        patient_results[q_obj.field] = formatted_answer
                        print(f"  Final formatted answer: {formatted_answer}")
                    except Exception as format_err:
                        print(f"  ERROR during formatting: {format_err}")
                        patient_results[q_obj.field] = "-99_FORMAT_ERROR"
                else:
                    patient_results[q_obj.field] = "-99_NO_ANSWER"
                    print(f"  No answer found. Using default: -99_NO_ANSWER")

            all_patient_data_rows.append(patient_results)

        print(f"Processed {len(all_patient_data_rows)} patient data rows.")

        return {
            **state,
            "patient_answers": all_patient_data_rows,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Processed {len(all_patient_data_rows)} patient data rows."}
            ]
        }

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
        print("\n=== Results ===")
        print(f"PDF: {Path(final_state['pdf_filepath']).name}")
        print(f"Mapping Items: {len(final_state.get('mapping_items', []))} items")
        print(f"Patient Identifiers: {len(final_state.get('patient_identifiers', []))} patients")
        print(f"Patient Answers: {len(final_state.get('patient_answers', []))} rows")

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
        print(f"Error: PDF file not found at {pdf_filepath}")
        sys.exit(1)

    print(f"Processing PDF: {pdf_filepath}")

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
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
