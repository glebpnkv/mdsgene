import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable

from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from ai.gemini_processor import GeminiProcessor
from ai.mapping_item import MappingItem, QuestionInfo
from ai.pdf_text_extractor import PdfTextExtractor
from ai.document_processor import DocumentProcessor
from langchain_community.vectorstores import FAISS

load_dotenv()

# Define cache directory and file path
CACHE_DIR = Path("cache")
if not CACHE_DIR.exists():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created at: {CACHE_DIR}")
    except Exception as e:
        print(f"Error creating cache directory: {e}")

PATIENT_CACHE_KEY = "__patient_identifiers_list_v1__"
CACHE_FILE_PATH = CACHE_DIR / "patient_cache.json"
QUESTIONS_DIR = Path(".questions")
MAPPING_DATA_PATH = QUESTIONS_DIR / "mapping_data.json"

def load_cache() -> Dict[str, Any]:
    """Load cache from file."""
    if CACHE_FILE_PATH.exists():
        with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, Any]):
    """Save cache to file."""
    with open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# Define vector store directory
VECTOR_STORE_DIR = Path("vector_store/faiss_index")
if not VECTOR_STORE_DIR.exists():
    try:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Vector store directory created at: {VECTOR_STORE_DIR}")
    except Exception as e:
        print(f"Error creating vector store directory: {e}")

# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    mapping_items: List[MappingItem]
    patient_identifiers: List[Dict[str, Optional[str]]]
    patient_questions: List[List[QuestionInfo]]
    patient_answers: List[Dict[str, str]]
    vector_store: Optional[FAISS]
    messages: Annotated[list, add_messages]

# Initialize GeminiProcessor
def initialize_gemini(pdf_filepath: str) -> GeminiProcessor:
    """Initialize GeminiProcessor with the given PDF filepath."""
    try:
        gemini_processor = GeminiProcessor(
            pdf_filepath=Path(pdf_filepath),
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        if gemini_processor.pdf_parts is None:
            raise ValueError("Failed to load PDF into Gemini parts.")
        print("Gemini Processor initialized successfully.")
        return gemini_processor
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize Gemini Processor: {e}")
        traceback.print_exc()
        return None

# Initialize Vector Store
def initialize_vector_store(state: State) -> State:
    """Initialize Vector Store with the given PDF filepath."""
    pdf_filepath = state["pdf_filepath"]

    try:
        # Extract text from PDF
        print("\nExtracting text from PDF...")
        extractor = PdfTextExtractor()
        pdf_text = extractor.extract_text(pdf_filepath)

        if pdf_text is None:
            error_msg = "ERROR: Could not extract text from PDF."
            print(error_msg)
            return {
                **state,
                "vector_store": None,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

        print(f"PDF Text extracted successfully ({len(pdf_text)} chars).")

        # Process and store the document
        print("\nProcessing document and creating vector store...")
        processor = DocumentProcessor(storage_path=str(VECTOR_STORE_DIR))
        processor.process_document(pdf_text, source_filename=Path(pdf_filepath).name)

        vector_store = processor.get_vector_store()
        if vector_store:
            # Optional: Check how many vectors are stored
            try:
                if hasattr(vector_store, 'index') and vector_store.index:
                    print(f"FAISS index contains {vector_store.index.ntotal} vectors.")
                else:
                    print("Could not determine the number of vectors in the store.")

                # Test the vector store with a simple query
                test_query = "What is this document about?"
                results = vector_store.similarity_search(test_query, k=1)
                print(f"\nTest search results for '{test_query}': {len(results)}")
                if results:
                    print(f"Top result snippet: {results[0].page_content[:100]}...")
            except Exception as vs_check_e:
                print(f"Could not check vector store details: {vs_check_e}")

            print("Vector store created successfully.")
            return {
                **state,
                "vector_store": vector_store,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": "Vector store created successfully."}
                ]
            }
        else:
            error_msg = "ERROR: Vector store was not created."
            print(error_msg)
            return {
                **state,
                "vector_store": None,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
    except Exception as e:
        error_msg = f"ERROR: Failed to initialize Vector Store: {e}"
        print(error_msg)
        traceback.print_exc()
        return {
            **state,
            "vector_store": None,
            "messages": state["messages"] + [
                {"role": "assistant", "content": error_msg}
            ]
        }

# Load mapping data
def load_mapping_data(state: State) -> State:
    """Load mapping data from JSON file."""
    if not MAPPING_DATA_PATH.exists():
        error_msg = f"ERROR: Mapping data file not found at {MAPPING_DATA_PATH}"
        print(error_msg)
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": error_msg}
            ]
        }

    try:
        with open(MAPPING_DATA_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)

        mapping_data = []
        for entry in items:
            mapping_data.append(
                MappingItem(
                    field=entry["field"],
                    question=entry["question"],
                    mapped_excel_column=entry["mapped_excel_column"],
                    response_convertion_strategy=entry["response_convertion_strategy"],
                    custom_processor=None  # We don't have processors in this simplified version
                )
            )

        print(f"Loaded {len(mapping_data)} mapping items from JSON.")

        return {
            **state,
            "mapping_items": mapping_data,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Loaded {len(mapping_data)} mapping items from JSON."}
            ]
        }
    except Exception as e:
        error_msg = f"ERROR: Failed to load mapping data: {e}"
        print(error_msg)
        traceback.print_exc()
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": error_msg}
            ]
        }

# Get patient identifiers
def get_patient_identifiers(state: State) -> State:
    """Get patient identifiers from cache or throw an exception."""
    cache = load_cache()
    loaded_from_cache = False

    print("\nGetting patient identifiers (checking cache)...")
    cached_data = cache.get(PATIENT_CACHE_KEY)

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

# Generate questions for patients
def generate_patient_questions(state: State) -> State:
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
                    family_id=family_id,
                    patient_id=patient_id
                )
                one_patient_questions.append(q_info)
            except Exception as e:
                print(f"ERROR: Failed to create QuestionInfo: {e}")
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

# Process patient questions
def process_patient_questions(state: State) -> State:
    """Process questions for each patient and get answers."""
    pdf_filepath = state["pdf_filepath"]
    patient_questions = state["patient_questions"]
    vector_store = state.get("vector_store")

    if not patient_questions:
        error_msg = "ERROR: No patient question sets generated. Cannot process patients."
        print(error_msg)
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": error_msg}
            ]
        }

    # Initialize Gemini processor
    gemini_processor = initialize_gemini(pdf_filepath)
    if not gemini_processor:
        error_msg = "ERROR: Failed to initialize Gemini Processor."
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

            # Try to use vector store first if available
            raw_answer = None
            if vector_store:
                try:
                    print(f"  Using vector store for query: {query_text[:50]}...")
                    results = vector_store.similarity_search(query_text, k=1)
                    if results:
                        raw_answer = results[0].page_content
                        print(f"  Vector store found answer: {raw_answer[:50]}...")
                except Exception as vs_err:
                    print(f"  ERROR using vector store: {vs_err}. Falling back to Gemini.")
                    raw_answer = None

            # Fall back to Gemini if vector store failed or is not available
            if raw_answer is None:
                try:
                    print(f"  Using Gemini for query: {query_text[:50]}...")
                    raw_answer = gemini_processor.answer_question(query_text)
                    if raw_answer:
                        print(f"  Gemini found answer: {raw_answer[:50]}...")
                    else:
                        print("  Gemini returned no answer.")
                except Exception as gemini_err:
                    print(f"  ERROR using Gemini: {gemini_err}")
                    raw_answer = None

            # Format the answer if we got one
            if raw_answer:
                try:
                    formatted_answer = gemini_processor.format_answer(raw_answer, q_obj.response_convertion_strategy)
                    patient_results[q_obj.field] = formatted_answer
                    print(f"  Formatted answer: {formatted_answer}")
                except Exception as format_err:
                    print(f"  ERROR formatting answer: {format_err}")
                    patient_results[q_obj.field] = "-99_FORMAT_ERROR"
            else:
                patient_results[q_obj.field] = "-99_NO_ANSWER"

        all_patient_data_rows.append(patient_results)

    print(f"Processed {len(all_patient_data_rows)} patient data rows.")

    return {
        **state,
        "patient_answers": all_patient_data_rows,
        "messages": state["messages"] + [
            {"role": "assistant", "content": f"Processed {len(all_patient_data_rows)} patient data rows."}
        ]
    }

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("load_mapping_data", load_mapping_data)
graph_builder.add_node("initialize_vector_store", initialize_vector_store)
graph_builder.add_node("get_patient_identifiers", get_patient_identifiers)
graph_builder.add_node("generate_patient_questions", generate_patient_questions)
graph_builder.add_node("process_patient_questions", process_patient_questions)

# Define the edges
graph_builder.add_edge(START, "load_mapping_data")
graph_builder.add_edge("load_mapping_data", "initialize_vector_store")
graph_builder.add_edge("initialize_vector_store", "get_patient_identifiers")
graph_builder.add_edge("get_patient_identifiers", "generate_patient_questions")
graph_builder.add_edge("generate_patient_questions", "process_patient_questions")
graph_builder.add_edge("process_patient_questions", END)

# Compile the graph
graph = graph_builder.compile()

# Run the agent if this file is executed directly
if __name__ == "__main__":
    import sys

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

    # Initialize the state
    initial_state = {
        "pdf_filepath": pdf_filepath,
        "mapping_items": [],
        "patient_identifiers": [],
        "patient_questions": [],
        "patient_answers": [],
        "vector_store": None,
        "messages": [
            {"role": "user", "content": f"Process questions for patients in {Path(pdf_filepath).name}"}
        ]
    }

    try:
        # Run the graph
        final_state = graph.invoke(initial_state)

        # Display the results
        print("\n=== Results ===")
        print(f"PDF: {Path(pdf_filepath).name}")
        print(f"Mapping Items: {len(final_state.get('mapping_items', []))} items")
        print(f"Patient Identifiers: {len(final_state.get('patient_identifiers', []))} patients")
        print(f"Patient Answers: {len(final_state.get('patient_answers', []))} rows")

        # Display vector store information
        vector_store = final_state.get('vector_store')
        if vector_store:
            try:
                if hasattr(vector_store, 'index') and vector_store.index:
                    print(f"Vector Store: Active with {vector_store.index.ntotal} vectors")
                else:
                    print("Vector Store: Active (vector count unknown)")
            except Exception:
                print("Vector Store: Active (could not determine vector count)")
        else:
            print("Vector Store: Not available")

        # Print the conversation
        print("\n=== Conversation ===")
        for message in final_state["messages"]:
            # Handle both dict-style messages and LangChain message objects
            if hasattr(message, "type") and hasattr(message, "content"):
                # This is a LangChain message object (like HumanMessage)
                role = message.type
                content = message.content
            elif isinstance(message, dict) and "role" in message and "content" in message:
                # This is a dictionary-style message
                role = message["role"]
                content = message["content"]
            else:
                # Unknown message format, print what we can
                print(f"Unknown message format: {message}")
                continue

            print(f"{role.upper()}: {content}")
    except ValueError as e:
        # This will catch the exception thrown when patient identifiers are not found in cache
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
