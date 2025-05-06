import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from ai.gemini_processor import GeminiProcessor

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

# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    patient_identifiers: List[Dict[str, Optional[str]]]
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

# Define the agent nodes
def extract_patient_identifiers(state: State) -> State:
    """Extract patient identifiers from PDF using Gemini with caching."""
    pdf_filepath = state["pdf_filepath"]
    pdf_name = Path(pdf_filepath).name

    patient_identifiers = []
    cache = load_cache()
    loaded_from_cache = False

    print("\nGetting patient identifiers (checking cache)...")
    cached_data_str = cache.get(PATIENT_CACHE_KEY)

    if cached_data_str is not None:
        print("  Cache HIT for patient identifiers.")
        try:
            patient_identifiers_parsed = cached_data_str
            if isinstance(patient_identifiers_parsed, list):
                patient_identifiers = patient_identifiers_parsed
                print(f"  Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                loaded_from_cache = True
            else:
                print("  WARNING: Cached data for patient identifiers is not a list. Re-fetching.")
        except Exception:
            print("  ERROR: Failed to parse cached patient identifiers. Re-fetching.")

    if not loaded_from_cache:
        print("  Cache MISS or invalid cache data. Querying Gemini API for patient identifiers...")
        try:
            gemini_processor = initialize_gemini(pdf_filepath)
            if not gemini_processor:
                return {
                    **state,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": "Failed to initialize Gemini Processor."}
                    ]
                }

            fetched_identifiers = gemini_processor.get_patient_identifiers()
            print(f"  Successfully fetched {len(fetched_identifiers)} identifiers via Gemini.")

            try:
                cache[PATIENT_CACHE_KEY] = fetched_identifiers
                save_cache(cache)
                print("  Stored fetched patient identifiers in cache.")
                patient_identifiers = fetched_identifiers
            except TypeError as json_err:
                print(f"  ERROR: Could not serialize fetched patient identifiers to JSON: {json_err}. Not caching.")
                patient_identifiers = fetched_identifiers
            except Exception as cache_err:
                print(f"  ERROR: Could not save patient identifiers to cache: {cache_err}")
                patient_identifiers = fetched_identifiers

        except Exception as e:
            error_msg = f"ERROR getting patient identifiers via Gemini: {e}"
            print(error_msg)
            traceback.print_exc()
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

    print(f"--> Proceeding with {len(patient_identifiers)} patient identifiers.")

    return {
        **state,
        "patient_identifiers": patient_identifiers,
        "messages": state["messages"] + [
            {"role": "assistant", "content": f"Found {len(patient_identifiers)} patient identifiers."}
        ]
    }

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("extract_patient_identifiers", extract_patient_identifiers)

# Define the edges
graph_builder.add_edge(START, "extract_patient_identifiers")
graph_builder.add_edge("extract_patient_identifiers", END)

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
        "patient_identifiers": [],
        "messages": [
            {"role": "user", "content": f"Extract patient identifiers from {Path(pdf_filepath).name}"}
        ]
    }

    # Run the graph
    final_state = graph.invoke(initial_state)

    # Display the results
    print("\n=== Results ===")
    print(f"PDF: {Path(pdf_filepath).name}")
    print(f"Patient Identifiers: {final_state.get('patient_identifiers')}")

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
