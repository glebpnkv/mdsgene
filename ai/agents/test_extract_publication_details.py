import os
import json
import traceback
from pathlib import Path
from typing import Dict, Optional, Any

from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from ai.gemini_processor import GeminiProcessor
from ai.pmid_extractor import PmidExtractor

load_dotenv()

# Define cache directory and file path
CACHE_DIR = Path("cache")
if not CACHE_DIR.exists():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created at: {CACHE_DIR}")
    except Exception as e:
        print(f"Error creating cache directory: {e}")

PMID_CACHE_PATH = CACHE_DIR / "pmid_cache.json"

def load_pmid_cache() -> Dict[str, dict]:
    """Load PMID cache from file."""
    if PMID_CACHE_PATH.exists():
        with open(PMID_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_pmid_cache(cache: Dict[str, dict]):
    """Save PMID cache to file."""
    with open(PMID_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    publication_details: Optional[Dict[str, str]]
    pmid: Optional[str]
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
def extract_publication_details(state: State) -> State:
    """Extract publication details from PDF using Gemini with caching."""
    pdf_filepath = state["pdf_filepath"]
    pdf_name = Path(pdf_filepath).name

    # Check cache first
    print(f"\nExtracting publication details for PDF: {pdf_name}")
    pmid_cache = load_pmid_cache()

    if pdf_name in pmid_cache:
        pub_details = pmid_cache[pdf_name]
        print(f"Loaded publication details from cache: {pub_details}")
        return {
            **state,
            "publication_details": pub_details,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Publication details loaded from cache: {pub_details}"}
            ]
        }

    # Not in cache, extract using Gemini
    try:
        gemini_processor = initialize_gemini(pdf_filepath)
        if not gemini_processor:
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": "Failed to initialize Gemini Processor."}
                ]
            }

        pub_details = gemini_processor.extract_publication_details()

        # Save to cache
        pmid_cache[pdf_name] = pub_details
        save_pmid_cache(pmid_cache)

        return {
            **state,
            "publication_details": pub_details,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Extracted publication details: {pub_details}"}
            ]
        }
    except Exception as e:
        error_msg = f"Error extracting publication details: {e}"
        print(error_msg)
        traceback.print_exc()
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": error_msg}
            ]
        }

def get_pmid(state: State) -> State:
    """Get PMID using PmidExtractor with the extracted publication details."""
    pub_details = state.get("publication_details", {})

    if not pub_details:
        return {
            **state,
            "messages": state["messages"] + [
                {"role": "assistant", "content": "No publication details available to extract PMID."}
            ]
        }

    pmid = PmidExtractor.get_pmid(
        title=pub_details.get("title"),
        author=pub_details.get("first_author_lastname", ""),
        year=pub_details.get("year", "")
    )

    if pmid:
        message = f"Successfully extracted PMID: {pmid}"
    else:
        message = "Could not extract PMID."

    print(message)

    return {
        **state,
        "pmid": pmid,
        "messages": state["messages"] + [
            {"role": "assistant", "content": message}
        ]
    }

# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("extract_publication_details", extract_publication_details)
graph_builder.add_node("get_pmid", get_pmid)

# Define the edges
graph_builder.add_edge(START, "extract_publication_details")
graph_builder.add_edge("extract_publication_details", "get_pmid")
graph_builder.add_edge("get_pmid", END)

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
        "publication_details": None,
        "pmid": None,
        "messages": [
            {"role": "user", "content": f"Extract publication details and PMID from {Path(pdf_filepath).name}"}
        ]
    }

    # Run the graph
    final_state = graph.invoke(initial_state)

    # Display the results
    print("\n=== Results ===")
    print(f"PDF: {Path(pdf_filepath).name}")
    print(f"Publication Details: {final_state.get('publication_details')}")
    print(f"PMID: {final_state.get('pmid')}")

    # Save PDF with PMID as filename if PMID was found
    pmid = final_state.get('pmid')
    if pmid:
        # Create pdf directory if it doesn't exist
        pdf_dir = Path("pdf")
        if not pdf_dir.exists():
            try:
                pdf_dir.mkdir(parents=True, exist_ok=True)
                print(f"PDF directory created at: {pdf_dir}")
            except Exception as e:
                print(f"Error creating PDF directory: {e}")

        # Copy the PDF file to the pdf directory with PMID as filename
        try:
            import shutil
            source_path = Path(pdf_filepath)
            target_path = pdf_dir / f"{pmid}.pdf"
            shutil.copy2(source_path, target_path)
            print(f"PDF saved as: {target_path}")
        except Exception as e:
            print(f"Error saving PDF with PMID filename: {e}")
            traceback.print_exc()

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
