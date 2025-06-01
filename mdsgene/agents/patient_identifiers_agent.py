import logging
import sys
import traceback
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from mdsgene.agents.base_agent import BaseAgent
from mdsgene.logging_config import configure_logging
from mdsgene.pdf_uri_utils import resolve_pdf_uri

# Get a logger for this module
configure_logging()
logger = logging.getLogger(__name__)


# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    patient_identifiers: List[Dict[str, Optional[str]]]
    messages: Annotated[list, add_messages]


class PatientIdentifiersAgent(BaseAgent[State]):
    """Agent for extracting patient identifiers from PDFs."""

    def __init__(self, pmid: str = None):
        """
        Initialize the patient identifier agent.

        Args:
            pmid: PMID of the document (optional)
        """
        super().__init__("patient_identifiers", "patient_cache.json", pmid)
        self.patient_cache_key = "__patient_identifiers_list_v1__"

    def extract_patient_identifiers(self, state: State) -> State:
        """Extract patient identifiers from PDF using Gemini with caching."""
        pdf_filepath = state["pdf_filepath"]

        patient_identifiers = []
        cache = self.load_cache()
        loaded_from_cache = False

        logger.info("Getting patient identifiers (checking cache)...")
        cached_data_str = cache.get(self.patient_cache_key)

        if cached_data_str is not None:
            logger.debug("Cache HIT for patient identifiers.")
            try:
                patient_identifiers_parsed = cached_data_str
                if isinstance(patient_identifiers_parsed, list):
                    patient_identifiers = patient_identifiers_parsed
                    logger.info(f"Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                    loaded_from_cache = True
                else:
                    logger.warning("Cached data for patient identifiers is not a list. Re-fetching.")
            except Exception:
                logger.error("Failed to parse cached patient identifiers. Re-fetching.")

        if not loaded_from_cache:
            logger.info("Cache MISS or invalid cache data. Querying AI Processor Service for patient identifiers...")
            try:
                pdf_uri = resolve_pdf_uri(Path(pdf_filepath))
                if pdf_uri:
                    fetched_identifiers = self.ai_processor_client.get_patient_identifiers(pdf_uri=pdf_uri)
                else:
                    fetched_identifiers = self.ai_processor_client.get_patient_identifiers(pdf_filepath)
                logger.info(f"Successfully fetched {len(fetched_identifiers)} identifiers via Gemini.")

                try:
                    cache[self.patient_cache_key] = fetched_identifiers
                    self.save_cache(cache)
                    logger.debug("Stored fetched patient identifiers in cache.")
                    patient_identifiers = fetched_identifiers
                except TypeError as json_err:
                    logger.error(f"Could not serialize fetched patient identifiers to JSON: {json_err}. Not caching.")
                    patient_identifiers = fetched_identifiers
                except Exception as cache_err:
                    logger.error(f"Could not save patient identifiers to cache: {cache_err}")
                    patient_identifiers = fetched_identifiers

            except Exception as e:
                error_msg = f"ERROR getting patient identifiers via Gemini: {e}"
                logger.error(error_msg)
                import traceback
                traceback.print_exc()
                return {
                    **state,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": error_msg}
                    ]
                }

        logger.info(f"Proceeding with {len(patient_identifiers)} patient identifiers.")

        return {
            **state,
            "patient_identifiers": patient_identifiers,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Found {len(patient_identifiers)} patient identifiers."}
            ]
        }

    def setup(self):
        """Set up the agent by building the graph."""
        nodes = {
            "extract_patient_identifiers": self.extract_patient_identifiers
        }
        return self.build_graph(State, nodes)

    def print_results(self, final_state: State):
        """Print the results of running the agent."""
        logger.info(
            f"=== Results ===\n"
            f"PDF: {Path(final_state['pdf_filepath']).name}\n"
            f"Patient Identifiers: {final_state.get('patient_identifiers')}"
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
    agent = PatientIdentifiersAgent()
    agent.setup()

    # Initialize the state
    initial_state = {
        "pdf_filepath": pdf_filepath,
        "patient_identifiers": [],
        "messages": [
            {"role": "user", "content": f"Extract patient identifiers from {Path(pdf_filepath).name}"}
        ]
    }

    try:
        # Run the agent
        final_state = agent.run(initial_state)

        # Display the results
        agent.print_results(final_state)

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
