import sys
import uuid
from pathlib import Path
from typing import Dict, Optional, Any

from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from ai.pmid_extractor import PmidExtractor
from ai.agents.base_agent import BaseAgent

# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    publication_details: Optional[Dict[str, str]]
    pmid: Optional[str]
    messages: Annotated[list, add_messages]

class PublicationDetailsAgent(BaseAgent[State]):
    """Agent for extracting publication details from PDFs."""

    def __init__(self):
        """Initialize the publication details agent."""
        super().__init__("publication_details", "pmid_cache.json")

    def extract_publication_details(self, state: State) -> State:
        """Extract publication details from PDF using Gemini with caching."""
        pdf_filepath = state["pdf_filepath"]
        pdf_name = Path(pdf_filepath).name

        # Check cache first
        print(f"\nExtracting publication details for PDF: {pdf_name}")
        pmid_cache = self.load_cache()

        if pdf_name in pmid_cache:
            pub_details = pmid_cache[pdf_name]
            print(f"Loaded publication details from cache: {pub_details}")

            # If PMID is in the cache, include it in the state
            if "pmid" in pub_details:
                pmid = pub_details["pmid"]
                print(f"Found PMID in cache: {pmid}")
                return {
                    **state,
                    "publication_details": pub_details,
                    "pmid": pmid,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"Publication details and PMID loaded from cache: {pub_details}"}
                    ]
                }
            else:
                return {
                    **state,
                    "publication_details": pub_details,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": f"Publication details loaded from cache: {pub_details}"}
                    ]
                }

        # Not in cache, extract using AI Processor Client
        try:
            pub_details = self.ai_processor_client.extract_publication_details(pdf_filepath)

            # Extract PMID using PmidExtractor
            pmid = PmidExtractor.get_pmid(
                title=pub_details.get("title"),
                author=pub_details.get("first_author_lastname", ""),
                year=pub_details.get("year", "")
            )

            # If PMID is not available, generate a UUID
            if not pmid:
                pmid = str(uuid.uuid4())
                print(f"Could not extract PMID. Generated UUID instead: {pmid}")
            else:
                print(f"Successfully extracted PMID: {pmid}")

            # Add PMID to publication details
            pub_details["pmid"] = pmid

            # Save to cache
            pmid_cache[pdf_name] = pub_details
            self.save_cache(pmid_cache)

            return {
                **state,
                "publication_details": pub_details,
                "pmid": pmid,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Extracted publication details with PMID: {pub_details}"}
                ]
            }
        except Exception as e:
            error_msg = f"Error extracting publication details: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }

    def _get_pmid(self, state: State) -> State:
        """Get PMID using PmidExtractor with the extracted publication details."""
        pdf_filepath = state["pdf_filepath"]
        pdf_name = Path(pdf_filepath).name
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

        # If PMID is not available, generate a UUID
        if not pmid:
            pmid = str(uuid.uuid4())
            message = f"Could not extract PMID. Generated UUID instead: {pmid}"
        else:
            message = f"Successfully extracted PMID: {pmid}"

        print(message)

        # Update the cache with the PMID
        pmid_cache = self.load_cache()
        if pdf_name in pmid_cache:
            pmid_cache[pdf_name]["pmid"] = pmid
        else:
            # If the PDF is not in the cache yet, create a new entry
            pmid_cache[pdf_name] = {"pmid": pmid}

        # Save the updated cache
        self.save_cache(pmid_cache)
        print(f"Saved PMID {pmid} to cache for {pdf_name}")

        return {
            **state,
            "pmid": pmid,
            "messages": state["messages"] + [
                {"role": "assistant", "content": message}
            ]
        }

    def setup(self):
        """Set up the agent by building the graph."""
        from langgraph.graph import StateGraph, START, END

        # Create a new graph builder
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("extract_publication_details", self.extract_publication_details)

        # Define the edges with conditional routing
        graph_builder.add_edge(START, "extract_publication_details")

        # Always go to END after extract_publication_details since PMID is now extracted there
        graph_builder.add_edge("extract_publication_details", END)

        # Compile the graph
        self.graph = graph_builder.compile()
        return self.graph

    def print_results(self, final_state: State):
        """Print the results of running the agent."""
        print("\n=== Results ===")
        print(f"PDF: {Path(final_state['pdf_filepath']).name}")
        print(f"Publication Details: {final_state.get('publication_details')}")
        print(f"PMID: {final_state.get('pmid')}")

        # Call the base class method to print the conversation
        super().print_results(final_state)

    def save_pdf_with_pmid(self, final_state: State):
        """Save PDF with PMID as filename if PMID was found."""
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
                source_path = Path(final_state['pdf_filepath'])
                target_path = pdf_dir / f"{pmid}.pdf"
                shutil.copy2(source_path, target_path)
                print(f"PDF saved as: {target_path}")
            except Exception as e:
                print(f"Error saving PDF with PMID filename: {e}")
                import traceback
                traceback.print_exc()

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
    agent = PublicationDetailsAgent()
    agent.setup()

    # Initialize the state
    initial_state = {
        "pdf_filepath": pdf_filepath,
        "publication_details": None,
        "pmid": None,
        "messages": [
            {"role": "user", "content": f"Extract publication details and PMID from {Path(pdf_filepath).name}"}
        ]
    }

    try:
        # Run the agent
        final_state = agent.run(initial_state)

        # Display the results
        agent.print_results(final_state)

        # Save PDF with PMID as filename if PMID was found
        agent.save_pdf_with_pmid(final_state)

    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
