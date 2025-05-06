import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, TypeVar, Generic, Callable, Union

from dotenv import load_dotenv
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from ai.ai_processor_client import AIProcessorClient

load_dotenv()

# Define cache directory
CACHE_DIR = Path("cache")
if not CACHE_DIR.exists():
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Cache directory created at: {CACHE_DIR}")
    except Exception as e:
        print(f"Error creating cache directory: {e}")

# Generic type for state
T = TypeVar('T', bound=TypedDict)

class BaseAgent(Generic[T]):
    """Base class for all agents that use GeminiProcessor."""

    def __init__(self, name: str, cache_file: str, pmid: Optional[str] = None):
        """
        Initialize the base agent.

        Args:
            name: Name of the agent
            cache_file: Name of the cache file to use
            pmid: PMID of the document (optional)
        """
        self.name = name
        self.pmid = pmid
        self.ai_processor_client = AIProcessorClient()

        if pmid:
            # Create PMID-specific subdirectory if it doesn't exist
            pmid_dir = CACHE_DIR / pmid
            if not pmid_dir.exists():
                try:
                    pmid_dir.mkdir(parents=True, exist_ok=True)
                    print(f"PMID-specific cache directory created at: {pmid_dir}")
                except Exception as e:
                    print(f"Error creating PMID-specific cache directory: {e}")

            # Use PMID-specific path for patient_cache.json
            self.cache_file_path = pmid_dir / cache_file
        else:
            # Use standard path for other cache files
            self.cache_file_path = CACHE_DIR / cache_file

        self.graph = None


    def load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file_path.exists():
            with open(self.cache_file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_cache(self, cache: Dict[str, Any]):
        """Save cache to file."""
        with open(self.cache_file_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def build_graph(self, state_class: type, nodes: Dict[str, Callable[[T], T]]) -> StateGraph:
        """
        Build the LangGraph for the agent.

        Args:
            state_class: The state class for the graph
            nodes: Dictionary mapping node names to node functions

        Returns:
            The compiled graph
        """
        graph_builder = StateGraph(state_class)

        # Add nodes
        for node_name, node_func in nodes.items():
            graph_builder.add_node(node_name, node_func)

        # Add edges - first node connects to START, last node connects to END
        node_names = list(nodes.keys())
        graph_builder.add_edge(START, node_names[0])

        # Connect nodes in sequence
        for i in range(len(node_names) - 1):
            graph_builder.add_edge(node_names[i], node_names[i + 1])

        # Connect last node to END
        graph_builder.add_edge(node_names[-1], END)

        # Compile the graph
        self.graph = graph_builder.compile()
        return self.graph

    def run(self, initial_state: T) -> T:
        """
        Run the agent with the given initial state.

        Args:
            initial_state: The initial state for the agent

        Returns:
            The final state after running the agent
        """
        if not self.graph:
            raise ValueError("Graph not built. Call build_graph() first.")

        return self.graph.invoke(initial_state)

    def print_results(self, final_state: T):
        """
        Print the results of running the agent.

        Args:
            final_state: The final state after running the agent
        """
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
