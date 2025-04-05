import sys
from datetime import timedelta

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# Configure your Ollama connection details
OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL_NAME = "llama3" # Example chat model from Java code
EMBEDDING_MODEL_NAME = "nomic-embed-text" # Example embedding model
TIMEOUT = timedelta(seconds=300)


class OllamaClient:
    """Wraps Ollama models (conceptual translation)."""

    def __init__(self):
        try:
            # Initialize Embedding Model Client
            self.embedding_model = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL_NAME,
                # Langchain timeout handling might differ, often set per-request
                # request_timeout=TIMEOUT.total_seconds(),
                 show_progress=True
            )
            print(f"Ollama Embedding Model client initialized for: {EMBEDDING_MODEL_NAME}")

            # Initialize Chat Model Client
            self.chat_model = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=CHAT_MODEL_NAME,
                # request_timeout=TIMEOUT.total_seconds(),
                # temperature=0.7, # Example parameter
                 # num_predict=100, # Example parameter (max tokens)
            )
            print(f"Ollama Chat Model client initialized for: {CHAT_MODEL_NAME}")

        except Exception as e:
            print(f"Error initializing Ollama clients: {e}", file=sys.stderr)
            raise # Re-raise exception to indicate failure

    def get_embedding(self, text: str) -> list[float] | None:
        """Generates an embedding for a given text."""
        try:
            print(f"\nGenerating embedding for: \"{text[:100]}...\"")
            embedding_vector = self.embedding_model.embed_query(text)
            print("Embedding generated successfully.")
            # Langchain embed_query returns List[float] directly
            return embedding_vector # Return the list of floats
        except Exception as e:
            print(f"Error generating embedding: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None

    def chat(self, message: str) -> AIMessage | None:
        """Sends a simple chat message to the LLM."""
        try:
            print(f"\nSending chat message: \"{message[:100]}...\"")
            # Wrap the String message in HumanMessage
            user_message = HumanMessage(content=message)
            # Invoke returns the AIMessage directly
            response_message = self.chat_model.invoke([user_message])
            print("Chat response received successfully.")
            return response_message # Return AIMessage
        except Exception as e:
            print(f"Error during chat generation: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None


# --- Example Usage ---
if __name__ == "__main__":
    try:
        client = OllamaClient()

        # --- Test Embedding ---
        text_to_embed = "This is a sample text to test embeddings."
        embedding_vector = client.get_embedding(text_to_embed)
        if embedding_vector:
            # Print some info about the embedding
            print(f"Embedding Dimension: {len(embedding_vector)}")
            # Print first few dimensions
            print("Embedding Vector (first 5): [", end="")
            print(", ".join(f"{x:.4f}" for x in embedding_vector[:5]), end="")
            print("]" if len(embedding_vector) <= 5 else ", ...]")

        print("\n------------------------------------\n")

        # --- Test Chat ---
        chat_message = "Explain the concept of Retrieval-Augmented Generation (RAG) in simple terms."
        chat_response = client.chat(chat_message)
        if chat_response and isinstance(chat_response, AIMessage):
             print(f"AI Response: {chat_response.content}")

    except Exception as main_e:
        print(f"Failed to run OllamaClient example: {main_e}", file=sys.stderr)
