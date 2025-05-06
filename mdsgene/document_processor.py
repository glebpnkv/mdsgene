import sys
from datetime import timedelta
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS # Using FAISS as in-memory store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# Langchain doesn't have a direct 'Ingestor' class like langchain4j.
# Ingestion is typically done by calling vector_store.add_documents().

from ai.internal.pdf_text_extractor_logic import PdfTextExtractorLogic # Import from internal module
from ai.vector_store_client import VectorStoreClient # Import the vector store client

# Load configuration from environment variables with defaults
import os
from dotenv import load_dotenv

# Try to load environment variables from .env file
load_dotenv()

# Ollama configuration constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Match embedding model from RagQueryEngine if used together
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "mxbai-embed-large")
TIMEOUT_EMBEDDING = timedelta(seconds=int(os.getenv("TIMEOUT_EMBEDDING_SECONDS", "60")))

# Define splitter parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


class DocumentProcessor:
    """Handles splitting, embedding, and storing document text."""

    def __init__(self, storage_path: str = None, use_vector_store: bool = True):
        """Initializes embedding model, splitter, and vector store."""
        try:
            # For backward compatibility, we still initialize the embedding model and splitter
            # but they will only be used if the vector store service is not available
            self.embedding_model = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL_NAME,
                # request_timeout=TIMEOUT_EMBEDDING.total_seconds(),
                show_progress=True
            )
            print("Ollama Embedding Model client initialized.")

            # Define a Document Splitter
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            print(f"Recursive Document Splitter initialized (chunkSize={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

            self._storage_path = storage_path
            self._use_vs = use_vector_store and storage_path is not None

            # Initialize the vector store client
            self.vector_store_client = VectorStoreClient()

            if self._use_vs:
                self._init_vector_store()
            else:
                # Skip FAISS
                self.vector_store = None
                print("Vector store disabled.")

        except Exception as e:
            print(f"Error initializing DocumentProcessor: {e}", file=sys.stderr)
            raise

    def _init_vector_store(self):
        """Initialize the vector store."""
        try:
            # Create the vector store if it doesn't exist
            self.vector_store_client.create_vector_store(self._storage_path)
            print("Vector Store initialized via service.")

            # For backward compatibility, we still load the vector store directly
            # This will be used by methods that haven't been updated to use the service
            index_file = Path(self._storage_path) / "index.faiss"
            if index_file.exists():
                self.vector_store = FAISS.load_local(self._storage_path, self.embedding_model, allow_dangerous_deserialization=True)
                print("FAISS Vector Store also loaded directly for backward compatibility.")
            else:
                self.vector_store = None
                print("FAISS Vector Store not loaded directly (file not found).")
        except Exception as e:
            print(f"Error initializing vector store: {e}", file=sys.stderr)
            self.vector_store = None

    def process_document(self, extracted_pdf_text: str, source_filename: str = "unknown.pdf"):
        """
        Processes the extracted text: splits, embeds, and stores it in FAISS.
        Checks if document already exists before adding.

        Args:
            extracted_pdf_text: The full text extracted from the PDF.
            source_filename: Optional name of the source file for metadata.
        """
        if not extracted_pdf_text or extracted_pdf_text.strip() == "":
            print("Error: Cannot process null or empty text.", file=sys.stderr)
            return

        print("\nStarting document ingestion...")

        # Skip vector store operations if disabled
        if not self._use_vs:
            print("Vector store is disabled. Skipping document ingestion.")
            return

        try:
            # Process the document using the vector store service
            response = self.vector_store_client.process_document(
                text=extracted_pdf_text,
                source_filename=source_filename,
                storage_path=self._storage_path
            )

            if "error" in response:
                print(f"Error during document ingestion via service: {response['error']}", file=sys.stderr)
                return

            print(response.get("message", "Document ingestion completed successfully."))

            # For backward compatibility, reload the vector store
            if self.vector_store is not None:
                try:
                    self.vector_store = FAISS.load_local(self._storage_path, self.embedding_model, allow_dangerous_deserialization=True)
                    print("FAISS Vector Store reloaded after document ingestion.")
                except Exception as e:
                    print(f"Error reloading vector store: {e}", file=sys.stderr)

        except Exception as e:
            print(f"Error during document ingestion: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def get_vector_store(self) -> FAISS | None:
        """
        Returns the initialized vector store.

        Note: This method is maintained for backward compatibility.
        New code should use the vector store service directly.
        """
        if not self._use_vs:
            print("Vector store is disabled.", file=sys.stderr)
            return None

        # If the vector store is not initialized, try to load it
        if self.vector_store is None:
            try:
                print("Vector store not initialized. Attempting to load from disk.")
                self.vector_store = FAISS.load_local(self._storage_path, self.embedding_model, allow_dangerous_deserialization=True)
                print("FAISS Vector Store loaded successfully.")
            except Exception as e:
                print(f"Error loading vector store: {e}", file=sys.stderr)
                return None

        return self.vector_store

    @staticmethod
    def create_vector_store(storage_path: str):
        """
        Create an empty vector store at the specified path.

        Args:
            storage_path: Path where the vector store will be created
        """
        try:
            # Create a vector store client
            vector_store_client = VectorStoreClient()

            # Create the vector store using the service
            response = vector_store_client.create_vector_store(storage_path)

            if "error" in response:
                print(f"Error creating vector store via service: {response['error']}", file=sys.stderr)
                # Fallback to direct creation
                print("Falling back to direct vector store creation.")

                # Initialize Embedding Model
                embedding_model = OllamaEmbeddings(
                    base_url=OLLAMA_BASE_URL,
                    model=EMBEDDING_MODEL_NAME,
                    show_progress=True
                )

                # Create a vector store with a dummy text to avoid "list index out of range" error
                # FAISS.from_texts fails with empty lists because it tries to access embeddings[0]
                dummy_text = ["dummy text for initialization"]
                try:
                    # First try with empty list (might work in future versions)
                    empty_texts = []
                    FAISS.from_texts(empty_texts, embedding_model, persist_directory=storage_path)
                except (IndexError, ValueError) as e:
                    # If empty list fails, use dummy text
                    print(f"Creating FAISS index with dummy text: {e}")
                    vs = FAISS.from_texts(dummy_text, embedding_model)
                    # Save to the specified directory
                    vs.save_local(storage_path)
            else:
                print(response.get("message", f"Empty FAISS index created at {storage_path}"))

        except Exception as e:
            print(f"Error creating vector store: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def search_document_content(self, question: str, document_name: str, k: int = 3):
        """
        Asks a question about a specific document in the vector store.

        Args:
            question: The question to ask.
            document_name: The name of the document to search in.
            k: Number of top results to return (default is 3)

        Returns:
            The combined search results as a single string, or None if no results found.
        """
        if not self._use_vs:
            print("Vector store is disabled. Cannot search document content.", file=sys.stderr)
            return None

        # Implement retry mechanism
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Search for content using the vector store service
                print(f"Attempt {attempt + 1}/{max_retries}: Searching document content via vector store service...")
                results = self.vector_store_client.search_document_content_with_path(
                    question=question,
                    document_name=document_name,
                    storage_path=self._storage_path,
                    k=k
                )

                if results:
                    print(f"Vector store service search successful on attempt {attempt + 1}.")
                    # Combine the results into a single string
                    combined_context = "\n\n".join([res["page_content"] for res in results])
                    return combined_context
                else:
                    print(f"No results found for document '{document_name}' via service.", file=sys.stderr)
                    # Don't retry if no results found, break the loop
                    break

            except Exception as e:
                print(f"Error searching document content via service (attempt {attempt + 1}/{max_retries}): {e}", file=sys.stderr)

                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                    # Exponential backoff
                    retry_delay *= 2
                else:
                    print("All retry attempts failed. Falling back to direct vector store access.", file=sys.stderr)

        # Fallback to direct vector store if available
        print("FALLBACK: Attempting to use direct vector store access as service calls failed.")
        if self.vector_store is not None:
            try:
                print("Using local FAISS vector store for search.")
                results = self.vector_store.similarity_search(question, k=k, filter={"source": document_name})
                if results:
                    print("Direct vector store search successful.")
                    # Combine the results into a single string
                    combined_context = "\n\n".join([result.page_content for result in results])
                    return combined_context
                else:
                    print(f"No results found for document '{document_name}' in fallback search.", file=sys.stderr)
            except Exception as fallback_e:
                print(f"CRITICAL ERROR: Fallback search also failed: {fallback_e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
        else:
            print("CRITICAL ERROR: No fallback vector store available. Search failed completely.", file=sys.stderr)

        return None
