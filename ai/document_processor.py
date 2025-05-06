import sys
from datetime import timedelta
from pathlib import Path

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS # Using FAISS as in-memory store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# Langchain doesn't have a direct 'Ingestor' class like langchain4j.
# Ingestion is typically done by calling vector_store.add_documents().

from ai.pdf_text_extractor import PdfTextExtractor # Import from the previous file

# Reuse Ollama configuration constants
OLLAMA_BASE_URL = "http://localhost:11434"
# Match embedding model from RagQueryEngine if used together
EMBEDDING_MODEL_NAME = "mxbai-embed-large" # Or "nomic-embed-text" from Java file
TIMEOUT_EMBEDDING = timedelta(seconds=60)

# Define splitter parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


class DocumentProcessor:
    """Handles splitting, embedding, and storing document text."""

    def __init__(self, storage_path=None):
        """Initializes embedding model, splitter, and vector store."""
        try:
            # Initialize Embedding Model
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

            if Path(self._storage_path).exists():
                self.vector_store = FAISS.load_local(self._storage_path, self.embedding_model, allow_dangerous_deserialization=True)
            else:
                # No existing index, set to None and it will be created when documents are added
                self.vector_store = None
            print("FAISS Vector Store placeholder ready.")

        except Exception as e:
            print(f"Error initializing DocumentProcessor: {e}", file=sys.stderr)
            raise

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

        # Wrap the text in a LangChain Document object with metadata
        document = Document(
            page_content=extracted_pdf_text,
            metadata={"source": source_filename}
        )

        try:
            # Split the document into chunks
            split_docs = self.splitter.split_documents([document])
            print(f"Document split into {len(split_docs)} chunks.")

            if not split_docs:
                print("Warning: Document splitting resulted in zero chunks.", file=sys.stderr)
                return

            # If vector store doesn't exist yet, create it
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)
                self.vector_store.save_local(str(self._storage_path))
                print("FAISS index created successfully.")
                return

            # Check if document already exists by comparing the filename in metadata
            # First, get all documents in the store
            existing_docs = self.vector_store.docstore._dict.values()

            # Check if any existing document has the same source filename
            document_exists = any(doc.metadata.get("source") == source_filename for doc in existing_docs)

            if document_exists:
                print(f"Document '{source_filename}' already exists in the vector store. Skipping ingestion.")
                return
            else:
                # Add to the existing store
                self.vector_store.add_documents(split_docs)
                self.vector_store.save_local(str(self._storage_path))
                print(f"Document '{source_filename}' added to existing FAISS index.")

            print("Document ingestion completed successfully.")

        except Exception as e:
            print(f"Error during document ingestion: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def get_vector_store(self) -> FAISS | None:
        """Returns the initialized vector store."""
        return self.vector_store

    def search_document_content(self, question: str, document_name: str):
        """
        Asks a question about a specific document in the vector store.

        Args:
            question: The question to ask.
            document_name: The name of the document to search in.

        Returns:
            The answer to the question.
        """
        if self.vector_store is None:
            print("Error: Vector store is not initialized.", file=sys.stderr)
            return None

        # Perform a similarity search for the specified document
        results = self.vector_store.similarity_search(question, k=1, filter={"source": document_name})

        if results:
            return results[0].page_content
        else:
            print(f"No results found for document '{document_name}'.", file=sys.stderr)
            return None