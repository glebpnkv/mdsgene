import sys
from datetime import timedelta
from typing import Optional

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS # Using FAISS as in-memory store
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# Langchain doesn't have a direct 'Ingestor' class like langchain4j.
# Ingestion is typically done by calling vector_store.add_documents().

from pdf_text_extractor import PdfTextExtractor # Import from the previous file

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

    def __init__(self):
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

            # Initialize an In-Memory Vector Store (using FAISS)
            # FAISS needs to be created from documents, so we initialize it as None
            # and create it during process_document.
            self.vector_store: Optional[FAISS] = None
            print("FAISS Vector Store placeholder ready.")

        except Exception as e:
            print(f"Error initializing DocumentProcessor: {e}", file=sys.stderr)
            raise

    def process_document(self, extracted_pdf_text: str, source_filename: str = "unknown.pdf"):
        """
        Processes the extracted text: splits, embeds, and stores it in FAISS.

        Args:
            extracted_pdf_text: The full text extracted from the PDF.
            source_filename: Optional name of the source file for metadata.
        """
        if not extracted_pdf_text or extracted_pdf_text.strip() == "":
            print("Error: Cannot process null or empty text.", file=sys.stderr)
            return

        print("\nStarting document ingestion...")

        # Wrap the text in a LangChain Document object with metadata
        # Metadata is crucial for RAG later if you need source tracking
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

            # Ingest the document chunks: Embeds and stores them in FAISS
            print("Creating/Updating FAISS index...")
            if self.vector_store is None:
                # Create the store from the first batch of documents
                self.vector_store = FAISS.from_documents(split_docs, self.embedding_model)
                print("FAISS index created successfully.")
            else:
                # Add to the existing store
                self.vector_store.add_documents(split_docs, embedding=self.embedding_model)
                print("Documents added to existing FAISS index.")

            print("Document ingestion completed successfully.")

        except Exception as e:
            print(f"Error during document ingestion: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def get_vector_store(self) -> Optional[FAISS]:
        """Returns the initialized vector store."""
        return self.vector_store

# --- Example Usage ---
if __name__ == "__main__":
    # --- Step 1: Get Text (using previous class) ---
    file_path = r"D:\000333\fine_tune_deepseek_r1\test_pdf\ando2012-22991136.pdf" # Change path

    if not Path(file_path).exists():
         print(f"ERROR: Test PDF file not found at: {file_path}")
    else:
        extractor = PdfTextExtractor()
        pdf_text = extractor.extract_text(file_path)

        if pdf_text is None:
            print("Exiting: Could not extract text from PDF.", file=sys.stderr)
        else:
            print(f"PDF Text extracted successfully ({len(pdf_text)} chars).")

            # --- Step 2: Process and Store ---
            try:
                processor = DocumentProcessor()
                processor.process_document(pdf_text, source_filename=Path(file_path).name)

                vector_store = processor.get_vector_store()
                if vector_store:
                    # Optional: Check how many vectors are stored
                    try:
                        # FAISS index might be accessible via `vector_store.index`
                        if hasattr(vector_store, 'index') and vector_store.index:
                             print(f"FAISS index contains {vector_store.index.ntotal} vectors.")
                        else:
                             print("Could not determine the number of vectors in the store.")
                        # You can perform a test similarity search
                        # test_query = "patient information"
                        # results = vector_store.similarity_search(test_query, k=1)
                        # print(f"\nTest search results for '{test_query}': {len(results)}")
                        # if results:
                        #     print(f"Top result metadata: {results[0].metadata}")
                        #     print(f"Top result snippet: {results[0].page_content[:200]}...")

                    except Exception as vs_check_e:
                         print(f"Could not check vector store details: {vs_check_e}")

                    print("Processing finished. Embeddings should be in the in-memory store.")
                else:
                    print("Processing finished, but vector store was not created.", file=sys.stderr)

            except Exception as proc_e:
                 print(f"Failed to initialize or run DocumentProcessor: {proc_e}", file=sys.stderr)