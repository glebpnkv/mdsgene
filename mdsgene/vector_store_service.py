# vector_store_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
import sys
import os
from typing import List, Dict, Optional, Any

app = FastAPI()

class DocumentQuery(BaseModel):
    question: str
    document_name: str

class DocumentContent(BaseModel):
    text: str
    source_filename: str
    storage_path: str

# Reuse Ollama configuration constants
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL_NAME = "mxbai-embed-large"

# Define splitter parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Global variables
embedding_model = None
splitter = None

@app.on_event("startup")
def initialize_components():
    """Initialize the embedding model and text splitter."""
    global embedding_model, splitter

    use_vector_store = os.environ.get("USE_VECTOR_STORE", "true").lower() != "false"
    if use_vector_store:
        try:
            # Initialize Embedding Model
            embedding_model = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL_NAME,
                show_progress=True
            )
            print("Ollama Embedding Model client initialized.")

            # Define a Document Splitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                is_separator_regex=False,
            )
            print(f"Recursive Document Splitter initialized (chunkSize={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
        except Exception as e:
            print(f"Error initializing components: {e}", file=sys.stderr)
            raise
    else:
        print("Vector Store initialization skipped (USE_VECTOR_STORE=false).")

def load_vector_store(storage_path: str) -> FAISS:
    """
    Load a vector store from the specified path.

    Args:
        storage_path: Path to the vector store

    Returns:
        The loaded vector store
    """
    try:
        vector_store = FAISS.load_local(storage_path, embedding_model, allow_dangerous_deserialization=True)
        print(f"Vector store loaded successfully from {storage_path}.")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {e}")

def create_empty_vector_store(storage_path: str):
    """
    Create an empty vector store at the specified path.

    Args:
        storage_path: Path where the vector store will be created
    """
    try:
        # Create a vector store with a dummy text to avoid "list index out of range" error
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

        print(f"Empty FAISS index created at {storage_path}")
    except Exception as e:
        print(f"Error creating vector store: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e}")

@app.post("/create_vector_store")
def api_create_vector_store(storage_path: str):
    """
    API endpoint to create an empty vector store.

    Args:
        storage_path: Path where the vector store will be created

    Returns:
        Success message
    """
    use_vector_store = os.environ.get("USE_VECTOR_STORE", "true").lower() != "false"
    if not use_vector_store:
        return {"message": "Vector Store creation skipped (USE_VECTOR_STORE=false)."}

    try:
        # Check if the directory exists, create it if not
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Check if index already exists
        index_file = storage_dir / "index.faiss"
        if index_file.exists():
            return {"message": f"Vector store already exists at {storage_path}"}

        create_empty_vector_store(storage_path)
        return {"message": f"Vector store created successfully at {storage_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_document")
def process_document(document: DocumentContent):
    """
    Process a document: split, embed, and store it in the vector store.

    Args:
        document: Document content, source filename, and storage path

    Returns:
        Success message
    """
    use_vector_store = os.environ.get("USE_VECTOR_STORE", "true").lower() != "false"
    if not use_vector_store:
        return {"message": "Document processing skipped (USE_VECTOR_STORE=false)."}

    if not document.text or document.text.strip() == "":
        raise HTTPException(status_code=400, detail="Cannot process null or empty text")

    storage_path = document.storage_path
    source_filename = document.source_filename

    try:
        # Check if the directory exists, create it if not
        storage_dir = Path(storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Check if index exists, create it if not
        index_file = storage_dir / "index.faiss"
        if not index_file.exists():
            create_empty_vector_store(storage_path)

        # Wrap the text in a LangChain Document object with metadata
        doc = Document(
            page_content=document.text,
            metadata={"source": source_filename}
        )

        # Split the document into chunks
        split_docs = splitter.split_documents([doc])
        print(f"Document split into {len(split_docs)} chunks.")

        if not split_docs:
            raise HTTPException(status_code=400, detail="Document splitting resulted in zero chunks")

        # Load the vector store
        vector_store = load_vector_store(storage_path)

        # Check if document already exists by comparing the filename in metadata
        existing_docs = vector_store.docstore._dict.values()
        document_exists = any(doc.metadata.get("source") == source_filename for doc in existing_docs)

        if document_exists:
            return {"message": f"Document '{source_filename}' already exists in the vector store"}
        else:
            # Add to the existing store
            vector_store.add_documents(split_docs)
            vector_store.save_local(storage_path)
            return {"message": f"Document '{source_filename}' added to vector store successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_document_content(query: DocumentQuery):
    """
    Search for content in a specific document.

    Args:
        query: Question and document name

    Returns:
        The search results
    """
    use_vector_store = os.environ.get("USE_VECTOR_STORE", "true").lower() != "false"
    if not use_vector_store:
        raise HTTPException(status_code=503, detail="Vector Store search disabled (USE_VECTOR_STORE=false)")

    try:
        # The storage path should be provided in the query or determined by the service
        # For now, we'll assume it's stored in an environment variable
        storage_path = os.getenv("VECTOR_STORE_PATH")
        if not storage_path:
            raise HTTPException(status_code=400, detail="Vector store path not configured")

        # Load the vector store
        vector_store = load_vector_store(storage_path)

        # Perform a similarity search for the specified document
        results = vector_store.similarity_search(
            query.question, 
            k=1, 
            filter={"source": query.document_name}
        )

        if results:
            return {"answer": results[0].page_content}
        else:
            raise HTTPException(status_code=404, detail=f"No results found for document '{query.document_name}'")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_with_path")
def search_document_content_with_path(question: str, document_name: str, storage_path: str, k: int = 3):
    """
    Search for content in a specific document with a specified storage path.

    Args:
        question: The question to search for
        document_name: The name of the document to search in
        storage_path: Path to the vector store
        k: Number of top results to return (default is 3)

    Returns:
        The search results as a list
    """
    use_vector_store = os.environ.get("USE_VECTOR_STORE", "true").lower() != "false"
    if not use_vector_store:
        raise HTTPException(status_code=503, detail="Vector Store search disabled (USE_VECTOR_STORE=false)")

    try:
        # Load the vector store
        vector_store = load_vector_store(storage_path)

        # Perform a similarity search for the specified document
        results = vector_store.similarity_search(
            question,
            k=k,
            filter={"source": document_name}
        )

        if results:
            return {
                "answers": [
                    {"page_content": result.page_content, "metadata": result.metadata}
                    for result in results
                ]
            }
        else:
            raise HTTPException(status_code=404, detail=f"No results found for document '{document_name}'")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete_document_from_store")
def delete_document_from_store(document_name: str, storage_path: str):
    """
    Delete a document from the vector store.

    Args:
        document_name: The name of the document to delete
        storage_path: Path to the vector store

    Returns:
        Success message
    """
    use_vector_store = os.environ.get("USE_VECTOR_STORE", "true").lower() != "false"
    if not use_vector_store:
        return {"message": "Vector Store deletion skipped (USE_VECTOR_STORE=false)."}

    try:
        vector_store = load_vector_store(storage_path)

        # Find all document IDs that match the document name
        to_delete_ids = [
            doc_id for doc_id, doc in vector_store.docstore._dict.items()
            if doc.metadata.get("source") == document_name
        ]

        if not to_delete_ids:
            return {"message": f"No entries found for '{document_name}'."}

        # Delete the documents from the vector store
        vector_store.delete(to_delete_ids)

        # Save the updated vector store
        vector_store.save_local(storage_path)

        return {"message": f"Deleted {len(to_delete_ids)} entries for '{document_name}' from vector store."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
