from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import os
import tempfile
import shutil
import httpx
import asyncio
from pathlib import Path
from uuid import UUID, uuid4
from datetime import datetime
import json

# Set dummy API keys
os.environ["OPENAI_API_KEY"] = "sk-dummy-key-for-local-models"
os.environ["ANTHROPIC_API_KEY"] = "dummy-key-for-local-models"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temporary directory for documents
DOCS_DIR = os.path.join(tempfile.gettempdir(), "paperqa_docs")
os.makedirs(DOCS_DIR, exist_ok=True)
logger.info(f"Documents directory: {DOCS_DIR}")

# Ollama API configuration for direct call
OLLAMA_BASE_URL = "http://localhost:11434/api"
MODEL_NAME = "deepseek-r1:14b"
EMBEDDING_MODEL = "nomic-embed-text"

# Initialization of PaperQA
paperqa_available = False
docs_instance = None
settings_instance = None


def init_paperqa():
    """Initialize PaperQA with proper settings"""
    global paperqa_available, docs_instance, settings_instance

    try:
        # Import required modules from PaperQA
        from paperqa import Settings, Doc, Docs
        from paperqa.settings import AgentSettings
        from paperqa.llms import NumpyVectorStore
        import litellm

        # Disable API key check in litellm
        litellm.drop_params = True

        # Patch embedding handlers for Ollama
        try:
            from litellm.llms.ollama.completion.handler import ollama_aembeddings, ollama_embeddings

            original_ollama_aembeddings = ollama_aembeddings
            original_ollama_embeddings = ollama_embeddings

            async def patched_ollama_aembeddings(*args, **kwargs):
                url = kwargs.get("url", "")
                if url and url.endswith("/api/embed"):
                    kwargs["url"] = url.replace("/api/embed", "/api/embeddings")
                return await original_ollama_aembeddings(*args, **kwargs)

            def patched_ollama_embeddings(*args, **kwargs):
                url = kwargs.get("url", "")
                if url and url.endswith("/api/embed"):
                    kwargs["url"] = url.replace("/api/embed", "/api/embeddings")
                return original_ollama_embeddings(*args, **kwargs)

            litellm.llms.ollama.completion.handler.ollama_aembeddings = patched_ollama_aembeddings
            litellm.llms.ollama.completion.handler.ollama_embeddings = patched_ollama_embeddings

            logger.info("LiteLLM successfully patched for Ollama")
        except Exception as e:
            logger.warning(f"Failed to patch Ollama handlers: {str(e)}")

        # Create an example of settings
        settings_instance = Settings(
            llm=f"ollama/{MODEL_NAME}",
            llm_config={
                "model_list": [
                    {
                        "model_name": f"ollama/{MODEL_NAME}",
                        "litellm_params": {
                            "model": f"ollama/{MODEL_NAME}",
                        }
                    }
                ]
            },
            summary_llm=f"ollama/{MODEL_NAME}",
            summary_llm_config={
                "model_list": [
                    {
                        "model_name": f"ollama/{MODEL_NAME}",
                        "litellm_params": {
                            "model": f"ollama/{MODEL_NAME}",
                        }
                    }
                ]
            },
            embedding=f"ollama/{EMBEDDING_MODEL}",
            agent=AgentSettings(
                agent_llm=f"ollama/{MODEL_NAME}",
                rebuild_index=False  # Disable index rebuilding
            )
        )

        # Create Docs instance
        # In your version of PaperQA, Docs does not accept parameters in constructor
        docs_instance = Docs()

        paperqa_available = True
        logger.info("PaperQA initialized successfully")
        return True
    except Exception as e:
        import traceback
        logger.error(f"Error initializing PaperQA: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        paperqa_available = False
        return False


# Class for storing documents in a simple format (backup storage)
class SimpleDocument:
    def __init__(self, doc_id, description, text, filepath):
        self.doc_id = doc_id
        self.description = description
        self.text = text
        self.filepath = filepath


# List for backup storage
backup_documents = []


# Extract text from files
async def extract_text_from_file(file_path):
    """Extracts text from a file"""
    # Determine file type by extension
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.txt' or ext == '.md':
        # For text files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # If UTF-8 fails, try other encodings
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

    elif ext == '.pdf':
        # For PDF files, use PyPDF2 or pdfplumber
        try:
            # Try importing PyPDF2
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except ImportError:
            try:
                # If PyPDF2 is not available, try pdfplumber
                import pdfplumber
                text = ""
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                return text
            except ImportError:
                # If neither library is installed
                return f"[Unable to extract text from PDF. Please install PyPDF2 or pdfplumber]"
    else:
        # Unsupported file format
        return f"[Unsupported file format: {ext}]"


# Direct query to the Ollama API
async def ollama_query(prompt, context=None):
    """Sends a direct query to the Ollama API"""
    async with httpx.AsyncClient() as client:
        try:
            # If there is context (documents), append it to the query
            if context:
                prompt = f"{context}\n\nQuestion: {prompt}\n\nAnswer:"

            response = await client.post(
                f"{OLLAMA_BASE_URL}/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=300.0
            )

            if response.status_code == 200:
                data = response.json()
                return data.get("response", "")
            else:
                error_msg = f"Error querying Ollama: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return error_msg
        except Exception as e:
            error_msg = f"Exception during Ollama query: {str(e)}"
            logger.error(error_msg)
            return error_msg


# Process query with documents via direct Ollama call
async def query_with_documents(query, documents):
    """Performs a query considering uploaded documents via Ollama"""
    if not documents:
        return await ollama_query(query)

    # Create context from documents
    context = "\n\n".join([
        f"Document: {doc.doc_id}\nDescription: {doc.description}\n\n{doc.text[:2000]}..."
        for doc in documents
    ])

    context_prompt = f"""Use the following documents to answer the question:

{context}"""

    return await ollama_query(query, context=context_prompt)


# Safely add a document to PaperQA
async def add_document_to_paperqa(file_path, description):
    """Safely adds a document to PaperQA"""
    global docs_instance, settings_instance

    if not paperqa_available or not docs_instance:
        return False, "PaperQA is not initialized"

    try:
        from paperqa import Doc
        from paperqa.utils import md5sum

        # In your version of PaperQA, the add method accepts a file path
        path = Path(file_path)
        docname = os.path.basename(file_path)
        dockey = md5sum(path)

        # Create a Doc instance
        citation = f"{description}, {os.path.basename(file_path)}, {datetime.now().year}"

        # Attempt to use the add method to add the document
        result = await docs_instance.aadd(
            path=path,
            citation=citation,
            docname=docname,
            dockey=dockey,
            settings=settings_instance
        )

        if result:
            return True, f"Document {docname} successfully added to PaperQA"
        else:
            return False, "Document was not added to PaperQA (possibly already exists)"
    except Exception as e:
        import traceback
        error_msg = f"Error adding document to PaperQA: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False, error_msg


# Execute query via PaperQA
async def query_paperqa(query, use_docs=True):
    """Executes a query via PaperQA"""
    global docs_instance, settings_instance

    if not paperqa_available or not docs_instance:
        return False, "PaperQA is not initialized"

    try:
        # Use the query method to execute the query
        result = await docs_instance.aquery(
            query=query,
            settings=settings_instance
        )

        # Return the answer
        return True, result.answer
    except Exception as e:
        error_msg = f"Error executing query via PaperQA: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


# API endpoints

@app.post("/upload_document")
async def upload_document(
        file: UploadFile = File(...),
        description: Optional[str] = Form(None)
):
    try:
        logger.info(f"Uploading document: {file.filename}")

        # Create a temporary file
        file_path = os.path.join(DOCS_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Add file to PaperQA
        doc_desc = description or file.filename

        # Extract text for backup storage
        text = await extract_text_from_file(file_path)

        # Add to backup storage
        backup_doc = SimpleDocument(
            doc_id=file.filename,
            description=doc_desc,
            text=text,
            filepath=file_path
        )
        backup_documents.append(backup_doc)

        # Attempt to add to PaperQA if available
        paperqa_success = False
        paperqa_message = ""

        if paperqa_available:
            paperqa_success, paperqa_message = await add_document_to_paperqa(file_path, doc_desc)

        # Formulate response
        result = {
            "status": "success" if paperqa_success else "partial",
            "message": f"Document {file.filename} uploaded successfully" +
                       (" and added to PaperQA" if paperqa_success else ", but not added to PaperQA"),
            "paperqa_status": paperqa_message if not paperqa_success else "OK",
            "backup_storage": True,
            "paperqa_storage": paperqa_success,
            "text_length": len(text),
            "file_path": file_path
        }

        logger.info(f"Document added: {result['message']}")
        return result

    except Exception as e:
        import traceback
        logger.error(f"Error uploading document: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}


@app.get("/documents")
async def get_documents():
    try:
        # Return the list of uploaded documents
        paperqa_docs = []
        if paperqa_available and docs_instance:
            try:
                paperqa_docs = [
                    {
                        "id": i,
                        "name": doc.docname,
                        "citation": doc.citation,
                        "dockey": str(doc.dockey),
                        "source": "paperqa"
                    }
                    for i, doc in enumerate(docs_instance.docs.values())
                ]
            except Exception as e:
                logger.error(f"Error fetching documents list from PaperQA: {str(e)}")

        backup_docs = [
            {
                "id": i,
                "name": doc.doc_id,
                "description": doc.description,
                "source": "backup",
                "text_length": len(doc.text)
            }
            for i, doc in enumerate(backup_documents)
        ]

        return {
            "documents": paperqa_docs + backup_docs,
            "paperqa_available": paperqa_available,
            "backup_count": len(backup_documents),
            "paperqa_count": len(paperqa_docs)
        }
    except Exception as e:
        logger.error(f"Error retrieving documents list: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/document/{doc_id}")
async def get_document_text(doc_id: int):
    try:
        if 0 <= doc_id < len(backup_documents):
            doc = backup_documents[doc_id]
            # Return first 1000 characters of text as a preview
            preview = doc.text[:1000] + "..." if len(doc.text) > 1000 else doc.text
            return {
                "id": doc_id,
                "name": doc.doc_id,
                "description": doc.description,
                "preview": preview,
                "text_length": len(doc.text)
            }
        else:
            return {"status": "error", "message": "Document with specified ID not found"}
    except Exception as e:
        logger.error(f"Error fetching document text: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/ai_prompt")
async def ai_prompt(prompt: str, use_docs: bool = True):
    try:
        logger.info(f"Received query: {prompt}")
        logger.info(f"Use documents: {use_docs}")

        # First, try using PaperQA
        if paperqa_available and docs_instance:
            try:
                success, response = await query_paperqa(prompt)
                if success:
                    logger.info("Query processed via PaperQA")
                    return {"response": response, "source": "paperqa"}
                else:
                    logger.warning(f"PaperQA failed to process query: {response}")
            except Exception as e:
                logger.error(f"Error querying via PaperQA: {str(e)}")
                logger.info("Falling back to direct Ollama call...")

        # If PaperQA is unavailable or there was an error, use direct Ollama call
        logger.info("Using direct Ollama call")

        if use_docs and backup_documents:
            response = await query_with_documents(prompt, backup_documents)
        else:
            response = await ollama_query(prompt)

        logger.info("Query processed via direct Ollama call")
        return {"response": response, "source": "ollama_direct"}

    except Exception as e:
        import traceback
        logger.error(f"Error processing query: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    try:
        # First, check PaperQA
        if paperqa_available and docs_instance and hasattr(docs_instance, "docs"):
            # In your version of PaperQA, documents are stored in the "docs" dictionary
            # We need to find the document by its index in the list of docs.values()
            try:
                docs_list = list(docs_instance.docs.values())
                if 0 <= doc_id < len(docs_list):
                    doc = docs_list[doc_id]
                    if hasattr(doc, "docname") and hasattr(doc, "dockey"):
                        # Use delete method to remove the document
                        docs_instance.delete(docname=doc.docname)
                        logger.info(f"Document deleted from PaperQA: {doc.docname}")
                        return {"status": "success", "message": f"Document {doc.docname} successfully deleted from PaperQA"}
            except Exception as e:
                logger.error(f"Error deleting document from PaperQA: {str(e)}")

        # If document not found in PaperQA, check backup storage
        if 0 <= doc_id < len(backup_documents):
            # Get document info before deletion
            doc_info = {
                "name": backup_documents[doc_id].doc_id,
                "description": backup_documents[doc_id].description
            }

            # Remove document from the list
            backup_documents.pop(doc_id)

            logger.info(f"Document deleted from backup storage: {doc_info['name']}")
            return {"status": "success",
                    "message": f"Document {doc_info['name']} successfully deleted from backup storage"}

        return {"status": "error", "message": "Document with specified ID not found"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/")
async def root():
    """Basic server information"""
    return {
        "server": "PaperQA Server for Custom Implementation",
        "paperqa_available": paperqa_available,
        "endpoints": [
            {"path": "/upload_document", "method": "POST", "description": "Upload document"},
            {"path": "/documents", "method": "GET", "description": "Get documents list"},
            {"path": "/document/{doc_id}", "method": "GET", "description": "Get document text"},
            {"path": "/ai_prompt", "method": "GET", "description": "Query the model"},
            {"path": "/documents/{doc_id}", "method": "DELETE", "description": "Delete document"}
        ],
        "models": {
            "llm": MODEL_NAME,
            "embedding": EMBEDDING_MODEL
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialization on server startup"""
    # Initialize PaperQA
    init_paperqa()

    # Check Ollama availability
    logger.info("Checking available Ollama models...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/tags")

            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_names = [model.get("name", "") for model in models]

                logger.info(f"Available Ollama models: {', '.join(model_names)}")

                # Check for required models
                if not any(MODEL_NAME in model for model in model_names):
                    logger.warning(f"Model {MODEL_NAME} not found! Please install it: ollama pull {MODEL_NAME}")

                if not any(EMBEDDING_MODEL in model for model in model_names):
                    logger.warning(f"Model {EMBEDDING_MODEL} not found! Please install it: ollama pull {EMBEDDING_MODEL}")
            else:
                logger.error(f"Error querying Ollama API: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error checking Ollama models: {str(e)}")
        logger.warning("Ensure that Ollama is running and accessible")


if __name__ == "__main__":
    logger.info("Starting PaperQA server for your implementation...")
    uvicorn.run("custom_paperqa_server:app", host="0.0.0.0", port=8000, reload=True)
