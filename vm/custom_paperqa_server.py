# -*- coding: utf-8 -*-
"""
Document Mapping Server with PaperQA Integration
Version: 1.2.2

This FastAPI server provides APIs for:
- Document upload, storage, and management
- Text extraction from various file formats
- Integration with PaperQA for document indexing and querying
- Patient information extraction
- Field mapping and processing with LLM assistance
"""
import asyncio
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import httpx
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Response
from fastapi.middleware.cors import CORSMiddleware
from paperqa import DocDetails
from pydantic import BaseModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

original_inject_clean_doi_url_into_data = DocDetails.inject_clean_doi_url_into_data


def patched_inject_clean_doi_url_into_data(data: dict) -> dict:
    doi_url = data.get("doi_url")
    doi = data.get("doi")
    if isinstance(doi, list):
        if len(doi) == 1:
            doi = doi[0]
        elif len(doi) > 1:
            logger.warning(f"Multiple DOIs found: {doi}. Using the first one.")
            doi = doi[0]
        data["doi"] = doi
    if doi and not doi_url:
        doi_url = "https://doi.org/" + doi
    if doi_url:
        data["doi_url"] = doi_url.replace("http://dx.doi.org/", "https://doi.org/").lower()
    return data

DocDetails.inject_clean_doi_url_into_data = patched_inject_clean_doi_url_into_data
logger.info("Applied DOI type patch to DocDetails.inject_clean_doi_url_into_data.")

# --- Import Mapping Logic ---
try:
    # Assuming mapping_item.py is in the same directory or PYTHONPATH
    from mapping_item import MappingItem, QuestionInfo

    logger.info("Successfully imported MappingItem and QuestionInfo from mapping_item.py")
except ImportError:
    print("CRITICAL ERROR: mapping_item.py not found. Mapping endpoint WILL NOT WORK.")
    # Define dummy classes to prevent NameErrors if import fails
    MappingItem = type('MappingItem', (object,), {
        'field': '', 'question': '', 'mapped_excel_column': '',
        'response_convertion_strategy': '', 'custom_processor': None
    })  # type: ignore
    QuestionInfo = type('QuestionInfo', (object,), {
        'field': '', 'query': '', 'response_convertion_strategy': '',
        'family_id': None, 'patient_id': ''
    })  # type: ignore
    logger.error("Failed to import mapping_item.py - using dummy classes")

# --- Global State Flags (Single Worker Environment ONLY) ---
is_processing_globally: bool = False
cancel_requested_globally: bool = False

# --- Configuration ---
# ++ Persistence Paths ++
PERSISTENT_DATA_DIR_STARTUP = Path(os.getenv("PERSISTENT_DATA_DIR", "./persistent_data"))
UPLOADED_FILES_DIR_STARTUP = PERSISTENT_DATA_DIR_STARTUP / "uploaded_files"
BACKUP_STORAGE_FILE_STARTUP = PERSISTENT_DATA_DIR_STARTUP / "backup_storage.json"

# ++ LLM Configuration ++
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://localhost:11434/api")
if OLLAMA_API_BASE_URL.endswith('/api'):
    OLLAMA_SERVICE_BASE_URL = OLLAMA_API_BASE_URL[:-4]
else:
    OLLAMA_SERVICE_BASE_URL = OLLAMA_API_BASE_URL

PAPERQA_MODEL_NAME = os.getenv("PAPERQA_MODEL_NAME", "deepseek-r1:14b")
FORMATTER_MODEL_NAME = os.getenv("FORMATTER_MODEL_NAME", "mistral")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300.0"))

# Log configuration values at startup
logger.info(f"Persistent Data Directory (Startup): {PERSISTENT_DATA_DIR_STARTUP.resolve()}")
logger.info(f"Uploaded Files Directory (Startup): {UPLOADED_FILES_DIR_STARTUP.resolve()}")
logger.info(f"Backup Storage File (Startup): {BACKUP_STORAGE_FILE_STARTUP.resolve()}")
logger.info(f"Ollama API Base URL: {OLLAMA_API_BASE_URL}")
logger.info(f"Ollama Service Base URL (derived): {OLLAMA_SERVICE_BASE_URL}")
logger.info(f"PaperQA/Main Query Model: {PAPERQA_MODEL_NAME}")
logger.info(f"Formatter Model: {FORMATTER_MODEL_NAME}")
logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
logger.info(f"Ollama Request Timeout: {OLLAMA_TIMEOUT}s")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Document Mapping Server (Persistent)",
    description="Uploads documents, uses persistent PaperQA for retrieval, and a separate LLM for formatting.",
    version="1.2.2" # Incremented version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

# --- Exception Classes ---
class CancelRequestException(Exception):
    """Raised when a processing operation is cancelled by user request."""
    pass


# Check for cancel request function
def check_cancel_request():
    """Checks if cancellation has been requested and raises exception if so."""
    if cancel_requested_globally:
        logger.warning("Cancel check: Request found! Raising exception.")
        raise CancelRequestException("Operation cancelled by user request.")


# --- PaperQA Initialization and State ---
paperqa_available = False
docs_instance = None
settings_instance = None


# --- Pydantic Models ---
class DocumentInfo(BaseModel):
    id: int
    filename: str
    description: str
    status: str = 'uploaded'
    paperqa_dockey: Optional[str] = None


class MappingItemInfo(BaseModel):
    field: str
    question: str
    column: Optional[str] = None


class ProcessFieldRequest(BaseModel):
    field: str


class ProcessFieldResponse(BaseModel):
    field: str
    value: str
    raw_answer: Optional[str] = None


class ProcessPatientFieldRequest(BaseModel):
    field: str
    patient_id: str
    family_id: Optional[str] = None


class ProcessPatientFieldResponse(BaseModel):
    field: str
    patient_id: str
    family_id: Optional[str] = None
    value: str
    raw_answer: Optional[str] = None


class PatientInfo(BaseModel):
    family_id: Optional[str] = None
    patient_id: str
    display_name: str


# --- Document Storage Class ---
class SimpleDocument:
    """Stores basic info and text content of an uploaded document."""

    def __init__(self, doc_id: str, description: str, text: str, filepath: str, paperqa_dockey: Optional[str] = None):
        self.doc_id: str = doc_id
        self.description: str = description
        self.text: str = text
        self.filepath: str = filepath
        self.paperqa_dockey: Optional[str] = paperqa_dockey

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "description": self.description,
            "text": self.text,
            "filepath": self.filepath,
            "paperqa_dockey": self.paperqa_dockey,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'SimpleDocument':
        required_keys = ["doc_id", "description", "text", "filepath"]
        if not all(k in data for k in required_keys):
            missing = [k for k in required_keys if k not in data]
            raise ValueError(f"Missing required keys in SimpleDocument data: {missing}")
        return SimpleDocument(
            doc_id=str(data["doc_id"]),
            description=str(data["description"]),
            text=str(data["text"]),
            filepath=str(data["filepath"]),
            paperqa_dockey=data.get("paperqa_dockey")
        )


# --- Global Storage ---
documents_storage: List[SimpleDocument] = []
COMMON_FIELDS = ["pmid", "author, year", "comments_study"]


async def _symptom_list_processor(raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
    """Parses a raw LLM response for symptoms."""
    results: Dict[str, str] = {}
    logger.debug(f"Running symptom processor. Raw input: '{str(raw_answer)[:100]}...'")

    # Check for empty or "none" responses
    if not raw_answer or raw_answer.strip().lower() in ["", "none", "not specified", "none reported",
                                                        "n/a", "no symptoms listed", "no key symptoms reported"]:
        logger.info("Symptom list raw answer is empty or indicates none found.")
        return results

    symptoms = re.split(r'[;\n]+', raw_answer.strip())
    found_structured_entry = False

    for entry in symptoms:
        entry = entry.strip()
        if not entry:
            continue

        parts = entry.rsplit(":", 1)

        # Handle special case of a single "None" entry
        if len(parts) == 1 and len(symptoms) == 1 and entry.lower() in ["none", "n/a", "not applicable", "no symptoms"]:
            logger.info("Symptom list explicitly 'None'.")
            found_structured_entry = True
            return {}

        # Process regular symptom entries in "Name: Status" format
        elif len(parts) == 2:
            name_raw = parts[0].strip()
            presence_raw = parts[1].strip().lower()

            # Clean the name to create a valid column identifier
            name_clean = re.sub(r"^\s*[\*\-\•\d\.]+\s*", "", name_raw).lower()
            name_clean = re.sub(r'\s+', '_', name_clean)
            name_clean = re.sub(r'[^\w_]+', '', name_clean).strip('_')

            if not name_clean:
                logger.warning(f"Could not derive valid column name from: '{entry}'")
                continue

            col_name = f"{name_clean}_sympt"
            col_value = "-99"

            # Map presence indicators to standard values
            if presence_raw in ["yes", "present", "positive", "true", "1", "observed", "reported", "affected"]:
                col_value = "yes"
            elif presence_raw in ["no", "absent", "negative", "false", "0", "not present",
                                  "not observed", "ruled out", "unaffected", "normal"]:
                col_value = "no"
            else:
                logger.warning(f"Unrecognized presence '{presence_raw}' for '{name_raw}'. Setting '{col_name}' to -99.")

            # Handle duplicates
            if col_name in results:
                logger.warning(
                    f"Duplicate symptom column '{col_name}'. Overwriting '{results[col_name]}' with '{col_value}'.")

            results[col_name] = col_value
            found_structured_entry = True
            logger.debug(f"Parsed symptom: '{name_raw}' -> Column: '{col_name}', Value: '{col_value}'")
        else:
            logger.warning(f"Could not parse symptom entry: '{entry}'")

    # Warning if no structured entries found
    if not found_structured_entry and raw_answer and raw_answer.strip().lower() not in ["none", "n/a"]:
        logger.warning(f"Symptom processor found no structured entries: '{raw_answer[:100]}...'")

    logger.info(f"Symptom processor generated {len(results)} columns.")
    return results


async def _hpo_symptom_processor(raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
    """Parses a raw LLM response for HPO terms."""
    results: Dict[str, str] = {}
    logger.debug(f"Running HPO processor. Raw input: '{str(raw_answer)[:100]}...'")

    # Check for empty or "none" responses
    if not raw_answer or raw_answer.strip().lower() in [
        "", "none", "not specified", "none reported", "n/a",
        "no hpo terms", "no hpo findings", "no hpo codes mentioned"
    ]:
        logger.info("HPO list raw answer is empty or indicates none found.")
        return results

    # Regular expression to match HPO terms in the format "Term Name (HP:XXXXXXX): Yes/No"
    pattern = re.compile(
        r"([a-zA-Z\s\(\)\-\+]+?)?\s*\(?\s*(HP[:_\s]?\d{7})\s*\)?\s*[:\-\s\(\)]+\s*"
        r"(yes|no|present|absent|positive|negative|true|false|1|0)\b",
        re.IGNORECASE
    )

    found_match = False
    entries = re.split(r'[;\n]+', raw_answer.strip())

    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        # Handle special "None" entries
        if entry.lower() in ["none", "n/a", "not applicable", "no hpo findings"]:
            logger.info("Skipping 'None' entry in HPO list.")
            if len(entries) == 1:
                return {}
            continue

        # Find HP terms using regex
        matches = pattern.findall(entry)
        if not matches:
            logger.warning(f"No HPO pattern match in entry: '{entry}'")
            continue

        for match in matches:
            try:
                hpo_name = match[0].strip() if match[0] else "Unknown Name"
                id_raw = match[1]
                presence_raw = match[2].lower()

                # Extract and normalize HP ID
                digits = re.sub(r'[^0-9]', '', id_raw)
                if len(digits) == 7:
                    hpo_id_normalized = f"HP_{digits}"
                else:
                    logger.warning(f"Invalid HPO ID format: '{id_raw}' (from '{hpo_name}'). Skipping.")
                    continue

                # Set column name and value
                col_name = hpo_id_normalized
                col_value = "yes" if presence_raw in ["yes", "present", "positive", "true", "1"] else "no"

                # Handle duplicates
                if col_name in results:
                    logger.warning(
                        f"Duplicate HPO column '{col_name}'. Overwriting '{results[col_name]}' with '{col_value}'.")

                results[col_name] = col_value
                found_match = True
                logger.debug(
                    f"Parsed HPO: Name='{hpo_name}', ID='{id_raw}' -> Column: '{col_name}', Value: '{col_value}'")

            except Exception as proc_err:
                logger.error(f"Error processing HPO match. Entry: '{entry}'. Match: {match}. Error: {proc_err}",
                             exc_info=True)
                continue

    # Warning if no matches found
    if not found_match and raw_answer and raw_answer.strip().lower() not in ["none", "n/a"]:
        logger.warning(f"HPO processor found no matching entries: '{raw_answer[:100]}...'")

    logger.info(f"HPO processor generated {len(results)} columns.")
    return results

# --- PaperQA Initialization Function ---
def init_paperqa():
    """
    Initialize PaperQA, attempting to load from persistent state defined by PQA_HOME.
    Reads PERSISTENT_DATA_DIR environment variable at runtime for consistency if PQA_HOME is not set.
    """
    global paperqa_available, docs_instance, settings_instance # Ensure settings_instance is global
    logger.info("Attempting to initialize PaperQA with persistence...")

    # --- Determine PaperQA Home Path ---
    pqa_home_env = os.getenv("PQA_HOME")
    if pqa_home_env:
        paperqa_storage_root = Path(pqa_home_env)
        logger.info(f"Using PaperQA storage root from PQA_HOME environment variable: {paperqa_storage_root.resolve()}")
    else:
        current_persistent_data_dir = Path(os.getenv("PERSISTENT_DATA_DIR", "./persistent_data"))
        paperqa_storage_root = current_persistent_data_dir / "pqa_data"
        logger.warning(
            f"PQA_HOME environment variable not set. "
            f"Falling back to default path within persistent data: {paperqa_storage_root.resolve()}"
        )
        os.environ["PQA_HOME"] = str(paperqa_storage_root)

    try:
        paperqa_storage_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured PaperQA storage directory exists: {paperqa_storage_root}")
    except OSError as e:
        logger.error(
            f"Failed to create PaperQA storage directory '{paperqa_storage_root}': {e}. "
            f"PaperQA initialization likely to fail."
        )
        paperqa_available = False
        return False

    # --- Initialize PaperQA Components ---
    try:
        from paperqa import Settings, Docs
        import litellm

        ollama_service_base = OLLAMA_SERVICE_BASE_URL

        litellm.set_verbose = False
        litellm.drop_params = True

        # LiteLLM Patching
        try:
            from litellm.llms.ollama.completion.handler import ollama_aembeddings, ollama_embeddings
            original_ollama_aembeddings = ollama_aembeddings
            original_ollama_embeddings = ollama_embeddings

            async def patched_ollama_aembeddings(*args, **kwargs):
                kwargs["api_base"] = ollama_service_base
                logger.debug(f"Patched ollama_aembeddings called with api_base: {ollama_service_base}")
                return await original_ollama_aembeddings(*args, **kwargs)

            def patched_ollama_embeddings(*args, **kwargs):
                kwargs["api_base"] = ollama_service_base
                logger.debug(f"Patched ollama_embeddings called with api_base: {ollama_service_base}")
                return original_ollama_embeddings(*args, **kwargs)

            litellm.llms.ollama.completion.handler.ollama_aembeddings = patched_ollama_aembeddings
            litellm.llms.ollama.completion.handler.ollama_embeddings = patched_ollama_embeddings
            logger.info("LiteLLM Ollama embedding handlers patched successfully.")
        except ImportError:
            logger.warning("Could not import LiteLLM Ollama handlers for patching.")
        except Exception as patch_err:
            logger.warning(f"Failed to patch LiteLLM Ollama handlers: {patch_err}.")

        # Define PaperQA Settings - store globally
        settings_instance = Settings( # Assign to global settings_instance
            llm=f"ollama/{PAPERQA_MODEL_NAME}",
            llm_config={"model": f"ollama/{PAPERQA_MODEL_NAME}", "api_base": ollama_service_base},
            summary_llm=f"ollama/{PAPERQA_MODEL_NAME}",
            summary_llm_config={"model": f"ollama/{PAPERQA_MODEL_NAME}", "api_base": ollama_service_base},
            embedding=f"ollama/{EMBEDDING_MODEL}",
            embedding_config={"model": f"ollama/{EMBEDDING_MODEL}", "api_base": ollama_service_base},
        )
        logger.info("PaperQA Settings configured and stored globally.")

        # Initialize Docs object WITHOUT the 'settings' parameter.
        logger.info(f"Initializing PaperQA Docs (using PQA_HOME: '{os.environ.get('PQA_HOME')}')")
        # REMOVED settings=settings_instance from the constructor call
        docs_instance = Docs()  # No 'name' or 'settings' argument here

        loaded_count = len(docs_instance.docs) if docs_instance.docs else 0
        logger.info(f"PaperQA initialized. Loaded {loaded_count} documents from index(es) found in PQA_HOME.")

        paperqa_available = True
        return True

    except ImportError as e:
        logger.error(f"PaperQA or LiteLLM import failed: {e}. PaperQA features will be disabled.", exc_info=True)
        paperqa_available = False
        return False
    except Exception as e:
        # Catch the specific validation error if needed, or general exceptions
        logger.error(f"Error initializing PaperQA: {e}", exc_info=True)
        paperqa_available = False
        return False


# --- Document Storage Functions ---
def save_documents_storage():
    """Saves the current state of `documents_storage` list to a JSON backup file."""
    current_persistent_data_dir = Path(os.getenv("PERSISTENT_DATA_DIR", "./persistent_data"))
    backup_file_name = "backup_storage.json"
    current_backup_storage_file = current_persistent_data_dir / backup_file_name
    logger.info(f"Attempting to save documents storage state to {current_backup_storage_file}...")

    try:
        current_persistent_data_dir.mkdir(parents=True, exist_ok=True)
        data_to_save = [doc.to_dict() for doc in documents_storage]

        with open(current_backup_storage_file, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)

        logger.info(f"Successfully saved {len(documents_storage)} document entries to {current_backup_storage_file}.")
    except Exception as e:
        logger.error(f"Failed to save documents storage to {current_backup_storage_file}: {e}", exc_info=True)


def load_documents_storage():
    """Loads the `documents_storage` list from the JSON backup file on startup."""
    global documents_storage
    current_persistent_data_dir = Path(os.getenv("PERSISTENT_DATA_DIR", "./persistent_data"))
    backup_file_name = "backup_storage.json"
    current_backup_storage_file = current_persistent_data_dir / backup_file_name

    if current_backup_storage_file.exists():
        logger.info(f"Loading documents storage from {current_backup_storage_file}...")
        try:
            with open(current_backup_storage_file, 'r', encoding='utf-8') as f:
                data_loaded = json.load(f)

            loaded_docs = []
            for i, item in enumerate(data_loaded):
                try:
                    loaded_docs.append(SimpleDocument.from_dict(item))
                except (ValueError, TypeError, KeyError) as item_err:
                    logger.error(
                        f"Skipping invalid document entry at index {i} in backup file: {item_err}. Data: {item}")

            documents_storage = loaded_docs
            logger.info(
                f"Successfully loaded {len(documents_storage)} document entries from {current_backup_storage_file}.")
        except Exception as e:
            logger.error(
                f"Failed to load documents storage from {current_backup_storage_file}: {e}",
                exc_info=True
            )
            documents_storage = []
    else:
        logger.info(
            f"Backup storage file not found at {current_backup_storage_file}. Starting with empty documents storage.")
        documents_storage = []


# --- Text Extraction Function ---
async def extract_text_from_file(file_path: str) -> str:
    """Extracts text content from various file types (PDF, TXT, MD)."""
    file_path_obj = Path(file_path)
    if not file_path_obj.is_file():
        logger.error(f"File not found for text extraction: {file_path}")
        return f"[Error: File not found at '{file_path}']"

    ext = file_path_obj.suffix.lower()
    logger.info(f"Attempting text extraction from: {file_path} (type: {ext})")

    try:
        # Handle text files
        if ext in ['.txt', '.md']:
            try:
                return file_path_obj.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decoding failed for {file_path}. Trying latin-1 encoding.")
                return file_path_obj.read_text(encoding='latin-1')
            except Exception as txt_err:
                logger.error(f"Error reading text file {file_path}: {txt_err}", exc_info=True)
                return f"[Error: Could not read text file '{ext}']"

        # Handle PDF files
        elif ext == '.pdf':
            text = ""
            try:
                # Try pypdf first
                from pypdf import PdfReader
                reader = PdfReader(file_path)

                # Handle encryption
                if reader.is_encrypted:
                    logger.warning(
                        f"PDF '{file_path_obj.name}' is encrypted. Attempting to decrypt with empty password.")
                    try:
                        if reader.decrypt("") == 0:
                            logger.error(f"Failed to decrypt PDF '{file_path_obj.name}' with empty password.")
                            return "[Error: Encrypted PDF could not be decrypted]"
                        else:
                            logger.info(f"Successfully decrypted PDF '{file_path_obj.name}'.")
                    except Exception as decrypt_err:
                        logger.error(f"Error during decryption attempt for '{file_path_obj.name}': {decrypt_err}")
                        return "[Error: Failed during PDF decryption attempt]"

                # Extract text from each page
                num_pages = len(reader.pages)
                logger.info(f"Extracting text from {num_pages} pages using pypdf...")

                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as page_err:
                        logger.warning(
                            f"pypdf failed to extract text from page {i + 1} of '{file_path_obj.name}': {page_err}")

                logger.info(f"pypdf extraction complete for '{file_path_obj.name}'. Total text length: {len(text)}")
                return text

            except ImportError:
                # Fallback to PyPDF2
                logger.warning("pypdf library not found. Falling back to PyPDF2 for PDF extraction.")
                try:
                    import PyPDF2
                    text = ""

                    with open(file_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)

                    # Handle encryption with PyPDF2
                    if reader.is_encrypted:
                        logger.warning(f"PyPDF2 detected encryption in '{file_path_obj.name}'. Attempting decrypt.")
                        try:
                            if reader.decrypt("") == 0:
                                logger.error(f"PyPDF2 failed to decrypt '{file_path_obj.name}'.")
                                return "[Error: Encrypted PDF (PyPDF2)]"
                            else:
                                logger.info(f"PyPDF2 decrypted '{file_path_obj.name}'.")
                        except Exception as decrypt_err_pypdf2:
                            logger.error(f"PyPDF2 decryption error for '{file_path_obj.name}': {decrypt_err_pypdf2}")
                            return "[Error: PyPDF2 decryption failed]"

                    # Extract text with PyPDF2
                    num_pages = len(reader.pages)
                    logger.info(f"Extracting text from {num_pages} pages using PyPDF2...")

                    for i, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text + "\n"
                        except Exception as page_err_pypdf2:
                            logger.warning(
                                f"PyPDF2 failed to extract text from page {i + 1} "
                                f"of '{file_path_obj.name}': {page_err_pypdf2}")

                    logger.info(
                        f"PyPDF2 extraction complete for '{file_path_obj.name}'. Total text length: {len(text)}")
                    return text

                except ImportError:
                    logger.error("Neither pypdf nor PyPDF2 library is installed. Cannot process PDF files.")
                    return "[Error: No PDF processing library (pypdf or PyPDF2) found]"
                except Exception as pypdf2_err:
                    logger.error(f"Error during PyPDF2 PDF processing for '{file_path_obj.name}': {pypdf2_err}",
                                 exc_info=True)
                    return "[Error: PyPDF2 library failed during processing]"

            except Exception as pdf_lib_err:
                logger.error(f"Error during pypdf PDF processing for '{file_path_obj.name}': {pdf_lib_err}",
                             exc_info=True)
                return "[Error: pypdf library failed during processing]"
        else:
            logger.warning(f"Unsupported file format '{ext}' for file: {file_path}")
            return f"[Error: Unsupported file format '{ext}']"

    except Exception as e:
        logger.error(f"Unexpected error during text extraction for {file_path}: {e}", exc_info=True)
        return "[Error: Unexpected text extraction failure]"


# --- PaperQA Document Functions ---
async def add_document_to_paperqa(doc_to_add: SimpleDocument) -> Optional[str]:
    """Adds a document to the global PaperQA instance. Returns dockey if successful."""
    if not paperqa_available or not docs_instance:
        logger.warning(
            f"PaperQA service is not available or not initialized. Cannot add document '{doc_to_add.doc_id}'.")
        return None

    try:
        from paperqa.utils import md5sum
        file_path = Path(doc_to_add.filepath)

        if not file_path.exists():
            logger.error(f"File path does not exist for document '{doc_to_add.doc_id}': {file_path}")
            return None

        try:
            expected_dockey = md5sum(file_path)
        except Exception as md5_err:
            logger.error(f"Failed to calculate md5sum for {file_path}: {md5_err}", exc_info=True)
            return None

        docname = doc_to_add.doc_id

        # Check if document already exists
        if expected_dockey in docs_instance.docs:
            logger.info(f"Document '{docname}' (key: {expected_dockey}) already exists in PaperQA index. Skipping add.")
            if doc_to_add.paperqa_dockey != expected_dockey:
                logger.warning(
                    f"Updating stored dockey for '{docname}' from '{doc_to_add.paperqa_dockey}' to '{expected_dockey}'.")
                doc_to_add.paperqa_dockey = expected_dockey
            return expected_dockey

        citation = f"{doc_to_add.description}, {docname}, {datetime.now().year}"
        logger.info(f"Attempting to add to PQA: name='{docname}', path='{file_path}', expected_key='{expected_dockey}'")
        logger.debug(f"Current PQA keys before add: {list(docs_instance.docs.keys())}")

        added_successfully = False
        try:
            # Helper function to ensure values are strings
            def ensure_string(value):
                if isinstance(value, list):
                    return " ".join(str(item) for item in value)
                return str(value) if value is not None else ""

            # Get title with safe fallback if the attribute doesn't exist
            title = ensure_string(getattr(doc_to_add, 'title', docname))

            # Create custom metadata where all fields are guaranteed to be strings
            extra_fields = {
                "title": title,
                "year": str(datetime.now().year)
            }

            # Add DOI if it exists, ensuring it's a string
            if hasattr(doc_to_add, 'doi'):
                extra_fields["doi"] = ensure_string(doc_to_add.doi)

            # Make sure all other potential metadata fields are strings
            for attr in dir(doc_to_add):
                if not attr.startswith('_') and attr not in ['title', 'year', 'doi']:
                    value = getattr(doc_to_add, attr, None)
                    if value is not None and not callable(value) and not isinstance(value, (dict, list, tuple)) or \
                            isinstance(value, (list, tuple)):
                        extra_fields[attr] = ensure_string(value)

            # Pass settings_instance to the add/aadd methods
            if hasattr(docs_instance, 'aadd'):
                await docs_instance.aadd(
                    path=file_path, citation=citation, docname=docname,
                    settings=settings_instance, extra_fields=extra_fields
                )
                added_successfully = True
            elif hasattr(docs_instance, 'add'):
                logger.info("Using synchronous PaperQA 'add' method in thread executor.")
                await asyncio.to_thread(
                    docs_instance.add, path=file_path, citation=citation, docname=docname,
                    settings=settings_instance, extra_fields=extra_fields
                )
                added_successfully = True
            else:
                logger.error("PQA Docs object has no recognized 'add' or 'aadd' method.")
                return None
        except Exception as add_err:
            logger.error(
                f"Error occurred during PaperQA add/aadd execution for '{docname}': {add_err}",
                exc_info=True
            )
            logger.debug(f"PQA keys after failed add attempt: {list(docs_instance.docs.keys())}")
            return None

        logger.debug(f"PQA keys after add attempt: {list(docs_instance.docs.keys())}")

        if added_successfully:
            if expected_dockey in docs_instance.docs:
                logger.info(f"Document '{docname}' successfully added to PaperQA with key: {expected_dockey}")
                doc_to_add.paperqa_dockey = expected_dockey
                return expected_dockey
            else:
                logger.error(
                    f"Document '{docname}' add process completed BUT key '{expected_dockey}' was NOT found "
                    f"in PaperQA index afterwards. Possible internal PQA error or persistence issue."
                )

                # Try to find key by docname
                found_key = None
                for key, doc_obj in docs_instance.docs.items():
                    if getattr(doc_obj, 'docname', None) == docname:
                        found_key = key
                        break

                if found_key:
                    logger.warning(
                        f"Found key '{found_key}' by looking up docname '{docname}'. "
                        f"Using this key, but it differs from expected md5sum '{expected_dockey}'."
                    )
                    doc_to_add.paperqa_dockey = found_key
                    return found_key
                else:
                    logger.error(f"Could not verify addition of '{docname}' by key or docname.")
                    return None
        else:
            logger.error(
                f"Add process for '{docname}' did not complete successfully (no exception, but flag not set)."
            )
            return None

    except ImportError as e:
        logger.error(f"Failed to import paperqa.utils: {e}. Cannot calculate/verify dockey.", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error in add_document_to_paperqa for '{doc_to_add.doc_id}': {e}", exc_info=True)
        return None


async def query_paperqa(query: str, target_doc_name: Optional[str] = None) -> str:
    """Queries the global PaperQA instance. Returns raw answer string or error message."""
    if not paperqa_available or not docs_instance or not settings_instance:
        logger.warning("PQA not available.")
        return "Error: PaperQA service unavailable."

    try:
        final_query = query
        if target_doc_name:
            final_query = f"Based *only* on the document named '{target_doc_name}', answer: {query}"
            logger.info(f"Querying PQA (focused on '{target_doc_name}'): '{query[:100]}...'")
        else:
            logger.info(f"Querying PQA (all docs): '{query[:100]}...'")

        # Pass the global settings_instance
        if hasattr(docs_instance, 'aquery'):
            result = await docs_instance.aquery(query=final_query, settings=settings_instance)
        elif hasattr(docs_instance, 'query'):
            result = await asyncio.to_thread(docs_instance.query, query=final_query, settings=settings_instance)
        else:
            logger.error("PQA Docs has no 'query'/'aquery' method.")
            return "Error: PQA query method not found."

        answer_text = getattr(result, 'answer', None)
        references = getattr(result, 'references', "")

        if answer_text is None:
            logger.warning(f"PQA returned no answer for query: '{query[:50]}...'")
            return "Error: PQA returned no answer."

        answer_text = str(answer_text).strip()
        logger.info(f"PQA Raw Answer received ({len(answer_text)} chars): {answer_text[:200]}...")

        if references:
            logger.info(f"PQA References: {references}")
            if target_doc_name and target_doc_name not in references:
                logger.warning(
                    f"PQA references ('{references}') might not include target "
                    f"('{target_doc_name}') for query '{query[:50]}...'."
                )
        else:
            logger.info("PQA provided no references for this query.")

        return answer_text

    except Exception as e:
        logger.error(f"Error querying PQA: {e}", exc_info=True)
        return f"Error: PQA query failed - {type(e).__name__}: {e}"


async def ollama_query(prompt: str, context: Optional[str] = None, model: str = FORMATTER_MODEL_NAME) -> str:
    """Sends a direct query to Ollama API (primarily for formatting)."""
    # Prepare the prompt with optional context
    full_prompt = prompt
    if context:
        full_prompt = (
            f"Context:\n\"\"\"\n{context}\n\"\"\"\n\nBased ONLY on the context provided above, "
            f"answer:\n\nQuestion: {prompt}\n\nAnswer:"
        )

    logger.info(f"Sending direct query to Ollama '{model}'. Prompt len: {len(full_prompt)}")
    logger.debug(f"Ollama Query Full Prompt:\n---\n{full_prompt}\n---")

    # Send the request to Ollama API
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT + 10.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_API_BASE_URL}/generate",
                json={"model": model, "prompt": full_prompt, "stream": False},
                timeout=OLLAMA_TIMEOUT
            )
            response.raise_for_status()

            # Process successful response
            data = response.json()
            result_text = data.get("response", "").strip()

            # Calculate stats for logging
            response_time_ns = data.get('total_duration')
            response_time_s = f"{(response_time_ns / 1e9):.2f}s" if response_time_ns else "N/A"
            eval_count = data.get('eval_count', 'N/A')
            eval_duration_s = f"{(data.get('eval_duration', 0) / 1e9):.2f}s" if data.get('eval_duration') else "N/A"

            logger.info(
                f"Ollama direct response received from '{model}'. Length: {len(result_text)}. "
                f"Time: {response_time_s} (Eval: {eval_count} tokens, {eval_duration_s})"
            )
            logger.debug(f"Ollama Raw Response Text:\n---\n{result_text}\n---")

            return result_text

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (4xx, 5xx)
            error_body = ""
            try:
                error_body = e.response.text
            except Exception:
                error_body = "(Could not read error response body)"

            error_msg = (f"Ollama API Error: Status {e.response.status_code} calling {e.request.url}. "
                         f"Response: {error_body[:500]}")
            logger.error(error_msg)
            return (f"Error: Ollama request failed - Status {e.response.status_code}. "
                    f"Check model name ('{model}') and Ollama server.")

        except httpx.TimeoutException as e:
            # Handle timeout errors
            error_msg = (f"Ollama API Error: Request timed out after {OLLAMA_TIMEOUT}s "
                         f"for model '{model}' at {e.request.url}.")
            logger.error(error_msg)
            return f"Error: Ollama request timed out ({OLLAMA_TIMEOUT}s)"

        except httpx.RequestError as e:
            # Handle network/connection errors
            error_msg = f"Ollama Network Error: Could not connect to Ollama service at {e.request.url}. Error: {e}"
            logger.error(error_msg)
            return (f"Error: Could not connect to Ollama service. "
                    f"Ensure it's running and URL is correct ({OLLAMA_API_BASE_URL}).")

        except json.JSONDecodeError:
            # Handle JSON parsing errors
            logger.error(
                f"Ollama API Error: Failed to decode JSON response from Ollama. "
                f"Status: {response.status_code}. Response text: {response.text[:500]}",
                exc_info=True)
            return "Error: Ollama returned an invalid response (not JSON)."

        except Exception as e:
            # Handle any other unexpected errors
            error_msg = f"Unexpected error during Ollama query for model '{model}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"Error: Unexpected issue during Ollama query - {type(e).__name__}"


# --- Mapping Logic Functions ---
def create_mapping_data() -> List[MappingItem]:
    """Defines the structure for extracting specific data points from documents."""

    # --- ADD THIS LINE ---
    global _symptom_list_processor, _hpo_symptom_processor
    # --- END ADD ---

    # Check if the MappingItem class was successfully imported or defined
    if 'MappingItem' not in globals() or not callable(MappingItem):
        logger.error("MappingItem class is not defined.")
        return []

    # Define symptom processor functions if not available
    if '_symptom_list_processor' not in globals():
        logger.error("Custom processor _symptom_list_processor not found.")

        async def _symptom_list_processor(raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
            return {}

    if '_hpo_symptom_processor' not in globals():
        logger.error("Custom processor _hpo_symptom_processor not found.")

        async def _hpo_symptom_processor(raw_answer: Optional[str], item: MappingItem) -> Dict[str, str]:
            return {}

    mapping_data: List[MappingItem] = []
    logger.info("Defining mapping data structure...")

    # Common fields
    mapping_data.append(MappingItem(
        field="pmid",
        question="What is the PubMed ID of the article?",
        mapped_excel_column="PMID",
        response_convertion_strategy="Extract the PubMed ID (PMID) as a numeric identifier "
                                     "(e.g., 28847615). If not found or mentioned, return -99."
    ))

    mapping_data.append(MappingItem(
        field="author, year",
        question="Who is the first author and what is the publication year?",
        mapped_excel_column="Author_year",
        response_convertion_strategy="Extract the last name of the first author and the four-digit publication year, "
                                     "formatted as 'LastName, Matisse' (e.g., 'Smith, 2018'). If not found, return -99."
    ))

    mapping_data.append(MappingItem(
        field="comments_study",
        question="Are there any general comments about the study design, limitations, or overall findings mentioned?",
        mapped_excel_column="Study Comments",
        response_convertion_strategy="Summarize any general comments about the study itself in free text "
                                     "(max 200 characters). If no relevant comments are found, return -99."
    ))

    # Patient fields
    mapping_data.append(MappingItem(
        field="family_id",
        question="What is the family ID or identifier reported for this patient?",
        mapped_excel_column="Family ID",
        response_convertion_strategy="Extract the family identifier exactly as reported "
                                     "(e.g., 'Family 1', 'F002', 'Pedigree 3'). If it's explicitly stated as a "
                                     "sporadic case or no family ID is mentioned, return -99."
    ))

    mapping_data.append(MappingItem(
        field="individual_id",
        question="What is the individual ID or identifier reported for this patient within their family "
                 "or study (e.g., 'II-1', 'Patient 3', 'Proband', 'Subject A')?",
        mapped_excel_column="Individual ID",
        response_convertion_strategy="Extract the individual's identifier as reported. If not provided or "
                                     "only 'proband' is used without a specific ID, return -99."
    ))

    mapping_data.append(MappingItem(
        field="sex",
        question="What is the reported sex of the individual (Male/Female)?",
        mapped_excel_column="Sex",
        response_convertion_strategy="Return 'M' for male or 'F' for female based on the text. "
                                     "If not explicitly reported, return -99."
    ))

    mapping_data.append(MappingItem(
        field="aao",
        question="What was the age at onset (AAO) of symptoms reported, specified in years?",
        mapped_excel_column="AAO",
        response_convertion_strategy="Extract the age at onset in years as a number (integer). "
                                     "If reported as a range, take the lower bound. If reported in months/days, "
                                     "convert to years (approximate if necessary). "
                                     "If not reported or unknown, return -99."
    ))

    mapping_data.append(MappingItem(
        field="age",
        question="What is the current age or age at last examination/report/death reported, specified in years?",
        mapped_excel_column="Age",
        response_convertion_strategy="Extract the age at last clinical evaluation, age at report, "
                                     "or age at death in years as a number (integer). If a range, use the value "
                                     "corresponding to the last examination. If not reported, return -99."
    ))

    mapping_data.append(MappingItem(
        field="consanguinity",
        question="Does the text mention if the parents of the patient were consanguineous (related)?",
        mapped_excel_column="Consanguinity",
        response_convertion_strategy="Return 'yes' if consanguinity (e.g., 'related parents', "
                                     "'consanguineous marriage') is mentioned for the patient's parents. "
                                     "Return 'no' if explicitly stated as non-consanguineous. "
                                     "If not mentioned either way, return -99."
    ))

    # Genetic information
    mapping_data.append(MappingItem(
        field="gene1",
        question="What is the primary gene symbol associated with the condition or mutation found in this patient?",
        mapped_excel_column="Gene1",
        response_convertion_strategy="Extract the official HGNC gene symbol (e.g., 'PARK2', 'SNCA', 'LRRK2'). "
                                     "Ensure correct capitalization. If multiple genes are mentioned, identify the "
                                     "primary one related to the patient's diagnosis or reported mutation. "
                                     "If not found, return -99."
    ))

    mapping_data.append(MappingItem(
        field="mut1_c",
        question="What is the specific mutation identified in gene1 at the DNA level, using HGVS cDNA "
                 "notation (e.g., c.511C>T, c.123_125del)?",
        mapped_excel_column="Mut1 cDNA",
        response_convertion_strategy="Extract the primary mutation using HGVS coding DNA (cDNA) nomenclature "
                                     "(e.g., c.511C>T, c.123delG, c.88+1G>A). Include the 'c.' prefix. "
                                     "If only protein level is given, try to infer if possible, otherwise return -99. "
                                     "If not found, return -99."
    ))

    mapping_data.append(MappingItem(
        field="mut1_p",
        question="What is the effect of the primary mutation (mut1_c) at the protein level, using HGVS protein notation "
                 "(e.g., p.Gln171*, p.Arg412His)?",
        mapped_excel_column="Mut1 Protein",
        response_convertion_strategy="Extract the protein change using HGVS protein nomenclature "
                                     "(e.g., p.Gln171*, p.Arg412His, p.Val7fs). Include the 'p.' prefix. "
                                     "If only cDNA level is given, try to infer if possible, otherwise return -99. "
                                     "If not found, return -99."
    ))

    mapping_data.append(MappingItem(
        field="mut1_zygosity",
        question="What is the zygosity reported for the primary mutation (mut1_c/mut1_p) in this patient "
                 "(heterozygous, homozygous, hemizygous, compound heterozygous)?",
        mapped_excel_column="Mut1 Zygosity",
        response_convertion_strategy="Return 'heterozygous', 'homozygous', 'hemizygous', or 'compound heterozygous' "
                                     "based on the text description for the primary mutation. "
                                     "If not explicitly reported, return -99."
    ))

    # Custom processors
    mapping_data.append(MappingItem(
        field="symptoms_list",
        question="List the key clinical symptoms or phenotypes reported for the patient. For each symptom mentioned, "
                 "indicate if it was present (Yes) or explicitly absent (No). Format each as 'Symptom Name: Yes/No'. "
                 "Examples: 'Tremor: Yes', 'Rigidity: No', 'Cognitive Impairment: Yes'. "
                 "If no specific symptoms are listed or assessed this way, state 'None'.",
        mapped_excel_column="Dynamic_Symptoms",
        response_convertion_strategy="Parse the list provided by the LLM...",
        custom_processor=_symptom_list_processor
    ))

    mapping_data.append(MappingItem(
        field="hpo_symptoms",
        question="List any Human Phenotype Ontology (HPO) terms and their status (present/absent) reported for the "
                 "patient. Format as 'HPO Term Name (HP:XXXXXXX): Yes/No'. Example: 'Seizures (HP:0001250): Yes', "
                 "'Ataxia (HP:0001251): No'. If no HPO terms are listed, state 'None'.",
        mapped_excel_column="Dynamic_HPO",
        response_convertion_strategy="Parse the list provided by the LLM...",
        custom_processor=_hpo_symptom_processor
    ))

    logger.info(f"Defined {len(mapping_data)} mapping items.")
    return mapping_data


async def format_answer(raw_answer: Optional[str], strategy: str) -> str:
    """Uses a Formatter LLM to clean and format a raw answer string."""
    # Pre-checks for empty or error messages
    if not raw_answer or raw_answer.strip().lower() in ["", "none", "n/a", "null", "-", "not applicable"]:
        logger.debug("Format pre-check: Raw empty/None/NA. Returning -99.")
        return "-99"

    if raw_answer.startswith("Error:") or any(e in raw_answer for e in
                                              [" query failed", " request failed", " returned no answer",
                                               "service unavailable"]):
        logger.warning(f"Format pre-check: Raw indicates upstream error ('{raw_answer[:60]}...'). Returning -99.")
        return "-99"

    # Check for common "unknown" responses
    unknown_phrases = [
        "don't know", "couldn't find", "not stated", "not mentioned", "not reported",
        "unknown", "no information", "not available", "not specified", "unable to determine",
        "information not found", "the text does not specify", "the document does not mention",
        "no mention of", "no data available"
    ]

    raw_answer_lower = raw_answer.lower()
    if any(phrase in raw_answer_lower for phrase in unknown_phrases) and len(raw_answer) < 100:
        simplified_answer = re.sub(r'\W+', ' ', raw_answer_lower).strip()
        is_just_unknown = False

        for phrase in unknown_phrases:
            if simplified_answer == re.sub(r'\W+', ' ', phrase).strip():
                is_just_unknown = True
                break

        if is_just_unknown:
            logger.info(f"Format pre-check: Raw indicates info not found ('{raw_answer[:60]}...'). Returning -99.")
            return "-99"
        else:
            logger.debug("Format pre-check: Raw contains 'unknown' phrase but has other content. Proceeding.")

    # Build prompt for the formatter LLM
    prompt = f"""Please analyze the following raw text and format it precisely according to the strategy provided. 
    Output ONLY the final formatted value requested by the strategy, with no extra explanations, apologies, 
    or introductory phrases like "The formatted value is:".

Raw Text:
\"\"\"
{raw_answer}
\"\"\"

Formatting Strategy:
\"\"\"
{strategy}
\"\"\"

Strict Output Rules:
1. **Adhere strictly to the strategy.**
2. **Handle Missing Information:** If the raw text clearly indicates the information is missing, not stated, unknown, N/A, or not applicable FOR THE SPECIFIC ITEM REQUESTED, return the exact value "-99".
3. **No Explanations:** Just the final formatted value or "-99".
4. **Specific Formats:**
    * Numeric: Integer only. Unknown=-99. Zero=0.
    * Yes/No: Lowercase "yes" or "no". Unknown=-99.
    * Sex: Uppercase "M" or "F". Unknown=-99.
    * Zygosity: "heterozygous", "homozygous", "hemizygous", or "compound heterozygous". Unknown=-99.
    * Gene/Mutation Notation: Exact notation (e.g., 'SNCA', 'c.153G>A'). Not found=-99.
    * Free Text: Relevant text, trimmed. No relevant text=-99.
5. **Focus:** Base the answer ONLY on the provided Raw Text and the Formatting Strategy.

Final Formatted Value:"""

    logger.debug(
        f"Sending to formatter LLM ({FORMATTER_MODEL_NAME}). Strategy: {strategy[:60]}... Raw Text: '{raw_answer[:80]}...'")
    formatted = await ollama_query(prompt=prompt, model=FORMATTER_MODEL_NAME)
    logger.debug(f"Raw Formatter LLM Response: '{formatted}'")

    if formatted.startswith("Error:"):
        logger.error(f"Formatter LLM query failed. Raw: '{raw_answer[:60]}...'. Error: {formatted}")
        return "FORMATTING_LLM_ERROR"

    # Post-process the formatted response
    if formatted.startswith('"') and formatted.endswith('"') and len(formatted) > 1:
        formatted = formatted[1:-1]
    elif formatted.startswith("'") and formatted.endswith("'") and len(formatted) > 1:
        formatted = formatted[1:-1]

    formatted = formatted.replace('`', '')
    formatted = re.sub(
        r"^(final formatted value|formatted value|formatted answer|final answer|result|output|answer)\s*[:\-]?\s*", "",
        formatted, flags=re.IGNORECASE | re.MULTILINE).strip()

    if not formatted:
        logger.warning(f"Formatter returned empty response for raw: '{raw_answer[:60]}...'. Returning -99.")
        return "-99"

    # Check for known "unknown" LLM responses
    unknown_llm_responses = [
        "unknown", "not stated", "not reported", "n/a", "none", "not applicable",
        "not mentioned", "-", "not found", "no data", "information not available",
        "cannot answer", "nan", "null", "-99.", "not determinable", "missing",
        "no relevant information found", "no specific information", "value not found"
    ]

    if formatted.lower() in unknown_llm_responses:
        logger.info(f"Formatter returned '{formatted}'. Interpreting as unknown (-99).")
        return "-99"

    # Handle multi-line responses
    lines = formatted.split('\n')
    if len(lines) > 1:
        first_line = lines[0].strip()
        if len(first_line) < 80 and first_line.lower() not in ["here is the formatted value:",
                                                               "the formatted value is:"]:
            logger.warning(
                f"Formatter returned multiple lines. Using only first: '{first_line}'. Full: '{formatted[:150]}...'")
            formatted = first_line

    logger.info(f"Final Formatted Answer after post-processing: '{formatted}'")
    return formatted


async def _generate_questions_from_patient_list(json_array_text: str, mapping_items: List[MappingItem]) -> List[
    List[QuestionInfo]]:
    """Parses patient JSON and generates questions."""
    if 'QuestionInfo' not in globals() or not callable(QuestionInfo):
        logger.error("QuestionInfo class not defined.")
        return []

    all_patient_question_sets: List[List[QuestionInfo]] = []
    processed_patients = set()

    try:
        patients = json.loads(json_array_text)
        if not isinstance(patients, list):
            raise ValueError("Input JSON is not a list.")

        logger.info(f"Generating questions for {len(patients)} potential patient entries.")

        for i, entry in enumerate(patients):
            if not isinstance(entry, dict):
                logger.warning(f"Skipping invalid patient entry {i}: {type(entry)}.")
                continue

            raw_family_id = entry.get("family")
            raw_patient_id = entry.get("patient")

            family_id_str = str(raw_family_id).strip() if raw_family_id is not None and str(
                raw_family_id).strip().lower() not in ['none', 'null', 'n/a', ''] else None
            patient_id_str = str(raw_patient_id).strip() if raw_patient_id is not None and str(
                raw_patient_id).strip().lower() not in ['none', 'null', 'n/a', ''] else None

            if not patient_id_str:
                logger.warning(f"Skipping entry {i}: missing 'patient' ID.")
                continue

            patient_key = f"FAM:{family_id_str or 'NONE'}_PAT:{patient_id_str}"

            if patient_key in processed_patients:
                logger.info(f"Skipping duplicate patient: {patient_key}")
                continue

            processed_patients.add(patient_key)
            one_patient_questions: List[QuestionInfo] = []

            log_prefix = f"Patient ID:'{patient_id_str}'" + (
                f" (Family ID:'{family_id_str}')" if family_id_str else " (No Family ID)")
            logger.info(f"  Generating questions for {log_prefix}")

            for item in mapping_items:
                if item.custom_processor:
                    logger.debug(f"  Skipping field '{item.field}' (custom processor).")
                    continue

                context_phrase = f"Regarding patient '{patient_id_str}'" + (
                    f" from family '{family_id_str}'" if family_id_str else "")
                final_query = f"{context_phrase}: {item.question}"

                q_info = QuestionInfo(
                    field=item.field,
                    query=final_query,
                    response_convertion_strategy=item.response_convertion_strategy,
                    family_id=family_id_str,
                    patient_id=patient_id_str
                )

                one_patient_questions.append(q_info)
                logger.debug(f"    Generated question for '{item.field}': {final_query[:100]}...")

            if one_patient_questions:
                all_patient_question_sets.append(one_patient_questions)
            else:
                logger.warning(f"No standard questions generated for {log_prefix}.")

        logger.info(f"Generated question sets for {len(all_patient_question_sets)} unique patients.")
        return all_patient_question_sets

    except json.JSONDecodeError as e:
        logger.error(f"Failed JSON parse: {e}. JSON: '{json_array_text[:500]}...'")
        raise ValueError(f"Invalid JSON format: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error generating questions: {e}", exc_info=True)
        raise RuntimeError(f"Failed to generate questions: {e}") from e


# --- API Endpoints ---

# --- Upload Document Endpoint ---
@app.post("/upload_document", status_code=status.HTTP_201_CREATED,
          summary="Upload and Process a Document",
          description="Uploads doc, extracts text, saves, adds to backup & PaperQA.")
async def upload_document(file: UploadFile = File(...), description: Optional[str] = Form(None)):
    start_time = datetime.now()
    original_filename = file.filename or f"upload_{start_time.strftime('%Y%m%d_%H%M%S')}"
    safe_filename = re.sub(r'[^\w\.\-]', '_', Path(original_filename).name)
    logger.info(f"Received upload: '{original_filename}' -> '{safe_filename}', Type: {file.content_type}")

    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(safe_filename).suffix.lower()
    if file_ext not in allowed_extensions:
        logger.error(f"Upload rejected: Unsupported type '{file_ext}'.")
        raise HTTPException(status_code=400,
                            detail=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(allowed_extensions)}")

    # Setup file paths
    try:
        current_persistent_data_dir = Path(os.getenv("PERSISTENT_DATA_DIR", "./persistent_data"))
        current_uploaded_files_dir = current_persistent_data_dir / "uploaded_files"
        current_uploaded_files_dir.mkdir(parents=True, exist_ok=True)
        persistent_file_path = current_uploaded_files_dir / safe_filename
    except Exception as path_err:
        logger.error(f"Failed path setup: {path_err}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server config error: storage directory.")

    # Save uploaded file
    try:
        logger.info(f"Saving to: {persistent_file_path}")
        with open(persistent_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File '{safe_filename}' saved.")
    except Exception as e:
        logger.error(f"Failed save: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        await file.close()

    # Extract text from the file
    logger.info(f"Extracting text from '{safe_filename}'...")
    extracted_text = await extract_text_from_file(str(persistent_file_path))

    if extracted_text.startswith("[Error:"):
        logger.error(f"Extraction failed: {extracted_text}")
        try:
            persistent_file_path.unlink(missing_ok=True)
            logger.info("Cleaned up file.")
        except OSError as e:
            logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=400, detail=f"Text extraction failed: {extracted_text}")

    logger.info(f"Text extracted. Length: {len(extracted_text)} chars.")

    # Create document and store
    doc_id = safe_filename
    doc_desc = description or doc_id
    new_doc = SimpleDocument(doc_id=doc_id, description=doc_desc, text=extracted_text,
                             filepath=str(persistent_file_path))

    # Check for existing document with same ID
    existing_doc_index = -1
    for i, doc in enumerate(documents_storage):
        if doc.doc_id == doc_id:
            existing_doc_index = i
            break

    action = ""
    assigned_id = -1

    if existing_doc_index != -1:
        logger.warning(f"Overwriting existing doc '{doc_id}' at index {existing_doc_index}.")
        old_doc = documents_storage[existing_doc_index]

        # Remove from PaperQA if exists
        if paperqa_available and docs_instance and old_doc.paperqa_dockey:
            logger.info(f"Removing old PQA entry (key: {old_doc.paperqa_dockey})...")
            delete_method_name = "delete"

            if hasattr(docs_instance, delete_method_name):
                try:
                    delete_method = getattr(docs_instance, delete_method_name)
                    await (delete_method(old_doc.paperqa_dockey) if asyncio.iscoroutinefunction(delete_method)
                           else asyncio.to_thread(delete_method, old_doc.paperqa_dockey))
                    logger.info("Removed old PQA key.")
                except Exception as e:
                    logger.error(f"Failed old PQA key removal: {e}")
            else:
                logger.warning(f"PQA lacks '{delete_method_name}'. Cannot remove old key.")

        documents_storage[existing_doc_index] = new_doc
        assigned_id = existing_doc_index
        action = "overwritten"
    else:
        documents_storage.append(new_doc)
        assigned_id = len(documents_storage) - 1
        action = "added"

    # Add to PaperQA
    logger.info(f"Attempting PQA add for '{doc_id}'...")
    paperqa_dockey = await add_document_to_paperqa(new_doc)
    paperqa_status = "Not attempted (PQA unavailable)"

    if paperqa_available:
        paperqa_status = f"Added/Found in PaperQA (Key: {paperqa_dockey})" if paperqa_dockey else "Failed to add/verify in PaperQA"

    logger.info(f"PQA status: {paperqa_status}")
    save_documents_storage()

    duration = datetime.now() - start_time
    logger.info(
        f"Doc '{doc_id}' {action} (ID: {assigned_id}). Text: {len(extracted_text)}. PQA: {paperqa_status}. Time: {duration.total_seconds():.2f}s")

    return {
        "message": f"Doc '{doc_id}' uploaded.",
        "assigned_doc_id": assigned_id,
        "filename": doc_id,
        "description": doc_desc,
        "text_length": len(extracted_text),
        "paperqa_status": paperqa_status,
        "paperqa_dockey": paperqa_dockey
    }


# --- Full Document Mapping Endpoint ---
@app.post("/process_document_mapping/{doc_id}", status_code=status.HTTP_200_OK,
          summary="Process Full Mapping",
          description="Performs full mapping for a document.")
async def process_document_mapping_synchronous_with_cancel(doc_id: int):
    global is_processing_globally, cancel_requested_globally, documents_storage
    start_time = datetime.now()

    if is_processing_globally:
        raise HTTPException(status_code=409, detail="Another process running.")

    if not (0 <= doc_id < len(documents_storage)):
        raise HTTPException(status_code=404, detail="Doc not found.")

    is_processing_globally = True
    cancel_requested_globally = False
    target_document = documents_storage[doc_id]
    doc_name = target_document.doc_id
    logger.info(f"Starting mapping doc_id {doc_id} ('{doc_name}'). Lock SET.")

    all_patient_data_rows = []
    final_header_order = []

    try:
        # Validate that we can proceed
        check_cancel_request()

        # Load mapping configurations
        all_mapping_items = create_mapping_data()
        mapping_item_dict = {item.field: item for item in all_mapping_items}
        logger.info(f"Loaded {len(all_mapping_items)} mappings.")

        # Check if cancelled
        check_cancel_request()

        # Extract patient identifiers from document
        logger.info(f"Identifying patients in '{doc_name}'...")
        extract_prompt = f"From '{doc_name}' only, list distinct patient identifiers. Format: 'FamilyID: PatientID' or just 'PatientID'. If none, output 'None'."
        raw_ids_answer = await query_paperqa(extract_prompt, target_doc_name=doc_name)

        # Check if cancelled
        check_cancel_request()

        list_of_patient_question_sets = []
        patient_identifiers_found = []

        if (
            raw_ids_answer.startswith("Error:")
            or not raw_ids_answer
            or raw_ids_answer.strip().lower()
            in ['none', 'no patients found']
        ):
            logger.warning(f"No patients extracted. PQA Resp: '{raw_ids_answer[:100]}...'")
        else:
            logger.info(f"Raw IDs: '{raw_ids_answer[:200]}...'")

            # Convert raw extraction to structured JSON
            json_prompt = (f"Convert to JSON array [{{'family': ID_or_null, 'patient': ID}}]:"
                           f"\n\"\"\"\n{raw_ids_answer}\n\"\"\"\n\nJSON Array Output ONLY:")
            json_resp = await ollama_query(prompt=json_prompt, model=FORMATTER_MODEL_NAME)

            # Check if cancelled
            check_cancel_request()

            # Extract JSON from response
            json_text = ""
            match = re.search(r'(\[.*?\])', json_resp, re.DOTALL | re.MULTILINE)

            if match:
                json_text = match.group(1)
            elif json_resp.strip().startswith('[') and json_resp.strip().endswith(']'):
                json_text = json_resp.strip()

            if json_text:
                try:
                    list_of_patient_question_sets = await _generate_questions_from_patient_list(
                        json_text, all_mapping_items
                    )
                except (ValueError, RuntimeError, json.JSONDecodeError) as e:
                    logger.error(f"Patient gen failed: {e}. JSON: '{json_text[:200]}...'")
                    list_of_patient_question_sets = []
                    patient_identifiers_found = []
            else:
                logger.warning(f"No JSON array from formatter: '{json_resp[:200]}...'")

        logger.info(f"Patient ID complete. Found {len(list_of_patient_question_sets)} sets.")

        # Process common fields first (document-level)
        common_data_results = {}
        common_fields_to_fetch = [f for f in COMMON_FIELDS if f in mapping_item_dict]
        logger.info(f"Fetching common fields: {common_fields_to_fetch}")

        for field_name in common_fields_to_fetch:
            check_cancel_request()
            item = mapping_item_dict[field_name]
            logger.debug(f"  Querying common: '{field_name}'...")
            raw_answer = await query_paperqa(item.question, target_doc_name=doc_name)
            formatted_value = await format_answer(raw_answer, item.response_convertion_strategy)
            common_data_results[field_name] = formatted_value
            logger.debug(f"  Result '{field_name}': '{formatted_value}'")

        logger.info("Common data fetch complete.")

        # Prepare header order
        standard_fields = [item.field for item in all_mapping_items if not item.custom_processor]
        dynamic_field_placeholders = [item.field for item in all_mapping_items if item.custom_processor]
        final_header_order = list(COMMON_FIELDS) + [f for f in standard_fields if f not in COMMON_FIELDS]
        processed_dynamic_headers = set()

        # Handle case with no patients - create a single row with common fields
        if not list_of_patient_question_sets:
            logger.warning(f"No patient sets for '{doc_name}'. Single row output.")
            single_row = common_data_results.copy()
            single_row.update({f: "-99" for f in standard_fields if f not in single_row})
            single_row["family_id"] = "-99"
            single_row["individual_id"] = "-99"
            all_patient_data_rows.append(single_row)
        else:
            # Process each patient
            logger.info(f"Processing {len(list_of_patient_question_sets)} patient sets...")

            for p_num, p_q_set in enumerate(list_of_patient_question_sets, 1):
                check_cancel_request()

                patient_id = p_q_set[0].patient_id if p_q_set else "Unk"
                family_id = p_q_set[0].family_id if p_q_set else None
                log_p_ctx = (f"Patient {p_num}/{len(list_of_patient_question_sets)} "
                             f"(ID: {patient_id}, Fam: {family_id or 'N/A'})")
                logger.info(f"--- Processing {log_p_ctx} ---")

                # Start with common data
                patient_row_data = common_data_results.copy()

                # Process standard fields
                for q_info in p_q_set:
                    check_cancel_request()
                    field_name = q_info.field
                    item = mapping_item_dict[field_name]

                    # Skip common fields and custom processors (handled separately)
                    if field_name in COMMON_FIELDS or item.custom_processor:
                        continue

                    logger.debug(f"  Querying {log_p_ctx}, Field: '{field_name}'...")
                    raw_answer = await query_paperqa(q_info.query, target_doc_name=doc_name)
                    formatted_value = await format_answer(raw_answer, q_info.response_convertion_strategy)
                    patient_row_data[field_name] = formatted_value
                    logger.debug(f"  Result '{field_name}': '{formatted_value}'")

                # Process fields with custom processors
                for field_name in dynamic_field_placeholders:
                    check_cancel_request()
                    item = mapping_item_dict[field_name]

                    if not item.custom_processor:
                        continue

                    logger.debug(f"  Processing dynamic '{field_name}' for {log_p_ctx}...")
                    patient_ctx = f"Regarding patient '{patient_id}'" + (
                        f" from family '{family_id}'" if family_id else "")
                    dynamic_query = f"{patient_ctx}: {item.question}"
                    raw_dynamic_answer = await query_paperqa(dynamic_query, target_doc_name=doc_name)

                    try:
                        dynamic_results: Dict[str, str] = await item.custom_processor(raw_dynamic_answer, item)
                        logger.debug(f"  Processor '{field_name}' returned {len(dynamic_results)} cols.")
                    except Exception as cp_err:
                        logger.error(f"Custom processor error '{field_name}' on {log_p_ctx}: "
                                     f"{cp_err}", exc_info=True)
                        error_col_name = f"{field_name}_proc_error"
                        patient_row_data[error_col_name] = str(cp_err)

                        if error_col_name not in processed_dynamic_headers:
                            final_header_order.append(error_col_name)
                            processed_dynamic_headers.add(error_col_name)
                        continue

                    # Add dynamic columns to header and row
                    for dynamic_col, dynamic_val in dynamic_results.items():
                        patient_row_data[dynamic_col] = dynamic_val

                        if dynamic_col not in processed_dynamic_headers:
                            final_header_order.append(dynamic_col)
                            processed_dynamic_headers.add(dynamic_col)
                            logger.info(f"Added dynamic header: '{dynamic_col}'")

                all_patient_data_rows.append(patient_row_data)
                logger.info(f"--- Finished {log_p_ctx} ---")

        # Final output preparation
        logger.info("Preparing final output...")
        final_output_rows = []

        for row_dict in all_patient_data_rows:
            output_row = {h: str(row_dict.get(h, "-99")) for h in final_header_order}
            final_output_rows.append(output_row)

        duration = datetime.now() - start_time
        logger.info(f"Mapping doc_id {doc_id} completed in {duration.total_seconds():.2f}s.")

        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "status": "completed",
            "processing_duration_seconds": round(duration.total_seconds(), 2),
            "patients_identified_count": len(patient_identifiers_found),
            "patients_processed_count": len(list_of_patient_question_sets) if list_of_patient_question_sets else 1,
            "patient_identifiers": [p.dict() for p in patient_identifiers_found],
            "total_columns_generated": len(final_header_order),
            "headers": final_header_order,
            "data": final_output_rows
        }

    except CancelRequestException:
        duration = datetime.now() - start_time
        logger.warning(f"Mapping doc_id {doc_id} cancelled after {duration.total_seconds():.2f}s.")
        raise HTTPException(status_code=400, detail="Mapping cancelled by user.")
    except HTTPException:
        raise
    except Exception as e:
        duration = datetime.now() - start_time
        logger.error(f"Mapping failed doc_id {doc_id} after {duration.total_seconds():.2f}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mapping failed: {type(e).__name__}")
    finally:
        is_processing_globally = False
        cancel_requested_globally = False
        logger.info(f"Mapping finished doc_id {doc_id}. Lock UNSET.")


# --- Processing Status Endpoint ---
@app.get("/processing_status", summary="Check Global Processing Status")
async def get_processing_status():
    return {"is_processing": is_processing_globally}


# --- Cancel Mapping Endpoint ---
@app.post("/cancel_current_mapping", status_code=status.HTTP_200_OK, summary="Request Cancellation")
async def request_cancellation():
    global cancel_requested_globally, is_processing_globally

    if not is_processing_globally:
        raise HTTPException(status_code=409, detail="No process running.")

    if cancel_requested_globally:
        logger.info("Cancel already pending.")
        return {"message": "Cancellation already pending."}
    else:
        logger.warning("Received cancel request.")
        cancel_requested_globally = True
        return {"message": "Cancellation requested."}


# --- List Documents Endpoint ---
@app.get("/documents", response_model=List[DocumentInfo], summary="Get List of Documents")
async def get_documents() -> List[DocumentInfo]:
    global documents_storage
    doc_list = []

    for idx, doc_obj in enumerate(documents_storage):
        if isinstance(doc_obj, SimpleDocument):
            doc_list.append(DocumentInfo(
                id=idx,
                filename=doc_obj.doc_id,
                description=doc_obj.description,
                status='uploaded',
                paperqa_dockey=doc_obj.paperqa_dockey
            ))
        else:
            logger.warning(f"Unexpected item type {type(doc_obj)} at index {idx}.")

    return doc_list


# --- Delete Document Endpoint ---
@app.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK, summary="Delete Document")
async def delete_document(doc_id: int, response: Response):
    global documents_storage, paperqa_available, docs_instance

    if not (0 <= doc_id < len(documents_storage)):
        logger.error(f"Delete failed: ID {doc_id} out of range.")
        raise HTTPException(status_code=404, detail=f"Doc ID {doc_id} not found.")

    try:
        doc_to_delete = documents_storage[doc_id]

        if not isinstance(doc_to_delete, SimpleDocument):
            logger.error(f"Inconsistency: Item at {doc_id} not SimpleDocument: {type(doc_to_delete)}")
            raise HTTPException(status_code=500, detail="Internal error: Invalid data type.")

        filename = doc_to_delete.doc_id
        filepath_str = doc_to_delete.filepath
        dockey = doc_to_delete.paperqa_dockey

        logger.info(f"Attempting delete: ID={doc_id}, File='{filename}', Path='{filepath_str}', Key='{dockey}'")

        # Delete file
        if filepath_str:
            try:
                file_path_to_delete = Path(filepath_str)
                file_path_to_delete.unlink(missing_ok=True)
                logger.info(f"Deleted file: {filepath_str}")
            except Exception as e:
                logger.error(f"Error deleting file {filepath_str}: {e}", exc_info=True)
        else:
            logger.warning(f"No filepath for doc ID {doc_id}. Cannot delete file.")

        # Remove from PaperQA if available
        if paperqa_available and docs_instance and dockey:
            logger.info(f"Attempting PQA remove key '{dockey}'...")
            delete_method_name = "delete"

            if hasattr(docs_instance, delete_method_name):
                try:
                    delete_method = getattr(docs_instance, delete_method_name)
                    await (delete_method(dockey) if asyncio.iscoroutinefunction(delete_method)
                           else asyncio.to_thread(delete_method, dockey))
                    logger.info(f"Requested PQA removal key '{dockey}'.")
                except KeyError:
                    logger.warning(f"PQA key '{dockey}' not found during delete.")
                except Exception as e:
                    logger.error(f"Failed PQA remove key '{dockey}': {e}", exc_info=True)
            else:
                logger.warning(f"PQA has no '{delete_method_name}' method. Cannot remove key '{dockey}'.")
        elif dockey:
            logger.warning(f"PQA unavailable, cannot remove key '{dockey}'.")
        else:
            logger.info(f"No PQA key for '{filename}'. Skipping PQA removal.")

        # Remove from storage list
        try:
            deleted_item = documents_storage.pop(doc_id)
            logger.info(f"Removed '{deleted_item.doc_id}' from list at index {doc_id}.")
        except IndexError:
            logger.error(f"IndexError popping {doc_id}. List length: {len(documents_storage)}.")
            raise HTTPException(status_code=500, detail="Internal error: List changed during delete.")

        # Save updated storage
        logger.info("Saving backup after deletion...")
        save_documents_storage()

        logger.info(f"Doc ID {doc_id} ('{filename}') deleted.")
        return {"message": f"Doc ID {doc_id} ('{filename}') deleted."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting doc ID {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Unexpected error deleting doc ID {doc_id}.")


# --- Reset All Documents Endpoint ---
@app.post("/documents/reset", status_code=status.HTTP_200_OK, summary="Reset All Documents")
async def reset_all_documents():
    global documents_storage, paperqa_available, docs_instance, is_processing_globally
    logger.warning("RESET ALL requested.")

    if is_processing_globally:
        logger.error("Reset denied: Process running.")
        raise HTTPException(status_code=409, detail="Cannot reset while process running.")

    is_processing_globally = True
    logger.warning("RESET LOCK ACQUIRED.")
    deleted_files_count = 0
    deleted_pqa_count = 0
    errors = []

    try:
        # Process documents in reverse order to avoid index shifting issues
        doc_indices_to_process = list(range(len(documents_storage)))
        doc_indices_to_process.reverse()

        for doc_id in doc_indices_to_process:
            try:
                doc_to_delete = documents_storage[doc_id]
                if not isinstance(doc_to_delete, SimpleDocument):
                    continue

                filename = doc_to_delete.doc_id
                filepath_str = doc_to_delete.filepath
                dockey = doc_to_delete.paperqa_dockey
                logger.info(f"Reset: Processing {doc_id}, file {filename}, key {dockey}")

                # Remove from PaperQA if available
                if paperqa_available and docs_instance and dockey:
                    delete_method_name = "delete"

                    if hasattr(docs_instance, delete_method_name):
                        try:
                            delete_method = getattr(docs_instance, delete_method_name)
                            await (delete_method(dockey) if asyncio.iscoroutinefunction(delete_method)
                                   else asyncio.to_thread(delete_method, dockey))
                            deleted_pqa_count += 1
                            logger.debug(f"Reset: Deleted key '{dockey}' from PQA.")
                        except Exception as e:
                            logger.error(f"Reset: Failed PQA delete key '{dockey}': {e}")
                            errors.append(f"PQA delete failed for {filename} (key: {dockey})")
                    else:
                        logger.warning(f"Reset: PQA has no '{delete_method_name}' method.")

                # Delete file
                if filepath_str:
                    try:
                        file_path = Path(filepath_str)
                        file_path.unlink(missing_ok=True)
                        deleted_files_count += 1
                        logger.debug(f"Reset: Deleted file {filepath_str}")
                    except Exception as e:
                        logger.error(f"Reset: Failed file delete {filepath_str}: {e}")
                        errors.append(f"File delete failed for {filename}")

                documents_storage.pop(doc_id)
            except Exception as e:
                logger.error(f"Reset: Error processing doc_id {doc_id}: {e}")
                errors.append(f"General error for doc_id {doc_id}")

        # Clear any remaining documents
        documents_storage.clear()
        logger.info("Reset: Cleared in-memory list.")

        # Save empty backup
        try:
            save_documents_storage()
            logger.info("Reset: Saved empty backup.")
        except Exception as e:
            logger.error(f"Reset: Failed save empty backup: {e}")
            errors.append("Failed save empty backup")

        # Optional Re-init PQA
        if paperqa_available and docs_instance and hasattr(docs_instance, 'clear'):
            try:
                docs_instance.clear()
                logger.info("Reset: Called docs_instance.clear()")
            except Exception as e:
                logger.error(f"Reset: Error calling docs_instance.clear(): {e}")
                errors.append("Failed docs_instance.clear()")
        elif paperqa_available and docs_instance:
            # Try deleting index dir if clear not available
            pqa_home_str = os.environ.get("PQA_HOME")

            if pqa_home_str:
                index_dir = Path(pqa_home_str) / "indexes"

                if index_dir.exists():
                    try:
                        shutil.rmtree(index_dir)
                        logger.info(f"Reset: Deleted PQA index dir {index_dir}")
                        init_paperqa()
                    except Exception as e:
                        logger.error(f"Reset: Failed delete PQA index dir {index_dir}: {e}")
                        errors.append("Failed delete PQA index dir")

        final_message = (f"Reset complete. Files deleted: {deleted_files_count}. "
                         f"PQA entries processed: {deleted_pqa_count}.")

        if errors:
            final_message += f" Errors: {len(errors)}. Check logs."
            logger.error(f"Reset errors: {errors}")

        logger.warning(f"RESET COMPLETE. Files: {deleted_files_count}. PQA: {deleted_pqa_count}. Errors: {len(errors)}")

        return {
            "message": "Reset complete.",
            "files_deleted": deleted_files_count,
            "paperqa_entries_processed": deleted_pqa_count,
            "errors": errors
        }
    except Exception as e:
        logger.error(f"Critical reset error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Critical reset error: {e}")
    finally:
        is_processing_globally = False
        logger.warning("RESET LOCK RELEASED.")


# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Doc Mapping Server running.",
        "paperqa_available": paperqa_available,
        "paperqa_model": PAPERQA_MODEL_NAME if paperqa_available else "N/A",
        "formatter_model": FORMATTER_MODEL_NAME,
        "total_docs": len(documents_storage),
        "pqa_docs": len(docs_instance.docs) if paperqa_available and docs_instance else 0,
        "docs_endpoint": "/docs"
    }


# --- Mapping Items Endpoint ---
@app.get("/mapping_items", response_model=List[MappingItemInfo], summary="Get Available Mapping Items")
async def get_mapping_items():
    try:
        mapping_items = create_mapping_data()
        return [
            MappingItemInfo(
                field=item.field,
                question=item.question,
                column=getattr(item, 'mapped_excel_column', None)
            ) for item in mapping_items
        ]
    except Exception as e:
        logger.error(f"Failed mapping data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed load mapping items")


# --- Process Single Field Endpoint ---
@app.post("/documents/{doc_id}/process_field", response_model=ProcessFieldResponse, summary="Process Single Field")
async def process_single_field(doc_id: int, request: ProcessFieldRequest):
    global is_processing_globally, cancel_requested_globally, documents_storage

    if is_processing_globally:
        raise HTTPException(status_code=409, detail="Process running.")

    if not (0 <= doc_id < len(documents_storage)):
        raise HTTPException(status_code=404, detail="Doc not found")

    target_document = documents_storage[doc_id]
    doc_name = target_document.doc_id
    field_to_process = request.field

    try:
        all_mapping_items = create_mapping_data()
        target_item = next((item for item in all_mapping_items if item.field == field_to_process), None)
    except Exception as e:
        logger.error(f"Error finding mapping item '{field_to_process}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Mapping config error.")

    if target_item is None:
        raise HTTPException(status_code=404, detail=f"Mapping field '{field_to_process}' not found.")

    if hasattr(target_item, 'custom_processor') and target_item.custom_processor:
        raise HTTPException(status_code=400, detail=f"Field '{field_to_process}' uses custom processor.")

    is_processing_globally = True
    cancel_requested_globally = False
    logger.info(f"Starting single field: doc {doc_id}, field '{field_to_process}'. Lock SET.")

    raw_answer_text = None
    formatted_value = "-99"

    try:
        question = f"Regarding document '{doc_name}': {target_item.question}"
        logger.info(f"Querying field '{field_to_process}': {question}")

        check_cancel_request()
        raw_answer = await query_paperqa(question, target_doc_name=doc_name)
        raw_answer_text = str(raw_answer)

        check_cancel_request()
        logger.info(f"Formatting field '{field_to_process}'...")
        formatted_value = await format_answer(raw_answer_text, target_item.response_convertion_strategy)

        logger.info(f"Single field complete: doc {doc_id}, field '{field_to_process}'. Value: '{formatted_value}'")

        return ProcessFieldResponse(
            field=field_to_process,
            value=formatted_value,
            raw_answer=raw_answer_text
        )
    except CancelRequestException:
        logger.warning(f"Single field cancelled: doc {doc_id}, field '{field_to_process}'.")
        raise HTTPException(status_code=400, detail="Processing cancelled.")
    except Exception as e:
        logger.error(f"Single field failed: doc {doc_id}, field '{field_to_process}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed field '{field_to_process}': {e}")
    finally:
        is_processing_globally = False
        cancel_requested_globally = False
        logger.info(f"Single field finished: doc {doc_id}, field '{field_to_process}'. Lock UNSET.")


# --- Get Patients Endpoint ---
@app.get("/documents/{doc_id}/patients", response_model=List[PatientInfo], summary="Extract Patients from Document")
async def get_document_patients(doc_id: int):
    global documents_storage

    if not (0 <= doc_id < len(documents_storage)):
        raise HTTPException(status_code=404, detail="Doc not found")

    target_document = documents_storage[doc_id]
    doc_name = target_document.doc_id
    logger.info(f"Extracting patients from doc {doc_id} ('{doc_name}')")

    try:
        # Query PaperQA for patient identifiers
        extract_prompt = (f"From '{doc_name}' only, list distinct patient identifiers. "
                          f"Format: 'FamilyID: PatientID' or just 'PatientID'. If none, output 'None'.")
        logger.info(f"Querying PQA for patient IDs in '{doc_name}'...")
        raw_ids_answer = await query_paperqa(extract_prompt, target_doc_name=doc_name)
        raw_ids = str(raw_ids_answer)

        if raw_ids.startswith("Error:") or not raw_ids or raw_ids.strip().lower() in ['none', 'no patients found']:
            logger.warning(f"Patient ID extraction failed/none. PQA Resp: {raw_ids}")
            return []

        logger.info(f"Raw IDs from PQA:\n---\n{raw_ids}\n---")

        # Clean the response
        raw_ids = re.sub(r'<think>.*?</think>', '', raw_ids, flags=re.DOTALL).strip()

        # Format into structured JSON
        json_prompt = f"Convert to JSON array [{{'family': ID_or_null, 'patient': ID}}]:\n\"\"\"\n{raw_ids}\n\"\"\"\n\nJSON Array Output ONLY:"
        logger.info("Requesting Formatter for patient JSON...")
        json_resp = await ollama_query(prompt=json_prompt, model=FORMATTER_MODEL_NAME)
        logger.info(f"Formatter patient JSON response:\n---\n{json_resp}\n---")

        # Extract JSON from response
        json_text = ""
        match = re.search(r'(\[.*?\])', json_resp, re.DOTALL | re.MULTILINE)

        if match:
            json_text = match.group(1)
        elif json_resp.strip().startswith('[') and json_resp.strip().endswith(']'):
            json_text = json_resp.strip()
        else:
            logger.error(f"No JSON array in Formatter response: {json_resp}")
            raise HTTPException(status_code=500, detail="Failed parse patient list from LLM.")

        try:
            patient_data = json.loads(json_text)
            assert isinstance(patient_data, list)
        except (json.JSONDecodeError, ValueError, AssertionError) as e:
            logger.error(f"Failed decode patient JSON: {e}. JSON: {json_text}")
            raise HTTPException(status_code=500, detail=f"Failed decode patient list JSON: {e}")

        # Process patient data for frontend
        patient_list_for_frontend: List[PatientInfo] = []
        seen_patients = set()

        for entry in patient_data:
            if isinstance(entry, dict) and 'patient' in entry and entry['patient']:
                p_id = str(entry['patient']).strip()
                f_id_raw = entry.get('family')
                f_id = str(f_id_raw).strip() if f_id_raw and str(f_id_raw).strip().lower() not in ['none', 'null',
                                                                                                   'n/a', ''] else None

                patient_key = f"{f_id or 'NOFAMILY'}_{p_id}"
                if patient_key in seen_patients:
                    continue

                seen_patients.add(patient_key)
                display = f"{p_id}" + (f" (Family: {f_id})" if f_id else " (Sporadic)")

                patient_list_for_frontend.append(PatientInfo(
                    family_id=f_id,
                    patient_id=p_id,
                    display_name=display
                ))
            else:
                logger.warning(f"Skipping invalid patient entry: {entry}")

        logger.info(f"Extracted {len(patient_list_for_frontend)} unique patients for doc_id {doc_id}")
        return patient_list_for_frontend

    except Exception as e:
        logger.error(f"Failed extract patients doc_id {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed extract patients: {e}")


# --- Process Patient Field Endpoint ---
@app.post("/documents/{doc_id}/process_patient_field", response_model=ProcessPatientFieldResponse,
          summary="Process Single Patient Field")
async def process_single_patient_field(doc_id: int, request: ProcessPatientFieldRequest):
    global is_processing_globally, cancel_requested_globally, documents_storage

    if is_processing_globally:
        raise HTTPException(status_code=409, detail="Process running.")

    if not (0 <= doc_id < len(documents_storage)):
        raise HTTPException(status_code=404, detail="Doc not found")

    target_document = documents_storage[doc_id]
    doc_name = target_document.doc_id
    field_to_process = request.field
    patient_id = request.patient_id
    family_id = request.family_id

    if not patient_id:
        raise HTTPException(status_code=400, detail="patient_id required.")

    try:
        all_mapping_items = create_mapping_data()
        target_item = next((item for item in all_mapping_items if item.field == field_to_process), None)
    except Exception as e:
        logger.error(f"Error finding mapping item '{field_to_process}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Mapping config error.")

    if target_item is None:
        raise HTTPException(status_code=404, detail=f"Mapping field '{field_to_process}' not found.")

    if hasattr(target_item, 'custom_processor') and target_item.custom_processor:
        raise HTTPException(status_code=400, detail=f"Field '{field_to_process}' uses custom processor.")

    is_processing_globally = True
    cancel_requested_globally = False
    log_context = f"doc {doc_id}, patient '{patient_id}'" + (
        f" (fam '{family_id}')" if family_id else "") + f", field '{field_to_process}'"
    logger.info(f"Starting single field {log_context}. Lock SET.")

    raw_answer_text = None
    formatted_value = "-99"

    try:
        patient_ctx = f"Regarding patient '{patient_id}'" + (f" from family '{family_id}'" if family_id else "")
        question = f"{patient_ctx} within document '{doc_name}': {target_item.question}"
        logger.info(f"Querying {log_context}: {question}")

        check_cancel_request()
        raw_answer = await query_paperqa(question, target_doc_name=doc_name)
        raw_answer_text = str(raw_answer)

        check_cancel_request()
        logger.info(f"Formatting {log_context}...")
        formatted_value = await format_answer(raw_answer_text, target_item.response_convertion_strategy)

        logger.info(f"Single field complete {log_context}. Value: '{formatted_value}'")

        return ProcessPatientFieldResponse(
            field=field_to_process,
            patient_id=patient_id,
            family_id=family_id,
            value=formatted_value,
            raw_answer=raw_answer_text
        )
    except CancelRequestException:
        logger.warning(f"Single field cancelled {log_context}.")
        raise HTTPException(status_code=400, detail="Processing cancelled.")
    except Exception as e:
        logger.error(f"Single field failed {log_context}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed field '{field_to_process}': {e}")
    finally:
        is_processing_globally = False
        cancel_requested_globally = False
        logger.info(f"Single field finished {log_context}. Lock UNSET.")


# --- Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """Loads state and initializes services on startup."""
    try:
        PERSISTENT_DATA_DIR_STARTUP.mkdir(parents=True, exist_ok=True)
        UPLOADED_FILES_DIR_STARTUP.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured persistent dirs exist: {PERSISTENT_DATA_DIR_STARTUP}")
    except OSError as e:
        logger.error(f"CRITICAL: Could not create persistent dirs: {e}. Persistence will fail.")

    # Load documents from backup
    load_documents_storage()

    # Initialize PaperQA
    init_paperqa()

    # Check Ollama connection
    logger.info("Checking Ollama connection...")
    try:
        async with httpx.AsyncClient() as client:
            ping_url = OLLAMA_API_BASE_URL.replace("/api", "/")
            await client.get(ping_url, timeout=10.0)
            logger.info(f"Ollama check OK at {ping_url}")
    except Exception as e:
        logger.error(f"Ollama check failed: {e}.")


@app.on_event("shutdown")
def shutdown_event():
    """Saves state on server shutdown."""
    logger.info("Server shutting down. Saving documents storage...")
    save_documents_storage()
    logger.info("Shutdown complete.")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting Document Mapping Server (Persistent)...")
    uvicorn.run("custom_paperqa_server:app", host="0.0.0.0", port=8000, reload=True)  # reload=True for dev# -*- coding: utf-8 -*-
