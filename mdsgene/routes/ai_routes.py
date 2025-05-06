import os
import json
import random
import shutil
import zipfile
import threading
import pandas as pd
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pathlib import Path
from uuid import uuid4
from fastapi.responses import FileResponse

from ai.agents.publication_details_agent import PublicationDetailsAgent
from ai.agents.patient_identifiers_agent import PatientIdentifiersAgent
from ai.agents.questions_processing_agent import QuestionsProcessingAgent
from ai.agents.base_agent import CACHE_DIR
from fastapi import Body
from typing import Any
from ai.document_processor import DocumentProcessor
from langchain_community.vectorstores import FAISS

from ai.cache_utils import delete_document_and_all_related_data
from ai.vector_store_client import VectorStoreClient

# Path to the PMID cache file
PMID_CACHE_PATH = os.path.join("cache", "pmid_cache.json")

def get_pmid_for_filename(filename: str) -> Optional[str]:
    """
    Get the PMID for a given filename by looking it up in the pmid_cache.json file.

    Args:
        filename: The name of the PDF file

    Returns:
        The PMID if found, None otherwise
    """
    try:
        if os.path.exists(PMID_CACHE_PATH):
            with open(PMID_CACHE_PATH, "r", encoding="utf-8") as f:
                pmid_cache = json.load(f)

            if filename in pmid_cache and "pmid" in pmid_cache[filename]:
                return pmid_cache[filename]["pmid"]
    except Exception as e:
        print(f"Error getting PMID for filename {filename}: {str(e)}")

    return None

# Получение путей из переменных окружения с fallback по умолчанию
LOCAL_DB_PATH = os.getenv("LOCAL_DB_PATH", r".\vector_store\faiss_index")
VECTOR_STORE_DIR = LOCAL_DB_PATH
TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", r".\temp_uploads")
# Создаем каталог для временных загрузок, если нужно
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Environment variable to control vector store usage
USE_VECTOR_STORE = os.getenv("USE_VECTOR_STORE", "false").lower() in ("1", "true", "yes")

def get_file_vector_store_dir(task_id: str, filename: str) -> str:
    """
    Create a temporary vector store directory for a specific file.

    Args:
        task_id: The unique task ID
        filename: The name of the file

    Returns:
        Path to the vector store directory
    """
    # Create a temporary directory inside TEMP_UPLOAD_DIR
    file_st_dir = Path(TEMP_UPLOAD_DIR) / f"vs_{task_id}_{Path(filename).stem}"
    file_st_dir.mkdir(parents=True, exist_ok=True)
    return str(file_st_dir)

def initialize_document_processor(storage_path: str = None, use_vector_store: bool = None):
    """
    Initialize DocumentProcessor with FAISS index check.
    If index doesn't exist, create an empty one.

    Args:
        storage_path: Path to the vector store directory
        use_vector_store: Override for USE_VECTOR_STORE environment variable

    Returns:
        Initialized DocumentProcessor
    """
    # Import here to avoid circular imports
    from ai.vector_store_client import VectorStoreClient

    # Use the provided value or fall back to the environment variable
    use_vs = USE_VECTOR_STORE if use_vector_store is None else use_vector_store

    # If vector store is disabled, return a processor without it
    if not use_vs or storage_path is None:
        print("Vector store is disabled. Initializing DocumentProcessor without vector store.")
        return DocumentProcessor(storage_path=None, use_vector_store=False)

    # Otherwise, initialize with vector store
    # Create a vector store client
    vector_store_client = VectorStoreClient()

    # Create the vector store if it doesn't exist
    vector_store_client.create_vector_store(storage_path)

    # Now we can safely load
    # For backward compatibility, still return a DocumentProcessor
    # but it will use the vector store service internally
    return DocumentProcessor(storage_path=storage_path, use_vector_store=True)

def ensure_patient_identifiers_cached(filename: str, file_path: str):
    """
    Ensure patient identifiers are cached for the given file.
    If not in cache, run PatientIdentifiersAgent to cache them.

    Args:
        filename: The name of the PDF file
        file_path: Full path to the PDF file
    """
    pmid = get_pmid_for_filename(filename)
    agent = PatientIdentifiersAgent(pmid)
    agent.setup()
    # Run agent to write to cache
    initial_state = {
        "pdf_filepath": file_path,
        "patient_identifiers": [],
        "messages": [{"role": "user", "content": f"Extract patient identifiers from {filename}"}]
    }
    final_state = agent.run(initial_state)
    # final_state.patient_identifiers will be cached via agent.save_cache()

# In-memory storage for tracking PDF analysis progress
# In a production environment, this should be replaced with a database
pdf_analysis_progress: Dict[str, Dict[str, float]] = {}

# In-memory storage for tracking ZIP analysis progress
# In a production environment, this should be replaced with a database
zip_analysis_progress: Dict[str, Dict[str, Any]] = {}

def convert_word_to_text(word_path: str) -> str:
    """
    Convert a Word document to plain text.

    Note: This function requires the python-docx library.
    Install it with: pip install python-docx

    Args:
        word_path: Path to the Word document

    Returns:
        The extracted text from the document
    """
    try:
        import docx
        doc = docx.Document(word_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error converting Word document to text: {e}")
        return f"ERROR: Could not extract text from {word_path}: {str(e)}"

def process_text_as_pdf(text: str, original_path: str) -> Dict[str, Any]:
    """
    Process text content as if it were extracted from a PDF.

    Args:
        text: The text content to process
        original_path: The original file path (for reference)

    Returns:
        Dictionary containing the processing results
    """
    from ai.agents.publication_details_agent import PublicationDetailsAgent
    from ai.agents.patient_identifiers_agent import PatientIdentifiersAgent
    from ai.agents.questions_processing_agent import QuestionsProcessingAgent

    try:
        # Create a temporary file with the text content
        temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"temp_text_{os.path.basename(original_path)}.txt")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Extract publication details
        pub_agent = PublicationDetailsAgent()
        pub_agent.setup()
        pub_state = {
            "pdf_filepath": temp_file_path,
            "publication_details": None,
            "pmid": None,
            "messages": [
                {"role": "user", "content": f"Extract publication details and PMID from text file"}
            ]
        }
        pub_final_state = pub_agent.run(pub_state)

        # Get PMID
        pmid = pub_final_state.get("pmid")

        # Extract patient identifiers
        patient_agent = PatientIdentifiersAgent(pmid)
        patient_agent.setup()
        patient_state = {
            "pdf_filepath": temp_file_path,
            "patient_identifiers": [],
            "messages": [
                {"role": "user", "content": f"Extract patient identifiers from text file"}
            ]
        }
        patient_final_state = patient_agent.run(patient_state)

        # Process patient questions
        questions_agent = QuestionsProcessingAgent(pmid)
        questions_agent.setup()
        questions_state = {
            "pdf_filepath": temp_file_path,
            "mapping_items": [],
            "patient_identifiers": patient_final_state.get("patient_identifiers", []),
            "patient_questions": [],
            "patient_answers": [],
            "vector_store": None,
            "pmid": pmid,
            "messages": [
                {"role": "user", "content": f"Process questions for patients in text file"}
            ]
        }
        questions_final_state = questions_agent.run(questions_state)

        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        # Return the results
        return {
            "filename": os.path.basename(original_path),
            "publication_details": pub_final_state.get("publication_details", {}),
            "pmid": pmid,
            "patient_identifiers": patient_final_state.get("patient_identifiers", []),
            "patient_answers": questions_final_state.get("patient_answers", [])
        }
    except Exception as e:
        print(f"Error processing text as PDF: {e}")
        import traceback
        traceback.print_exc()
        return {
            "filename": os.path.basename(original_path),
            "error": str(e)
        }

def process_zip_archive(task_id: str, zip_path: str):
    """
    Process a ZIP archive containing PDF, Word, and Excel files.

    Args:
        task_id: The unique task ID for tracking progress
        zip_path: Path to the ZIP file

    Returns:
        None (updates zip_analysis_progress and creates result file)
    """
    try:
        # Create extraction directory
        extract_dir = os.path.join(TEMP_UPLOAD_DIR, f"zip_{task_id}")
        os.makedirs(extract_dir, exist_ok=True)

        # 1) Extract ZIP
        with zipfile.ZipFile(zip_path, 'r') as archive:
            archive.extractall(extract_dir)

        # 2) Collect all files
        file_paths = []
        for root, _, files in os.walk(extract_dir):
            for name in files:
                file_paths.append(os.path.join(root, name))
        total = len(file_paths)

        if total == 0:
            zip_analysis_progress[task_id] = {"status": "error", "progress": 0, "message": "No files found in ZIP archive"}
            # Clean up
            shutil.rmtree(extract_dir)
            os.remove(zip_path)
            return

        # Initialize results list
        results = []

        # Process each file
        for idx, path in enumerate(file_paths, start=1):
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == ".pdf":
                    filename = os.path.basename(path)

                    # Create per-file VectorStore, if enabled
                    if USE_VECTOR_STORE:
                        vs_dir = get_file_vector_store_dir(task_id, filename)
                        processor = initialize_document_processor(storage_path=vs_dir, use_vector_store=True)
                    else:
                        # Initialize without VectorStore
                        processor = initialize_document_processor(storage_path=None, use_vector_store=False)

                    # Cache patient identifiers
                    ensure_patient_identifiers_cached(filename, path)

                    # Process PDF using refactored workflow
                    from ai.workflows.pdf_processing import process_pdf_file
                    pdf_results = process_pdf_file(path)

                    # Clean up per-file vector store if it was created
                    if USE_VECTOR_STORE:
                        vs_dir = get_file_vector_store_dir(task_id, filename)
                        if os.path.exists(vs_dir):
                            try:
                                shutil.rmtree(vs_dir, ignore_errors=True)
                                print(f"Removed temporary vector store directory: {vs_dir}")
                            except Exception as e:
                                print(f"Error removing vector store directory {vs_dir}: {e}")

                    # Extract patient answers if available
                    if "patient_answers" in pdf_results:
                        results.extend(pdf_results["patient_answers"])
                    else:
                        # Try to get patient answers from the questions processing agent
                        pmid = pdf_results.get("pmid")
                        if pmid:
                            questions_agent = QuestionsProcessingAgent(pmid)
                            questions_agent.setup()
                            questions_state = {
                                "pdf_filepath": path,
                                "mapping_items": [],
                                "patient_identifiers": [],
                                "patient_questions": [],
                                "patient_answers": [],
                                "vector_store": None,
                                "pmid": pmid,
                                "messages": [
                                    {"role": "user", "content": f"Process questions for patients in {os.path.basename(path)}"}
                                ]
                            }
                            try:
                                questions_final_state = questions_agent.run(questions_state)
                                patient_answers = questions_final_state.get("patient_answers", [])
                                results.extend(patient_answers)
                            except Exception as e:
                                print(f"Error processing questions for PDF {path}: {e}")
                                # Add error entry
                                results.append({
                                    "filename": os.path.basename(path),
                                    "error": f"Error processing questions: {str(e)}"
                                })
                        else:
                            # Add basic info if no patient answers
                            results.append({
                                "filename": os.path.basename(path),
                                "publication_details": pdf_results.get("publication_details", {}),
                                "pmid": pdf_results.get("pmid")
                            })

                elif ext in (".doc", ".docx"):
                    # Convert Word to text and process
                    text = convert_word_to_text(path)
                    if text.startswith("ERROR:"):
                        # Add error entry
                        results.append({
                            "filename": os.path.basename(path),
                            "error": text
                        })
                    else:
                        # Process text as PDF
                        text_results = process_text_as_pdf(text, path)
                        if "patient_answers" in text_results:
                            results.extend(text_results["patient_answers"])
                        elif "error" in text_results:
                            results.append(text_results)
                        else:
                            results.append({
                                "filename": os.path.basename(path),
                                "publication_details": text_results.get("publication_details", {}),
                                "pmid": text_results.get("pmid")
                            })

                elif ext in (".xls", ".xlsx"):
                    # Process Excel file
                    try:
                        df = pd.read_excel(path)
                        for _, row in df.iterrows():
                            results.append(row.to_dict())
                    except Exception as e:
                        print(f"Error processing Excel file {path}: {e}")
                        results.append({
                            "filename": os.path.basename(path),
                            "error": f"Error processing Excel file: {str(e)}"
                        })
                # Skip other file types
            except Exception as e:
                print(f"Error processing file {path}: {e}")
                import traceback
                traceback.print_exc()
                # Add error entry
                results.append({
                    "filename": os.path.basename(path),
                    "error": f"Error processing file: {str(e)}"
                })

            # Update progress
            zip_analysis_progress[task_id]["progress"] = int(idx / total * 100)

        # Create final Excel file
        if results:
            # Convert results to DataFrame
            final_df = pd.DataFrame(results)

            # Save to Excel
            out_path = os.path.join(TEMP_UPLOAD_DIR, f"{task_id}_results.xlsx")
            final_df.to_excel(out_path, index=False)

            # Update status
            zip_analysis_progress[task_id]["status"] = "done"
            zip_analysis_progress[task_id]["progress"] = 100
        else:
            # No results
            zip_analysis_progress[task_id]["status"] = "error"
            zip_analysis_progress[task_id]["message"] = "No valid results found in any files"

        # Clean up
        shutil.rmtree(extract_dir)
        os.remove(zip_path)

    except Exception as e:
        print(f"Error processing ZIP archive: {e}")
        import traceback
        traceback.print_exc()

        # Update status
        zip_analysis_progress[task_id]["status"] = "error"
        zip_analysis_progress[task_id]["message"] = str(e)

        # Clean up
        try:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            if os.path.exists(zip_path):
                os.remove(zip_path)
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")



router = APIRouter()

@router.post("/file/upload_zip")
async def upload_zip(file: UploadFile = File(...)):
    """
    Upload a ZIP file for batch processing.

    This endpoint accepts a ZIP archive containing multiple files (PDF, Word, Excel)
    and initiates background processing. It returns a task_id that can be used to
    check the processing status and download results.

    Args:
        file: The ZIP file to upload

    Returns:
        Dictionary with task_id for tracking the processing
    """
    # Check file extension
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    # Generate a unique task ID
    task_id = str(uuid4())

    # Save the ZIP file
    zip_path = os.path.join(TEMP_UPLOAD_DIR, f"{task_id}.zip")

    # Read and save the file
    content = await file.read()
    with open(zip_path, "wb") as buffer:
        buffer.write(content)

    # Initialize progress tracking
    zip_analysis_progress[task_id] = {"status": "processing", "progress": 0}

    # Start background processing
    thread = threading.Thread(
        target=process_zip_archive,
        args=(task_id, zip_path),
        daemon=True
    )
    thread.start()

    return {"task_id": task_id}

@router.get("/file/zip_analysis_status")
async def get_zip_analysis_status(task_id: str = Query(..., description="Task ID from upload_zip")):
    """
    Check the status of a ZIP file processing task.

    This endpoint returns the current status and progress of a ZIP processing task.

    Args:
        task_id: The task ID returned by the upload_zip endpoint

    Returns:
        Dictionary with status and progress information
    """
    # Check if task exists
    if task_id not in zip_analysis_progress:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' not found")

    # Return the current status
    return zip_analysis_progress[task_id]

@router.get("/file/download_zip_results")
async def download_zip_results(task_id: str = Query(..., description="Task ID from upload_zip")):
    """
    Download the results of a completed ZIP processing task.

    This endpoint returns the Excel file containing the results of the ZIP processing.
    The task must be completed (status="done") before results can be downloaded.

    Args:
        task_id: The task ID returned by the upload_zip endpoint

    Returns:
        Excel file as a download
    """
    # Check if task exists
    if task_id not in zip_analysis_progress:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' not found")

    # Check if task is completed
    task_status = zip_analysis_progress[task_id]
    if task_status.get("status") != "done":
        raise HTTPException(
            status_code=404, 
            detail=f"Results not ready. Current status: {task_status.get('status')}, progress: {task_status.get('progress')}%"
        )

    # Check if results file exists
    results_path = os.path.join(TEMP_UPLOAD_DIR, f"{task_id}_results.xlsx")
    if not os.path.exists(results_path):
        raise HTTPException(status_code=404, detail="Results file not found")

    # Return the file
    return FileResponse(
        path=results_path,
        filename="batch_results.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

@router.post("/gene/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Недопустимый тип файла")

    # Временный путь файла
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, file.filename)

    # Сохраняем файл временно - без copyfileobj
    content = await file.read()
    with open(temp_file_path, "wb") as buffer:
        buffer.write(content)

    return {"filename": file.filename, "message": "PDF ready for processing"}


@router.post("/file/upload_pdf_for_processing")
async def upload_pdf_for_processing(file: UploadFile = File(...)):
    """
    Upload a PDF file for processing.

    This endpoint accepts a PDF file, validates its type, and saves it to a temporary directory
    for further processing by other endpoints.

    Args:
        file: The PDF file to upload

    Returns:
        JSON with filename and success message
    """
    if not file.content_type == "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF is allowed.")

    # Save file to temporary directory
    file_location = Path(TEMP_UPLOAD_DIR) / file.filename
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    finally:
        await file.close()  # Ensure file is closed

    return {"filename": file.filename, "message": "File uploaded successfully, ready for processing."}



@router.get("/file/patient_identifiers")
async def get_patient_identifiers(filename: str = Query(..., description="Filename of the uploaded PDF")):
    """
    Get patient identifiers from a previously uploaded PDF file.

    This endpoint uses the PatientIdentifiersAgent to extract patient identifiers
    from the specified PDF file. The file must have been previously uploaded.

    Args:
        filename: The name of the PDF file to analyze

    Returns:
        A list of patient identifiers found in the document
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    try:
        # Get PMID for the file if available
        pmid = get_pmid_for_filename(filename)

        # Initialize the patient identifiers agent with PMID if available
        agent = PatientIdentifiersAgent(pmid)
        agent.setup()

        # Initialize the state
        initial_state = {
            "pdf_filepath": file_path,
            "patient_identifiers": [],
            "messages": [
                {"role": "user", "content": f"Extract patient identifiers from {filename}"}
            ]
        }

        # Run the agent
        final_state = agent.run(initial_state)

        # Return the patient identifiers
        return {
            "filename": filename,
            "patient_identifiers": final_state.get("patient_identifiers", [])
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing file {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/file/publication_details")
async def get_publication_details(filename: str = Query(..., description="Filename of the uploaded PDF")):
    """
    Get publication metadata from a previously uploaded PDF file.

    This endpoint uses the PublicationDetailsAgent to extract publication details
    including title, author information, year, and PMID from the specified PDF file.
    The file must have been previously uploaded.

    Args:
        filename: The name of the PDF file to analyze

    Returns:
        Publication details and PMID if available
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    try:
        # Initialize the publication details agent
        agent = PublicationDetailsAgent()
        agent.setup()

        # Initialize the state
        initial_state = {
            "pdf_filepath": file_path,
            "publication_details": None,
            "pmid": None,
            "messages": [
                {"role": "user", "content": f"Extract publication details and PMID from {filename}"}
            ]
        }

        # Run the agent
        final_state = agent.run(initial_state)

        # Return the publication details and PMID
        return {
            "filename": filename,
            "publication_details": final_state.get("publication_details"),
            "pmid": final_state.get("pmid")
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing file {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/file/process_patient_questions")
async def process_patient_questions(filename: str = Query(..., description="Filename of the uploaded PDF")):
    """
    Process patient questions for a previously uploaded PDF file.

    This endpoint uses the QuestionsProcessingAgent to:
    1. Load mapping data from JSON
    2. Initialize a vector store for efficient querying
    3. Get patient identifiers from cache (requires prior extraction)
    4. Generate questions for each patient
    5. Process questions to get answers

    Note: Patient identifiers must have been previously extracted using
    the patient_identifiers endpoint before calling this endpoint.

    Args:
        filename: The name of the PDF file to analyze

    Returns:
        Processed patient data with answers to configured questions
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    try:
        # Get PMID for the file if available
        pmid = get_pmid_for_filename(filename)

        # Initialize the questions processing agent with PMID if available
        agent = QuestionsProcessingAgent(pmid)
        agent.setup()

        # Initialize the state
        initial_state = {
            "pdf_filepath": file_path,
            "mapping_items": [],
            "patient_identifiers": [],
            "patient_questions": [],
            "patient_answers": [],
            "vector_store": None,
            "pmid": pmid,  # Include PMID in the state
            "messages": [
                {"role": "user", "content": f"Process questions for patients in {filename}"}
            ]
        }

        # Run the agent
        try:
            final_state = agent.run(initial_state)

            # Return the processed data
            return {
                "filename": filename,
                "mapping_items_count": len(final_state.get("mapping_items", [])),
                "patient_identifiers_count": len(final_state.get("patient_identifiers", [])),
                "patient_answers": final_state.get("patient_answers", [])
            }
        except ValueError as e:
            # This catches the specific error when patient identifiers are not found in cache
            raise HTTPException(
                status_code=400,
                detail="Patient identifiers not found in cache. Please extract patient identifiers first."
            )

    except Exception as e:
        # Log the error for debugging
        print(f"Error processing file {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")



@router.post("/file/update_publication_details")
async def update_publication_details(
    filename: str = Body(..., description="Filename of the uploaded PDF"),
    details: Dict[str, str] = Body(..., description="Updated publication details")
):
    """
    Update publication details for a previously uploaded PDF file.

    This endpoint allows manual updates to publication metadata such as title,
    author information, and year. The details are stored in the cache for
    future use by the publication details agent.

    Args:
        filename: The name of the PDF file to update
        details: Dictionary containing updated publication details

    Returns:
        Updated publication details
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    try:
        # Initialize the publication details agent to access its cache
        agent = PublicationDetailsAgent()

        # Load the current cache
        pmid_cache = agent.load_cache()



        # Update the cache with the new details
        pmid_cache[filename] = details

        # Save the updated cache
        agent.save_cache(pmid_cache)

        return {
            "filename": filename,
            "publication_details": details,
            "message": "Publication details updated successfully"
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error updating publication details for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating publication details: {str(e)}")



@router.post("/file/update_patient_identifier")
async def update_patient_identifier(
    filename: str = Body(..., description="Filename of the uploaded PDF"),
    patient_id: str = Body(..., description="Patient identifier"),
    family_id: Optional[str] = Body(None, description="Family identifier"),
    data: Dict[str, Any] = Body(..., description="Additional patient data")
):
    """
    Update or add a patient identifier for a previously uploaded PDF file.

    This endpoint allows manual updates to patient identifier information.
    The patient data is stored in the cache for future use by the patient
    identifiers agent.

    Args:
        filename: The name of the PDF file to update
        patient_id: The patient identifier
        family_id: The family identifier (optional)
        data: Dictionary containing additional patient data

    Returns:
        Updated patient identifier information
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    try:
        # Get PMID for the file if available
        pmid = get_pmid_for_filename(filename)

        # Initialize the patient identifiers agent with PMID if available
        agent = PatientIdentifiersAgent(pmid)

        # Load the current cache
        cache = agent.load_cache()

        # Get the patient identifiers list from cache
        patient_cache_key = agent.patient_cache_key
        patient_identifiers = cache.get(patient_cache_key, [])

        # Check if patient identifier is not a list
        if not isinstance(patient_identifiers, list):
            patient_identifiers = []

        # Create or update patient identifier
        patient_entry = {"patient": patient_id}
        if family_id:
            patient_entry["family"] = family_id

        # Add additional data
        patient_entry.update(data)

        # Check if this patient already exists in the list
        updated = False
        for i, existing in enumerate(patient_identifiers):
            if (existing.get("patient") == patient_id and 
                existing.get("family") == family_id):
                # Update existing entry
                patient_identifiers[i] = patient_entry
                updated = True
                break

        # If patient wasn't found, add as new
        if not updated:
            patient_identifiers.append(patient_entry)

        # Save updated patient identifiers list back to cache
        cache[patient_cache_key] = patient_identifiers
        agent.save_cache(cache)

        return {
            "filename": filename,
            "patient_id": patient_id,
            "family_id": family_id,
            "data": data,
            "message": "Patient identifier updated successfully",
            "patient_identifiers_count": len(patient_identifiers)
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Error updating patient identifier for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error updating patient identifier: {str(e)}")

    @router.delete("/file/delete_patient_identifier")
    async def delete_patient_identifier(
            filename: str = Body(..., description="Filename of the uploaded PDF"),
            patient_id: str = Body(..., description="Patient identifier to delete"),
            family_id: Optional[str] = Body(None, description="Family identifier")
    ):
        """
        Delete a patient identifier for a previously uploaded PDF file.

        This endpoint removes a specific patient identifier from the cache.
        It requires both the patient ID and optionally the family ID to ensure
        the correct patient is deleted.

        Args:
            filename: The name of the PDF file
            patient_id: The patient identifier to delete
            family_id: The family identifier (optional)

        Returns:
            Confirmation of deletion and updated count
        """
        # Construct the full path to the uploaded file
        file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

        # Check if the file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

        try:
            # Initialize the patient identifiers agent to access its cache
            agent = PatientIdentifiersAgent()

            # Load the current cache
            cache = agent.load_cache()

            # Get the patient identifiers list from cache
            patient_cache_key = agent.patient_cache_key
            patient_identifiers = cache.get(patient_cache_key, [])

            # Check if patient identifier is not a list
            if not isinstance(patient_identifiers, list):
                raise HTTPException(status_code=400, detail="Patient identifiers cache is not in the expected format")

            # Find and remove the patient
            original_count = len(patient_identifiers)
            new_patient_identifiers = []

            for entry in patient_identifiers:
                # Skip the patient we want to delete
                if (entry.get("patient") == patient_id and
                        (family_id is None or entry.get("family") == family_id)):
                    continue
                new_patient_identifiers.append(entry)

            new_count = len(new_patient_identifiers)

            # If no patient was removed, return an error
            if original_count == new_count:
                raise HTTPException(
                    status_code=404,
                    detail=f"Patient with ID '{patient_id}' and family ID '{family_id}' not found"
                )

            # Save updated patient identifiers list back to cache
            cache[patient_cache_key] = new_patient_identifiers
            agent.save_cache(cache)

            return {
                "filename": filename,
                "patient_id": patient_id,
                "family_id": family_id,
                "message": "Patient identifier deleted successfully",
                "deleted_count": original_count - new_count,
                "patient_identifiers_count": new_count
            }
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log the error for debugging
            print(f"Error deleting patient identifier for {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error deleting patient identifier: {str(e)}")


@router.post("/file/add_patient_identifier")
async def add_patient_identifier(
        filename: str = Body(..., description="Filename of the uploaded PDF"),
        patient_data: Dict[str, Any] = Body(..., description="Complete patient data to add")
):
    """
    Add a new patient identifier for a previously uploaded PDF file.

    This endpoint adds a complete patient entry to the patient identifiers list.
    The patient_data should include at minimum a 'patient' key with the patient ID,
    and optionally a 'family' key with the family ID.

    Args:
        filename: The name of the PDF file
        patient_data: Dictionary containing the complete patient data to add

    Returns:
        Confirmation of addition and updated count
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    # Validate that patient_data contains at least a patient ID
    if not patient_data.get("patient"):
        raise HTTPException(status_code=400, detail="Patient data must include a 'patient' identifier")

    try:
        # Get PMID for the file if available
        pmid = get_pmid_for_filename(filename)

        # Initialize the patient identifiers agent with PMID if available
        agent = PatientIdentifiersAgent(pmid)

        # Load the current cache
        cache = agent.load_cache()

        # Get the patient identifiers list from cache
        patient_cache_key = agent.patient_cache_key
        patient_identifiers = cache.get(patient_cache_key, [])

        # Check if patient identifier is not a list
        if not isinstance(patient_identifiers, list):
            patient_identifiers = []

        # Check if a patient with the same identifiers already exists
        patient_id = patient_data.get("patient")
        family_id = patient_data.get("family")

        for existing in patient_identifiers:
            if (existing.get("patient") == patient_id and
                    existing.get("family") == family_id):
                raise HTTPException(
                    status_code=409,
                    detail=f"Patient with ID '{patient_id}' and family ID '{family_id}' already exists"
                )

        # Add the new patient data
        patient_identifiers.append(patient_data)

        # Save updated patient identifiers list back to cache
        cache[patient_cache_key] = patient_identifiers
        agent.save_cache(cache)

        return {
            "filename": filename,
            "patient_data": patient_data,
            "message": "Patient identifier added successfully",
            "patient_identifiers_count": len(patient_identifiers)
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Error adding patient identifier for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error adding patient identifier: {str(e)}")


@router.delete("/file/delete_patient_identifier")
async def delete_patient_identifier(
        filename: str = Body(..., description="Filename of the uploaded PDF"),
        patient_id: str = Body(..., description="Patient identifier to delete"),
        family_id: Optional[str] = Body(None, description="Family identifier")
):
    """
    Delete a patient identifier for a previously uploaded PDF file.

    This endpoint removes a specific patient identifier from the cache.
    It requires both the patient ID and optionally the family ID to ensure
    the correct patient is deleted.

    Args:
        filename: The name of the PDF file
        patient_id: The patient identifier to delete
        family_id: The family identifier (optional)

    Returns:
        Confirmation of deletion and updated count
    """
    # Construct the full path to the uploaded file
    file_path = os.path.join(TEMP_UPLOAD_DIR, filename)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")

    try:
        # Get PMID for the file if available
        pmid = get_pmid_for_filename(filename)

        # Initialize the patient identifiers agent with PMID if available
        agent = PatientIdentifiersAgent(pmid)

        # Load the current cache
        cache = agent.load_cache()

        # Get the patient identifiers list from cache
        patient_cache_key = agent.patient_cache_key
        patient_identifiers = cache.get(patient_cache_key, [])

        # Check if patient identifier is not a list
        if not isinstance(patient_identifiers, list):
            raise HTTPException(status_code=400, detail="Patient identifiers cache is not in the expected format")

        # Find and remove the patient
        original_count = len(patient_identifiers)
        new_patient_identifiers = []

        for entry in patient_identifiers:
            # Skip the patient we want to delete
            if (entry.get("patient") == patient_id and
                    (family_id is None or entry.get("family") == family_id)):
                continue
            new_patient_identifiers.append(entry)

        new_count = len(new_patient_identifiers)

        # If no patient was removed, return an error
        if original_count == new_count:
            raise HTTPException(
                status_code=404,
                detail=f"Patient with ID '{patient_id}' and family ID '{family_id}' not found"
            )

        # Save updated patient identifiers list back to cache
        cache[patient_cache_key] = new_patient_identifiers
        agent.save_cache(cache)

        return {
            "filename": filename,
            "patient_id": patient_id,
            "family_id": family_id,
            "message": "Patient identifier deleted successfully",
            "deleted_count": original_count - new_count,
            "patient_identifiers_count": new_count
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Error deleting patient identifier for {filename}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting patient identifier: {str(e)}")

@router.get("/file/mapping_data")
async def get_mapping_data():
    """
    Retrieve the content of the mapping_data_all.json file.

    This endpoint returns the complete content of the mapping_data_all.json file
    which contains all the mappings and configurations.

    Returns:
        The complete content of the mapping_data_all.json file as JSON
    """
    try:
        mapping_file_path = os.path.join(".questions", "mapping_data.json")

        if not os.path.exists(mapping_file_path):
            raise HTTPException(status_code=404, detail="Mapping data file not found")

        with open(mapping_file_path, 'r', encoding='utf-8') as file:
            mapping_data = json.load(file)

        return mapping_data

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing mapping data file: {str(e)}")
    except Exception as e:
        print(f"Error retrieving mapping data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving mapping data: {str(e)}")


@router.post("/file/mapping_data")
async def manage_mapping_data(
    action: str = Body(..., description="Action to perform: add, update, delete"),
    field: Optional[str] = Body(None, description="Field name to update or delete"),
    data: Optional[Dict[str, Any]] = Body(None, description="Field data for add or update")
):
    """
    Manage the mapping_data_all.json file by adding, updating, or deleting fields.

    This endpoint handles multiple operations on the mapping data file based on the action parameter.

    Args:
        action: The operation to perform (add, update, delete)
        field: Field name to update or delete (required for update and delete)
        data: Field data for add or update operations (required for add and update)

    Returns:
        Confirmation of the action and the updated mapping data
    """
    try:
        # Validate parameters based on action
        if action not in ["add", "update", "delete"]:
            raise HTTPException(status_code=400, detail="Action must be one of: add, update, delete")

        if action in ["update", "delete"] and not field:
            raise HTTPException(status_code=400, detail=f"Field name is required for {action} action")

        if action in ["add", "update"] and not data:
            raise HTTPException(status_code=400, detail=f"Field data is required for {action} action")

        if action == "add" and not data.get("field"):
            raise HTTPException(status_code=400, detail="New mapping data must include a 'field' property")

        # Get the path to mapping_data_all.json
        mapping_file_path = os.path.join(".questions", "mapping_data.json")

        # Check if the file exists
        if not os.path.exists(mapping_file_path):
            raise HTTPException(status_code=404, detail="Mapping data file not found")

        # Open and read the file
        with open(mapping_file_path, 'r', encoding='utf-8') as file:
            mapping_data = json.load(file)

        # Perform the requested action
        if action == "add":
            # Check if the field already exists
            for item in mapping_data:
                if item.get("field") == data["field"]:
                    raise HTTPException(
                        status_code=409, 
                        detail=f"Field '{data['field']}' already exists in mapping data"
                    )

            # Add the new field
            mapping_data.append(data)
            message = f"Field '{data['field']}' added successfully"

        elif action == "update":
            # Find and update the field
            field_found = False
            for i, item in enumerate(mapping_data):
                if item.get("field") == field:
                    mapping_data[i] = data
                    field_found = True
                    break

            if not field_found:
                raise HTTPException(status_code=404, detail=f"Field '{field}' not found in mapping data")

            message = f"Field '{field}' updated successfully"

        elif action == "delete":
            # Check if the field exists before trying to delete
            original_length = len(mapping_data)
            mapping_data = [item for item in mapping_data if item.get("field") != field]

            if len(mapping_data) == original_length:
                raise HTTPException(status_code=404, detail=f"Field '{field}' not found in mapping data")

            message = f"Field '{field}' deleted successfully"

        # Write the updated data back to the file
        with open(mapping_file_path, 'w', encoding='utf-8') as file:
            json.dump(mapping_data, file, indent=2)

        return {
            "message": message,
            "mapping_data": mapping_data
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing mapping data file: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error managing mapping data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error managing mapping data: {str(e)}")


@router.get("/cache/questions")
async def get_cached_questions(
    pmid: Optional[str] = Query(None, description="PMID of the document. Leave empty to retrieve all cached questions.")
):
    """
    Retrieve cached questions for a specific PMID or all PMIDs.

    This endpoint allows retrieving cached questions by providing the PMID.
    If no PMID is provided, it returns all cached questions from all PMIDs.

    Args:
        pmid: PMID of the document (optional)

    Returns:
        A list of questions found in the cache
    """
    try:
        if pmid:
            # Initialize the questions processing agent with the provided PMID
            agent = QuestionsProcessingAgent(pmid)

            # Get cached questions for this PMID
            cached_questions = agent.get_cached_questions()

            # If no cached questions are found, return an empty list
            if not cached_questions:
                return {
                    "pmid": pmid,
                    "questions": [],
                    "count": 0,
                    "message": f"No cached questions found for PMID '{pmid}'"
                }

            # Return the list of cached questions
            return {
                "pmid": pmid,
                "questions": cached_questions,
                "count": len(cached_questions)
            }
        else:
            # Initialize the questions processing agent without a specific PMID
            agent = QuestionsProcessingAgent()

            # Get all cached questions from all PMIDs
            all_cached_questions = agent.get_cached_questions(all_pmids=True)

            # If no cached questions are found, return an empty list
            if not all_cached_questions:
                return {
                    "pmid": "all",
                    "questions": [],
                    "count": 0,
                    "message": "No cached questions found"
                }

            # Return the list of all cached questions
            return {
                "pmid": "all",
                "questions": all_cached_questions,
                "count": len(all_cached_questions)
            }
    except Exception as e:
        # Log the error for debugging
        print(f"Error retrieving cached questions: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving cached questions: {str(e)}")

@router.get("/cache/response")
async def get_cached_response(
    pmid: str = Query(..., description="PMID of the document"),
    question: str = Query(..., description="Question to retrieve the cached response for")
):
    """
    Retrieve a cached response for a specific PMID and question.

    This endpoint allows retrieving cached responses by providing the PMID and question.
    It returns the raw response text if found in the cache.

    Args:
        pmid: PMID of the document
        question: Question to retrieve the cached response for

    Returns:
        The cached response if found
    """
    try:
        # Initialize the questions processing agent with the provided PMID
        agent = QuestionsProcessingAgent(pmid)

        # Generate a cache identifier for the question
        cache_identifier = agent.generate_cache_identifier(question)

        # Try to load the cached response
        cached_response = agent.load_from_cache(cache_identifier)

        # If no cached response is found, return an error
        if cached_response is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached response found for PMID '{pmid}' and the provided question"
            )

        # Return the cached response
        return {
            "pmid": pmid,
            "question": question,
            "response": cached_response
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error for debugging
        print(f"Error retrieving cached response: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving cached response: {str(e)}")

@router.delete("/cache")
async def clear_cache(
    agent_type: Optional[str] = Query(None, description="Type of agent cache to clear (publication_details, patient_identifiers, questions_processing)"),
    pmid: Optional[str] = Query(None, description="PMID of the document whose cache to clear")
):
    """
    Clear the cache files.

    This endpoint allows clearing of cache files with various filtering options:
    - If no parameters are provided, all cache files will be cleared
    - If agent_type is provided, only the cache files for that agent will be cleared
    - If pmid is provided, only the cache files for that document will be cleared
    - If both agent_type and pmid are provided, only the cache files for that agent and document will be cleared

    Args:
        agent_type: Type of agent cache to clear (optional)
        pmid: PMID of the document whose cache to clear (optional)

    Returns:
        Confirmation of the cache clearing operation
    """
    try:
        # Validate agent_type if provided
        valid_agent_types = ["publication_details", "patient_identifiers", "questions_processing"]
        if agent_type and agent_type not in valid_agent_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid agent_type. Must be one of: {', '.join(valid_agent_types)}"
            )

        # Map agent types to their cache file names
        agent_cache_files = {
            "publication_details": "pmid_cache.json",
            "patient_identifiers": "patient_cache.json",
            "questions_processing": "patient_cache.json"  # Uses the same cache file as patient_identifiers
        }

        # Track what was cleared for the response
        cleared_items = []

        # Case 1: Clear all cache
        if not agent_type and not pmid:
            if CACHE_DIR.exists():
                # Clear all files in the main cache directory
                for cache_file in CACHE_DIR.glob("*.json"):
                    cache_file.unlink()
                    cleared_items.append(str(cache_file))

                # Clear all PMID-specific subdirectories
                for pmid_dir in CACHE_DIR.glob("*"):
                    if pmid_dir.is_dir():
                        shutil.rmtree(pmid_dir)
                        cleared_items.append(str(pmid_dir))

                return {
                    "message": "All cache files cleared successfully",
                    "cleared_items": cleared_items
                }
            else:
                return {
                    "message": "Cache directory does not exist, nothing to clear"
                }

        # Case 2: Clear cache for a specific agent type
        elif agent_type and not pmid:
            cache_file = agent_cache_files[agent_type]

            # Clear the main cache file for this agent
            main_cache_file = CACHE_DIR / cache_file
            if main_cache_file.exists():
                main_cache_file.unlink()
                cleared_items.append(str(main_cache_file))

            # Clear all PMID-specific cache files for this agent
            for pmid_dir in CACHE_DIR.glob("*"):
                if pmid_dir.is_dir():
                    pmid_cache_file = pmid_dir / cache_file
                    if pmid_cache_file.exists():
                        pmid_cache_file.unlink()
                        cleared_items.append(str(pmid_cache_file))

            return {
                "message": f"Cache files for agent type '{agent_type}' cleared successfully",
                "cleared_items": cleared_items
            }

        # Case 3: Clear cache for a specific PMID
        elif pmid and not agent_type:
            pmid_dir = CACHE_DIR / pmid

            if pmid_dir.exists():
                shutil.rmtree(pmid_dir)
                cleared_items.append(str(pmid_dir))

                return {
                    "message": f"Cache files for PMID '{pmid}' cleared successfully",
                    "cleared_items": cleared_items
                }
            else:
                return {
                    "message": f"No cache directory found for PMID '{pmid}', nothing to clear"
                }

        # Case 4: Clear cache for a specific agent type and PMID
        else:  # both agent_type and pmid are provided
            cache_file = agent_cache_files[agent_type]
            pmid_dir = CACHE_DIR / pmid

            if pmid_dir.exists():
                pmid_cache_file = pmid_dir / cache_file
                if pmid_cache_file.exists():
                    pmid_cache_file.unlink()
                    cleared_items.append(str(pmid_cache_file))

                    return {
                        "message": f"Cache file for agent type '{agent_type}' and PMID '{pmid}' cleared successfully",
                        "cleared_items": cleared_items
                    }
                else:
                    return {
                        "message": f"No cache file found for agent type '{agent_type}' and PMID '{pmid}', nothing to clear"
                    }
            else:
                return {
                    "message": f"No cache directory found for PMID '{pmid}', nothing to clear"
                }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error clearing cache: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@router.get("/documents")
def get_documents():
    """
    Возвращает список документов из pmid_cache.json.
    Если файла нет, возвращается пустой список.

    Returns:
        Список документов.
    """
    try:
        if os.path.exists(PMID_CACHE_PATH):
            with open(PMID_CACHE_PATH, "r", encoding="utf-8") as f:
                pmid_cache = json.load(f)

            documents = [
                {
                    "filename": filename,
                    **details
                }
                for filename, details in pmid_cache.items()
            ]

            return documents
        else:
            # Если файла нет, просто возвращаем пустой список
            return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/file/delete_document_and_cache")
async def delete_document_and_cache(pdf_filename: str, pmid: str):
    try:
        # Initialize vector store client
        vector_store_client = VectorStoreClient(service_url="http://localhost:8002")

        storage_path = os.getenv("VECTOR_STORE_PATH")
        # Вызов существующей функции для полной очистки
        delete_document_and_all_related_data(
            pdf_filename=pdf_filename,
            pmid=pmid,
            storage_path=storage_path,
            cache_root=str(CACHE_DIR),
            pmid_cache_path=str(PMID_CACHE_PATH),
            vector_store_client=vector_store_client
        )
        return {"message": f"Документ {pdf_filename} и связанные данные успешно удалены."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
