import os
from dotenv import load_dotenv  # для загрузки переменных окружения из .env
# Загружаем переменные из .env
load_dotenv()
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict
import uuid
from pathlib import Path
import shutil
import logging
import pandas as pd
from pydantic import BaseModel

import gemini_processor  # реальная интеграция с модулем обработки
from gemini_processor import GeminiProcessor
from excel_mapping_app import ExcelMappingApp

# Загружаем настройки из окружения
FOLDER_PATH = Path(os.environ.get("FOLDER_PATH", "../.pdf_docs"))
UPLOAD_DIR = FOLDER_PATH
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL")
OUTPUT_EXCEL_PATH = Path(os.environ.get("OUTPUT_EXCEL_PATH", "output.xlsx"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- В памяти (для продакшена замените на БД) ---
documents: Dict[str, Dict] = {}

def _auto_load_documents():
    """Автоматически загружает все PDF из папки FOLDER_PATH при старте сервера. ID = pmid (через pmid_extractor) если найдено, иначе stem."""
    try:
        from excel_mapping_app import load_pmid_cache
    except ImportError:
        def load_pmid_cache():
            return {}
    try:
        from pmid_extractor import PmidExtractor
    except ImportError:
        PmidExtractor = None
    pmid_cache = load_pmid_cache()
    print(f"[DEBUG] FOLDER_PATH: {FOLDER_PATH.resolve()}")
    print(f"[DEBUG] Files in FOLDER_PATH: {[f for f in FOLDER_PATH.glob('*.pdf')]}")
    for pdf_file in FOLDER_PATH.glob("*.pdf"):
        pdf_name = pdf_file.name
        pmid = None
        # 1. Попытаться взять pmid из pmid_cache
        pmid_entry = pmid_cache.get(pdf_name, {})
        if "pmid" in pmid_entry:
            pmid = pmid_entry["pmid"]
        # 2. Если нет pmid, но есть title/author/year — воспользоваться pmid_extractor
        elif PmidExtractor and all(x in pmid_entry for x in ("title", "first_author_lastname", "year")):
            pmid = PmidExtractor.get_pmid(
                pmid_entry["title"],
                pmid_entry["first_author_lastname"],
                pmid_entry["year"]
            )
        doc_id = pmid if pmid else pdf_file.stem
        documents[doc_id] = {
            "id": doc_id,
            "filename": pdf_file.name,
            "status": "uploaded",
            "pdf_path": str(pdf_file),
            "steps": []
        }
        logger.info(f"Auto-loaded {doc_id}: {pdf_file.name}")
    print(f"[DEBUG] Documents after autoload: {documents}")

_auto_load_documents()

class AnalyzeRequest(BaseModel):
    document_id: str

app = FastAPI()

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """
    Загружает PDF и регистрирует документ.
    """
    result = []
    for file in files:
        doc_id = str(uuid.uuid4())
        dest = UPLOAD_DIR / f"{doc_id}_{file.filename}"
        with open(dest, "wb") as f_out:
            shutil.copyfileobj(file.file, f_out)

        documents[doc_id] = {
            "id": doc_id,
            "filename": file.filename,
            "status": "uploaded",
            "pdf_path": str(dest),
            "steps": []
        }
        result.append({"id": doc_id, "filename": file.filename})
        logger.info(f"Uploaded {doc_id}: {file.filename}")
    return result

@app.get("/status/{document_id}")
def get_status(document_id: str):
    doc = documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc

@app.get("/status")
def get_all_statuses():
    return list(documents.values())

@app.get("/pdf/{document_id}")
def get_pdf(document_id: str):
    doc = documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return FileResponse(doc["pdf_path"], media_type="application/pdf", filename=doc["filename"])

@app.post("/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Запускает фоновую обработку документов.
    """
    doc = documents.get(req.document_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    if doc["status"] != "uploaded":
        raise HTTPException(400, "Already processing or done")

    documents[req.document_id]["status"] = "processing"
    background_tasks.add_task(process_document, req.document_id)
    return {"document_id": req.document_id, "status": "processing"}

def process_document(document_id: str):
    """Фоновая обработка: извлечение пациентов, вопросов и ответов."""
    doc = documents[document_id]
    pdf_path = Path(doc["pdf_path"])
    logger.info(f"[PROCESS] Start {document_id}")

    # Передаём GEMINI_API_KEY и модель в процессор
    processor = GeminiProcessor(pdf_filepath=pdf_path, model_name=GEMINI_MODEL, api_key=GEMINI_API_KEY)

    # Инициализируем ExcelMappingApp с передачей documents и document_id
    app_inst = ExcelMappingApp(
        pdf_filepath=pdf_path, 
        model_name=GEMINI_MODEL,
        document_id=document_id,
        documents=documents
    )

    app_inst.gemini_processor = processor

    # Инициализируем пустой список steps перед запуском
    documents[document_id]["steps"] = []

    # Запускаем обработку - теперь steps будет обновляться во время выполнения
    app_inst.run()

    # Проверяем, есть ли вопросы для обработки
    questions = app_inst.create_mapping_data()
    if not questions:
        documents[document_id]["status"] = "no_questions_left"
        logger.info(f"[PROCESS] No more questions left for {document_id}")
        return

    # Устанавливаем статус "done" после завершения
    documents[document_id]["status"] = "done"
    logger.info(f"[PROCESS] Done {document_id}")

@app.get("/excel")
def get_excel():
    """Генерация и отдача Excel-файла с результатами."""
    rows = []
    for doc_id, doc in documents.items():
        if doc["status"] != "done":
            continue
        for step in doc["steps"]:
            rows.append({
                "doc_id": doc_id,
                "filename": doc["filename"],
                **{k: step[k] for k in ["patient_index","question_index","field","question","answer"]}
            })
    if not rows:
        raise HTTPException(404, "No results")

    df = pd.DataFrame(rows)
    OUTPUT_EXCEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_EXCEL_PATH, index=False)
    return FileResponse(str(OUTPUT_EXCEL_PATH), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=OUTPUT_EXCEL_PATH.name)
