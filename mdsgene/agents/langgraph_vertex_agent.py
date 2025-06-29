# langgraph_vertex_agent.py - LangGraph agents made with Vertex AI API
import asyncio
import io
import json
import logging
import threading
import uuid

import google
import pandas as pd
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, HttpOptions, Part
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pypdf import PdfReader, PdfWriter

from mdsgene.logging_config import configure_logging

configure_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_CONCURRENT_REQUESTS = 50

class Settings(BaseSettings):
    project_id: str
    region: str = "europe-west4"
    model_name: str = "gemini-2.5-flash"
    max_bytes_per_file: int = int(50 * 1024 * 1024)


class AgentState(BaseModel):
    pdf_uri: str
    pdf_file_paths: list[str] = Field(default_factory=list)
    pdf_parts: list[Part] = Field(default_factory=list)

    title: str = None
    first_author_lastname: str = None
    publication_year: int = None

    pmid: str = None
    patients: list[dict] = Field(default_factory=list)
    patient_info: dict = Field(default_factory=dict)


# Singleton VertexAI client
_, project_id = google.auth.default()
settings = Settings(project_id=project_id)

CLIENT = genai.Client(
    vertexai=True,
    project=settings.project_id,
    location=settings.region,
    http_options=HttpOptions(api_version="v1")
)

async def acall_gemini(prompt: str, response_schema: dict, pdf_parts: list[Part]):
    """Call Gemini API asynchronously."""
    resp = await CLIENT.aio.models.generate_content(
        model=settings.model_name,
        contents=pdf_parts + [prompt],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            candidate_count=1,
            temperature=0.0,
        ),
    )

    try:
        json_text = resp.text
        output = json.loads(json_text)
        return output
    except Exception as e:
        logger.exception(f"Exception: {e}")
        return None


def split_pdf_into_parts(bucket_name: str, blob_name: str) -> list[Part]:
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    data = blob.download_as_bytes()

    # If it’s already small enough, wrap and return
    if len(data) <= settings.max_bytes_per_file:
        return [Part.from_bytes(data=data, mime_type="application/pdf")]

    reader = PdfReader(io.BytesIO(data))
    total_pages = len(reader.pages)

    # Decide on a pages-per-chunk so that roughly each chunk is ≤ MAX_BYTES
    # (this is heuristic — needs extra work for pages that are wildly different sizes)
    chunks = (len(data) // settings.max_bytes_per_file) + 1
    pages_per_chunk = max(1, total_pages // chunks)

    parts: list[Part] = []
    for start in range(0, total_pages, pages_per_chunk):
        writer = PdfWriter()
        for p in reader.pages[start : start + pages_per_chunk]:
            writer.add_page(p)
        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)
        parts.append(Part.from_bytes(data=buf.read(), mime_type="application/pdf"))
    return parts


def process_pdf(state: AgentState):
    """Discover all the pdf files in the pdf folder of the state, and pre-process them."""

    client = storage.Client()
    bucket_name, folder_name = state.pdf_uri.replace('gs://', '').split('/', 1)
    blobs = client.list_blobs(bucket_name, prefix=f"{folder_name}")
    pdf_files = [
        f"gs://{bucket_name}/{b.name}"
        for b in blobs
        if b.name.lower().endswith('.pdf')
    ]

    # We cannot do anything if no PDF files are found
    if len(pdf_files) == 0:
        return state

    state.pdf_file_paths = pdf_files
    state.pdf_parts = []
    for uri in state.pdf_file_paths:
        _, path = uri.replace("gs://", "").split("/", 1)
        # split into one or more Parts, each ≤ 50 MB
        parts = split_pdf_into_parts(bucket_name, path)
        state.pdf_parts.extend(parts)

    return state


def extract_publication_details(state: AgentState):
    """Extract publication details from the PDF using Vertex AI API structured output."""
    # We cannot do anything if there are no PDFs to work with
    if len(state.pdf_parts) == 0:
        logger.info(f"No PDFs were registered for the paper: {state.pdf_uri}")
        return state

    # Defining the schema of the response
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "title": {"type": "STRING"},
            "first_author_lastname": {"type": "STRING"},
            "year": {"type": "INTEGER"},
        },
        "required": ["title", "first_author_lastname", "year"],
    }

    # Extraction prompt with instructions
    extraction_prompt = (
        "Based *only* on the provided document, identify the following publication details:\n"
        "1. The full title of the article.\n"
        "2. The last name of the *first* author listed.\n"
        "3. The four-digit publication year.\n\n"
        "Example: {\"title\": \"Actual Title of Paper\", \"first_author_lastname\": \"Smith\", \"year\": \"2023\"}\n"
        "If any detail cannot be found, use a null value for that key."
    )

    response = CLIENT.models.generate_content(
        model=settings.model_name,
        contents=state.pdf_parts + [f"{extraction_prompt}"],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
            candidate_count=1
        ),
    )

    json_text = response.text

    try:
        output_dict = json.loads(json_text)

        state.title = output_dict.get("title")
        state.first_author_lastname = output_dict.get("first_author_lastname")
        state.publication_year = output_dict.get("year")
    except Exception as e:
        logger.exception(f"Exception: {e}")

    return state


def extract_patients(state: AgentState):
    # We cannot do anything if there are no PDFs to work with
    if len(state.pdf_parts) == 0:
        logger.info(f"No PDFs were registered for the paper: {state.pdf_uri}")
        return state

    # Defining the schema of the response
    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "patient": {"type": "STRING"},
                "family": {"type": "STRING"},
                "context": {"type": "STRING"},
            },
            "required": ["patient", "family", "context"],
        }
    }

    # Extraction prompt with instructions
    extraction_prompt = (
        "Based on the provided document, extract a list of all distinct patient cases mentioned. "
        "For each case, identify the specific patient identifier (e.g., 'Patient 1', 'II-1', 'P3'), the "
        "family identifier if available (e.g., 'Family A', 'F1'), and part of the article where this case "
        "was first mentioned ('context' field); if the patient was first mentioned in the text, please return the "
        "paragraph where they were first mentioned, if they were mentioned in the graph or a table - please identify "
        "the graph or table where they were first mentioned. "
        "Example: [{'family': 'Family 1', 'patient': 'II-1', 'context': 'Patient II-1 is part of the Family 1 from "
        "Italy.'}]. "
        "If any detail cannot be found, use a null value for that key."
    )

    response = CLIENT.models.generate_content(
        model=settings.model_name,
        contents=state.pdf_parts + [f"{extraction_prompt}"],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.0,
            candidate_count=1
        ),
    )

    json_text = response.text

    try:
        patients = json.loads(json_text)

        # Adding a unique UUID to each extracted patient
        if len(patients) > 0:
            for patient in patients:
                patient['patient_id'] = str(uuid.uuid4())

        state.patients = patients
    except Exception as e:
        logger.exception(f"Exception: {e}")

    return state


async def aanswer_questions(
    state: AgentState,
    questions: list[str],
    batch_size_questions: int = 25,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,

):
    """Answer questions about patients using the Vertex AI API structured output."""
    # We cannot do anything if we have no patients to work with
    if len(state.patients) == 0:
        logger.info(f"No patients were registered for the paper: {state.pdf_uri}")
        return state

    # We cannot do anything if there are no PDFs to work with
    if len(state.pdf_parts) == 0:
        logger.info(f"No PDFs were registered for the paper: {state.pdf_uri}")
        return state

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def limited_acall_gemini(prompt: str, response_schema: dict, pdf_parts: list[Part]):
        """Wrapper for acall_gemini with semaphore limiting."""
        async with semaphore:
            return await acall_gemini(prompt, response_schema, pdf_parts)

    patient_tasks = []
    # Iterating over patients
    for cur_patient in state.patients:
        cur_patient_id = cur_patient["patient_id"]
        state.patient_info[cur_patient_id] = {}

        for i, cur_idx in enumerate(range(0, len(questions), batch_size_questions)):
            cur_questions = questions[cur_idx:cur_idx + batch_size_questions]
            cur_fields = [x["field"] for x in cur_questions]

            # Generating the response schema for the current batch of questions
            response_schema = {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        x: {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "answer": {"type": "STRING", "nullable": True},
                                    "context": {"type": "STRING", "nullable": True}
                                },
                                "required": ["answer", "context"]
                            }
                        }
                        for x in cur_fields
                    },
                    "required": cur_fields,
                }
            }

            # Generating the extraction prompt for the current patient and batch of questions
            extraction_prompt = (
                f"Based *only* on the information contained within the provided document, answer a set of questions "
                f"about the patient {cur_patient['patient']}; the patient was first identified in the following "
                f"context: {cur_patient['context']}. \n\n"
                f"Questions are presented as a dictionary where a key is the field where you need to put the answer, "
                f"and the value is the question. For example 'gene1': 'What is the first gene with a pathogenic "
                f"variant found in the patient?' means you should put the answer to the question 'What is the first "
                f"gene with a pathogenic variant found in the patient?' about the patient {cur_patient['patient']} "
                f"into the 'answer' key 'gene1'; please also provide the context where you found the answer to the "
                f"question into the 'context' key: \n\n"
                f"{''.join([f"{q['field']}: {q['question']} \n" for q in cur_questions])} \n"
                f"If the information is not found in the document, use a null value for the answer key of that "
                f"question, and put 'Information not found' into context."
            )

            coro = limited_acall_gemini(extraction_prompt, response_schema, state.pdf_parts)
            task = asyncio.create_task(coro)
            patient_tasks.append({"patient_id": cur_patient_id, "batch": i, "task": task})

    # Gathering all results
    try:
        results = await asyncio.gather(
            *(entry["task"] for entry in patient_tasks),
            return_exceptions=True
        )
    except Exception as e:
        logger.exception(f"Error in asyncio.gather: {e}")
        return state

    for entry, result in zip(patient_tasks, results):
        cur_patient_id = entry["patient_id"]

        # Handle exceptions from individual tasks
        if isinstance(result, Exception):
            logger.error(f"Task failed for patient {cur_patient_id}: {result}")
            continue

        if result is None:
            continue

        if len(result) == 0:
            continue

        cur_fields = result[0]
        for k, v in cur_fields.items():
            if len(v) > 0:
                state.patient_info[cur_patient_id][k] = v[0]

    return state


def answer_questions(
    state: AgentState,
    questions: list[str],
    batch_size_questions: int = 25,
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,

):
    def run_in_thread():
        """Run the async function in a new thread with its own event loop."""

        def thread_target():
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    aanswer_questions(state, questions, batch_size_questions, max_concurrent_requests)
                )
            finally:
                loop.close()

        # Use threading to avoid event loop conflicts
        result_container = [None]
        exception_container = [None]

        def wrapper():
            try:
                result_container[0] = thread_target()
            except Exception as e:
                exception_container[0] = e

        thread = threading.Thread(target=wrapper)
        thread.start()
        thread.join()

        if exception_container[0]:
            raise exception_container[0]

        return result_container[0]

    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()

        if loop.is_running():
            # We're in an environment with a running event loop (like Jupyter)
            # Use threading to avoid conflicts
            logger.info("Running event loop detected, using thread-based execution")
            return run_in_thread()
        else:
            # This shouldn't happen, but handle it just in case
            return asyncio.run(aanswer_questions(state, questions, batch_size_questions, max_concurrent_requests))

    except RuntimeError:
        # No event loop is running, we can use asyncio.run safely
        logger.info("No running event loop detected, using asyncio.run")
        return asyncio.run(aanswer_questions(state, questions, batch_size_questions, max_concurrent_requests))


def post_process_result(state: AgentState):
    df_out = pd.DataFrame()

    if len(state.patient_info) == 0:
        return df_out

    for k, v in state.patient_info.items():
        try:
            df_cur = pd.DataFrame.from_dict(v, orient="index")
            df_cur.index.name = 'field'
            df_cur = df_cur.reset_index()
            df_cur['patient_id'] = k

            df_out = pd.concat([df_out, df_cur], ignore_index=True)
        except Exception as e:
            print(e)
            continue

    # Adding Patient data
    df_patients = pd.DataFrame(state.patients)
    df_patients = df_patients.rename(columns={'context': 'patient_context'})

    df_out = pd.merge(df_out, df_patients, on='patient_id')

    # Adding publication details
    df_out["title"] = state.title
    df_out["first_author_lastname"] = state.first_author_lastname
    df_out["publication_year"] = state.publication_year

    return df_out
