# langgraph_vertex_agent.py - LangGraph agents made with Vertex AI API
import asyncio
import json
import logging
import tempfile
import uuid

import google
import pandas as pd
import pymupdf
from google import genai
from google.cloud import storage
from google.genai.types import GenerateContentConfig, HttpOptions, Part
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from mdsgene.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    project_id: str
    region: str = "europe-west4"
    model_name: str = "gemini-2.5-flash"


class AgentState(BaseModel):
    pdf_uri: str

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

async def acall_gemini(prompt: str, response_schema: dict, pdf_uri: str):
    """Call Gemini API asynchronously."""
    pdf_part = Part.from_uri(file_uri=pdf_uri, mime_type="application/pdf")

    resp = await CLIENT.aio.models.generate_content(
        model=settings.model_name,
        contents=[pdf_part, prompt],
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

def download_and_load_pdf(bucket_uri: str):
    """Download PDF from Google Cloud Storage and load it into PyMuPDF."""
    client = storage.Client()
    bucket_name, blob_name = bucket_uri.replace('gs://', '').split('/', 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    blob.download_to_filename(tmp_file.name)

    doc = pymupdf.open(tmp_file.name)
    first_page_text = doc.load_page(0).get_text()
    full_text = ''.join(page.get_text() for page in doc)
    doc.close()

    return first_page_text, full_text

def extract_publication_details(state: AgentState):
    """Extract publication details from the PDF using Vertex AI API structured output."""

    # Preparing input from the PDF file
    pdf_part = Part.from_uri(
        file_uri=state.pdf_uri,
        mime_type="application/pdf",
    )

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
        contents=[pdf_part, f"{extraction_prompt}"],
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
    # Preparing input from the PDF file
    pdf_part = Part.from_uri(
        file_uri=state.pdf_uri,
        mime_type="application/pdf",
    )

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

    # _, full_text = download_and_load_pdf(state.pdf_uri)

    response = CLIENT.models.generate_content(
        model=settings.model_name,
        # contents=f"{extraction_prompt}:\n{full_text}",
        contents=[pdf_part, f"{extraction_prompt}"],
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
):
    """Answer questions about patients using the Vertex AI API structured output."""
    # We cannot do anything if we have no patients to work with
    if len(state.patients) == 0:
        logger.info(f"No patients were registered for the paper: {state.pdf_uri}")
        return state

    patient_tasks = []
    # Iterating over patients
    for cur_patient in state.patients:
        cur_patient_id = cur_patient['patient_id']
        state.patient_info[cur_patient_id] = {}

        for i, cur_idx in enumerate(range(0, len(questions), batch_size_questions)):
            cur_questions = questions[cur_idx:cur_idx + batch_size_questions]
            cur_fields = [x['field'] for x in cur_questions]

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
                f"Questions are presented as a dictionary where a key is the field where you need to put the answer, and "
                f"the value is the question. For example 'gene1': 'What is the first gene with a pathogenic variant "
                f"found in the patient?' means you should put the answer to the question 'What is the first gene with a "
                f"pathogenic variant found in the patient?' about the patient {cur_patient['patient']} into the "
                f"'answer' key 'gene1'; please also provide the context where you found the answer to the question into "
                f"the 'context' key: \n\n"
                f"{''.join([f"{q['field']}: {q['question']} \n" for q in cur_questions])} \n"
                f"If the information is not found in the document, use a null value for the answer key of that question, "
                f"and put 'Information not found' into context."
            )

            coro = acall_gemini(extraction_prompt, response_schema, state.pdf_uri)
            task = asyncio.create_task(coro)
            patient_tasks.append({"patient_id": cur_patient_id, "batch": i, "task": task})

    # Gathering all results
    results = await asyncio.gather(
        *(entry["task"] for entry in patient_tasks),
        return_exceptions=True
    )

    for entry, result in zip(patient_tasks, results):
        cur_patient_id = entry["patient_id"]
        cur_fields = result[0]
        for k, v in cur_fields.items():
            state.patient_info[cur_patient_id][k] = v[0]

    return state

def answer_questions(
    state: AgentState,
    questions: list[str],
    batch_size_questions: int = 25,
):
    return asyncio.run(aanswer_questions(state, questions, batch_size_questions))

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

    return df_out
