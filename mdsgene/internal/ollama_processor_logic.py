import os
import sys
import json
import requests
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from mdsgene.processor import Processor
from mdsgene.internal.pdf_text_extractor_logic import PdfTextExtractorLogic

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


class OllamaProcessorLogic(Processor):
    """Processor logic for interacting with the Ollama API using PDF context."""

    def __init__(
        self,
        pdf_filepath: Optional[Path] = None,
        *,
        pdf_uri: Optional[str] = None,
        api_url: str = OLLAMA_API_URL,
        model_name: str = OLLAMA_MODEL,
    ) -> None:
        if not pdf_filepath and not pdf_uri:
            raise ValueError("Either pdf_filepath or pdf_uri must be provided")
        self.api_url = api_url
        self.model_name = model_name
        self.pdf_filepath = Path(pdf_filepath) if pdf_filepath else None
        self.pdf_uri = pdf_uri
        self.pdf_text: str = ""
        self._load_pdf()
        print(
            f"OllamaProcessorLogic initialized with model '{self.model_name}' at '{self.api_url}'."
        )

    def _download_pdf_from_uri(self, uri: str) -> Optional[Path]:
        try:
            response = requests.get(uri, timeout=30)
            response.raise_for_status()
            tmp_path = Path("/tmp") / Path(uri).name
            tmp_path.write_bytes(response.content)
            return tmp_path
        except Exception as e:
            print(f"ERROR downloading PDF from {uri}: {e}", file=sys.stderr)
            return None

    def _load_pdf(self) -> None:
        if self.pdf_uri and not self.pdf_filepath:
            downloaded = self._download_pdf_from_uri(self.pdf_uri)
            if downloaded:
                self.pdf_filepath = downloaded
        if self.pdf_filepath:
            extractor = PdfTextExtractorLogic()
            text = extractor.extract_text(str(self.pdf_filepath))
            if text:
                self.pdf_text = text

    def _make_ollama_request(self, prompt: str, task: str) -> Optional[str]:
        payload = {"model": self.model_name, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data.get("response") or data.get("generated_text") or data.get("text")
        except Exception as e:
            print(f"ERROR calling Ollama for {task}: {e}", file=sys.stderr)
            return None

    def answer_question(self, question_text: str) -> Optional[Tuple[str, str]]:
        context = (
            f"{self.pdf_text}\n\n"
            f"Question: {question_text}\n"
            "Answer succinctly based only on the text."
        )
        answer = self._make_ollama_request(context, "question answering")
        return (answer, context) if answer is not None else None

    def get_patient_identifiers(self) -> List[Dict[str, Optional[str]]]:
        prompt = (
            f"{self.pdf_text}\n\nExtract a JSON array of objects with 'patient' and 'family' keys for all patient identifiers mentioned."
        )
        json_text = self._make_ollama_request(prompt, "patient identification")
        if not json_text:
            return []
        try:
            return json.loads(json_text)
        except Exception:
            return []

    def extract_publication_details(self) -> Dict[str, Optional[str]]:
        prompt = (
            f"{self.pdf_text}\n\nProvide a JSON object with keys 'title', 'first_author_lastname' and 'year' describing the publication."
        )
        json_text = self._make_ollama_request(prompt, "publication detail extraction")
        details = {"title": None, "first_author_lastname": None, "year": None}
        if not json_text:
            return details
        try:
            data = json.loads(json_text)
            details.update({
                "title": data.get("title"),
                "first_author_lastname": data.get("first_author_lastname"),
                "year": data.get("year"),
            })
            return details
        except Exception:
            return details
