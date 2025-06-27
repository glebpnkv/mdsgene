import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict

import requests

from mdsgene.cache_utils import save_formatted_result

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")


class OllamaTextProcessor:
    """Handles general text formatting using the Ollama API."""

    def __init__(self, api_url: str = OLLAMA_API_URL, model_name: str = OLLAMA_MODEL, *, pmid: Optional[str] = None) -> None:
        self.api_url = api_url
        self.model_name = model_name
        self.pmid = pmid
        cache_dir = Path("cache") / pmid if pmid else Path("cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._format_cache_path = cache_dir / "formatted_answer_cache.json"
        self._format_cache: Dict[str, Dict[str, str]] = {}
        print(
            f"OllamaTextProcessor initialized with model '{self.model_name}' at '{self.api_url}'."
        )

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

    def format_answer(self, raw_answer: Optional[str], strategy: str) -> Tuple[str, Optional[str]]:
        """Format a raw answer according to a strategy using Ollama."""
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "n/a", "information not found", "not specified", "not reported", "unknown"]:
            return "-99", None
        formatting_prompt = f"""
Given the following raw text extracted from a document and a desired formatting strategy, reformat the text.

Raw Text:
{raw_answer}

Formatting Strategy / Expected Output:
{strategy}

Provide only the formatted value without explanation.
"""
        formatted_answer = self._make_ollama_request(formatting_prompt, "formatting")
        if formatted_answer is None or not formatted_answer.strip():
            return "-99", raw_answer
        formatted_answer = formatted_answer.strip().strip('"').strip("'")
        if self.pmid:
            try:
                parsed_json = json.loads(formatted_answer)
            except Exception:
                parsed_json = None
            if isinstance(parsed_json, dict):
               save_formatted_result(self.pmid, formatting_prompt, raw_answer, strategy, parsed_json, "ollama")
        return formatted_answer, raw_answer
