import json
import logging
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

# Use google.generativeai for Gemini interaction
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai.types import Part

from mdsgene.cache_utils import save_formatted_result
from mdsgene.internal.defines import NO_INFORMATION_LIST, DEFAULT_SAFETY_SETTINGS
from mdsgene.logging_config import configure_logging

# Get a logger for this module
configure_logging()
logger = logging.getLogger(__name__)


# --- Configuration (Copied and adapted from GeminiProcessorLogic) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL") # Or "gemini-1.5-pro", etc.
DEFAULT_GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.0, # Lower temperature for deterministic formatting
)


class GeminiTextProcessor:
    """Handles general text interactions with the Gemini API (e.g., formatting)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        pmid: str | None = None
    ):
        """
        Initializes the Gemini client for text operations.

        Args:
            api_key: Google AI API key. If None, uses GEMINI_API_KEY env var.
            model_name: The specific Gemini model to use.
        """
        resolved_api_key = api_key or GEMINI_API_KEY
        if not resolved_api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY environment variable not set.")

        try:
            self.client = genai.Client(api_key=resolved_api_key)
            self.model_name = model_name
            self.safety_settings = DEFAULT_SAFETY_SETTINGS
            self.generation_config = DEFAULT_GENERATION_CONFIG
            logger.info(f"GeminiTextProcessor client initialized for model '{model_name}'.")

            self.pmid = pmid
            cache_dir = Path("cache") / pmid if pmid else Path("cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._format_cache_path = cache_dir / "formatted_answer_cache.json"
            self._format_cache: dict[str, dict[str, str]] = {}

        except Exception as e:
            logger.error(f"Error initializing GeminiTextProcessor client: {e}")
            raise

    def load_from_cache(self, cache_id: str) -> dict[str, str] | None:
        """Load a cached formatted answer by id."""
        return self._format_cache.get(cache_id)

    def save_to_cache(self, cache_id: str, raw_answer: str, strategy: str) -> None:
        """Save a formatted answer to cache."""
        self._format_cache[cache_id] = {"raw_answer": raw_answer, "strategy": strategy}
        try:
            with open(self._format_cache_path, "w", encoding="utf-8") as f:
                json.dump(self._format_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving format cache: {e}")

    def _make_gemini_request(self, prompt_parts: list[Any], task_description: str) -> str | None:
        """Makes a request to the Gemini API using the client with retry logic (Text only)."""
        # Ensure all parts are text for this processor
        contents = [Part.from_text(text=p) if isinstance(p, str) else p for p in prompt_parts]

        max_retries = 3
        delay = 5 # seconds

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting Gemini request ({task_description})... (Attempt {attempt + 1}/{max_retries})")
                response = self.client.models.generate_content(
                    model=f"models/{self.model_name}",
                    contents=contents,
                    # safety_settings=self.safety_settings, # Consider if needed for formatting
                    # generation_config=self.generation_config
                )

                if hasattr(response, 'text'):
                    logger.info(f"Gemini request successful ({task_description}).")
                    return response.text.strip()
                else:
                     # Handle cases like blocked prompts, etc.
                     feedback = getattr(response, 'prompt_feedback', 'No feedback available')
                     logger.warning(f"Gemini response for '{task_description}' has no text. Feedback: {feedback}")
                     # Check for blocked content
                     if hasattr(feedback, 'block_reason') and feedback.block_reason:
                         logger.warning(f"REASON: Blocked due to {feedback.block_reason}")
                     return None # Or raise an error, depending on desired behavior

            except google_exceptions.ResourceExhausted as e:
                logger.warning(f"Gemini API quota exceeded: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except google_exceptions.ServiceUnavailable as e:
                logger.warning(f"Gemini service unavailable: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except google_exceptions.InternalServerError as e:
                logger.warning(f"Gemini internal server error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except google_exceptions.InvalidArgument as e:
                logger.error(f"Invalid argument for Gemini API call ({task_description}): {e}")
                logger.error(
                    f"Check if model name '{self.model_name}' is valid and correctly formatted ('models/...') "
                                     f"for the client API."
                 )
                return None
            except Exception as e:
                logger.error(f"Unhandled exception during Gemini API call ({task_description}): {e}")
                traceback.print_exc()
                # Maybe retry once more for generic errors? Or just fail.
                if attempt < max_retries - 1:
                    logger.info(f"Retrying generic error in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    return None # Fail after retries

        logger.error(f"Gemini request failed after {max_retries} attempts ({task_description}).")
        return None

    def format_answer(self, raw_answer: str | None, strategy: str) -> tuple[str, str | None]:
        """
        Uses Gemini to format a raw answer according to a specific strategy.
        (Code exactly as in GeminiProcessorLogic.format_answer, including the prompt)
        """
        logger.info("Formatting answer using GeminiTextProcessor...")
        if not raw_answer or raw_answer.lower() in NO_INFORMATION_LIST:
            logger.info(f"Raw answer indicates missing info ('{raw_answer}'). Returning -99.")
            return "-99", None
        if "error" in raw_answer.lower() or "failed" in raw_answer.lower():
            logger.info(f"Raw answer indicates error ('{raw_answer[:50]}...'). Returning -99.")
            return "-99", None

        formatting_prompt = f"""
        Given the following raw text extracted from a document and a desired formatting strategy, reformat the text.
        
        Raw Text:
        \"\"\"
        {raw_answer}
        \"\"\"

        Formatting Strategy / Expected Output:
        \"\"\"
        {strategy}
        \"\"\"
        
        Formatting Rules:
        1.  Follow the strategy precisely.
        2.  Numeric answers (age, counts): Extract ONLY the number. If unknown/not given/not found in raw text, 
        output -99. If explicitly zero/none, output 0.
        3.  Yes/no questions: Output ONLY lowercase "yes" or "no". If unknown/not mentioned/unclear in raw text, 
        output -99.
        4.  Sex: Output ONLY uppercase "M" or "F". If unknown/not reported, output -99.
        5.  Zygosity: Output ONLY "heterozygous", "homozygous", or "hemizygous". If unknown/not applicable, output -99.
        6.  Inclusion decisions (IN/EX): Output ONLY uppercase "IN" or "EX". If unclear, output -99.
        7.  HGNC Gene Symbols: Output the official symbol if found (e.g., "PARK2"). If multiple possibilities, 
        list them comma-separated unless strategy specifies otherwise. If none found or not applicable, output -99.
        8.  Mutation Notation (cDNA/protein): Output the notation as described in the strategy 
        (e.g., c.511C>T, p.Gln171*). Standardize if possible, but preserve original if instructed. 
        If none found/not applicable, output -99.
        9.  General Text/Comments: Extract the relevant information as described. If none found, output -99.
        10. IMPORTANT: Output ONLY the final formatted value. Do NOT include explanations, apologies, or 
        any text like "Formatted answer:", "Based on the text:", "The value is:", etc. JUST the value or -99.
        11. If the Raw Text explicitly states the information is missing (e.g., "not reported", "unknown", "N/A"), 
        output -99.
        
        Formatted Value:"""

        formatted_answer = self._make_gemini_request([formatting_prompt], "formatting")

        if formatted_answer is None:
            logger.error(f"Gemini formatting failed for raw answer: '{raw_answer[:50]}...'. Returning -99.")
            # Return the original raw_answer as context even on failure
            return "-99", raw_answer

        if not formatted_answer or not formatted_answer.strip():
            logger.error("Empty string returned from Gemini.")
            return "-99", raw_answer

        # Post-processing cleanup (same as before)
        formatted_answer = formatted_answer.strip()
        if formatted_answer.startswith('"') and formatted_answer.endswith('"') and len(formatted_answer) > 1:
            formatted_answer = formatted_answer[1:-1].strip()
        elif formatted_answer.startswith("'") and formatted_answer.endswith("'") and len(formatted_answer) > 1:
            formatted_answer = formatted_answer[1:-1].strip()

        preambles = ["formatted value:", "formatted answer:", "the formatted value is:", "output:"]
        for preamble in preambles:
            if formatted_answer.lower().startswith(preamble):
                formatted_answer = formatted_answer[len(preamble):].strip()
                break

        if not formatted_answer or formatted_answer.lower() in NO_INFORMATION_LIST:
            logger.info(f"Formatted answer resulted in an 'unknown' value ('{formatted_answer}'). Returning -99.")
            # Return the original raw_answer as context
            return "-99", raw_answer

        if self.pmid:
            text_for_parse = formatted_answer.strip()
            if text_for_parse.startswith("```"):
                text_for_parse = re.sub(r"^```(?:json)?\s*", "", text_for_parse)
                text_for_parse = re.sub(r"\s*```$", "", text_for_parse).strip()
            try:
                parsed_json = json.loads(text_for_parse)
            except Exception:
                parsed_json = None

            if isinstance(parsed_json, dict):
                save_formatted_result(
                    self.pmid,
                    formatting_prompt,
                    raw_answer,
                    strategy,
                    parsed_json,
                )
            else:
                logger.info("[Cache] Formatted result is not JSON. Skipping cache save.")

        logger.info(f"Formatted result: '{formatted_answer}'")

        # Return the formatted answer and the original raw_answer as context
        return formatted_answer, raw_answer
