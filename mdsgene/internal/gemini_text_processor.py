import os
import sys
import re
import time
import traceback
from typing import Optional, Tuple, List, Any

# Use google.generativeai for Gemini interaction
from google.api_core import exceptions as google_exceptions
from google.genai.types import Part
from google import genai

# --- Configuration (Copied and adapted from GeminiProcessorLogic) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL") # Or "gemini-1.5-pro", etc.
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
DEFAULT_GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.1, # Lower temperature for deterministic formatting
)

class GeminiTextProcessor:
    """Handles general text interactions with the Gemini API (e.g., formatting)."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = DEFAULT_GEMINI_MODEL):
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
            print(f"GeminiTextProcessor client initialized for model '{model_name}'.")
        except Exception as e:
            print(f"Error initializing GeminiTextProcessor client: {e}", file=sys.stderr)
            raise

    def _make_gemini_request(self, prompt_parts: List[Any], task_description: str) -> Optional[str]:
        """Makes a request to the Gemini API using the client with retry logic (Text only)."""
        # Ensure all parts are text for this processor
        contents = [Part.from_text(text=p) if isinstance(p, str) else p for p in prompt_parts]

        max_retries = 3
        delay = 5 # seconds

        for attempt in range(max_retries):
            try:
                print(f"  Attempting Gemini request ({task_description})... (Attempt {attempt + 1}/{max_retries})")
                response = self.client.models.generate_content(
                    model=f"models/{self.model_name}",
                    contents=contents,
                    # safety_settings=self.safety_settings, # Consider if needed for formatting
                    # generation_config=self.generation_config
                )

                if hasattr(response, 'text'):
                     print(f"  Gemini request successful ({task_description}).")
                     return response.text.strip()
                else:
                     # Handle cases like blocked prompts, etc.
                     feedback = getattr(response, 'prompt_feedback', 'No feedback available')
                     print(f"  WARNING: Gemini response for '{task_description}' has no text. Feedback: {feedback}", file=sys.stderr)
                     # Check for blocked content
                     if hasattr(feedback, 'block_reason') and feedback.block_reason:
                         print(f"  REASON: Blocked due to {feedback.block_reason}")
                     return None # Or raise an error, depending on desired behavior

            except google_exceptions.ResourceExhausted as e:
                print(f"  WARNING: Gemini API quota exceeded: {e}. Retrying in {delay}s...", file=sys.stderr)
                time.sleep(delay)
                delay *= 2
            except google_exceptions.ServiceUnavailable as e:
                 print(f"  WARNING: Gemini service unavailable: {e}. Retrying in {delay}s...", file=sys.stderr)
                 time.sleep(delay)
                 delay *= 2
            except google_exceptions.InternalServerError as e:
                 print(f"  WARNING: Gemini internal server error: {e}. Retrying in {delay}s...", file=sys.stderr)
                 time.sleep(delay)
                 delay *= 2
            except google_exceptions.InvalidArgument as e:
                 print(f"  ERROR: Invalid argument for Gemini API call ({task_description}): {e}", file=sys.stderr)
                 print(f"  Check if model name '{self.model_name}' is valid and correctly formatted ('models/...') for the client API.")
                 return None
            except Exception as e:
                print(f"  ERROR: Unhandled exception during Gemini API call ({task_description}): {e}", file=sys.stderr)
                traceback.print_exc()
                # Maybe retry once more for generic errors? Or just fail.
                if attempt < max_retries - 1:
                    print(f"Retrying generic error in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    return None # Fail after retries

        print(f"  ERROR: Gemini request failed after {max_retries} attempts ({task_description}).", file=sys.stderr)
        return None


    def format_answer(self, raw_answer: Optional[str], strategy: str) -> Tuple[str, Optional[str]]:
        """
        Uses Gemini to format a raw answer according to a specific strategy.
        (Code exactly as in GeminiProcessorLogic.format_answer, including the prompt)
        """
        print("  Formatting answer using GeminiTextProcessor...")
        if not raw_answer or raw_answer.strip().lower() in ["", "none", "n/a", "information not found", "not specified", "not reported", "unknown"]:
            print(f"  Raw answer indicates missing info ('{raw_answer}'). Returning -99.")
            return "-99", None
        if "error" in raw_answer.lower() or "failed" in raw_answer.lower():
             print(f"  Raw answer indicates error ('{raw_answer[:50]}...'). Returning -99.")
             return "-99", None
        if "don't know" in raw_answer.lower() or "couldn't find" in raw_answer.lower() or "not stated" in raw_answer.lower() or "not mentioned" in raw_answer.lower():
             print(f"  Raw answer indicates info not found ('{raw_answer[:50]}...'). Returning -99.")
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
2.  Numeric answers (age, counts): Extract ONLY the number. If unknown/not given/not found in raw text, output -99. If explicitly zero/none, output 0.
3.  Yes/no questions: Output ONLY lowercase "yes" or "no". If unknown/not mentioned/unclear in raw text, output -99.
4.  Sex: Output ONLY uppercase "M" or "F". If unknown/not reported, output -99.
5.  Zygosity: Output ONLY "heterozygous", "homozygous", or "hemizygous". If unknown/not applicable, output -99.
6.  Inclusion decisions (IN/EX): Output ONLY uppercase "IN" or "EX". If unclear, output -99.
7.  HGNC Gene Symbols: Output the official symbol if found (e.g., "PARK2"). If multiple possibilities, list them comma-separated unless strategy specifies otherwise. If none found or not applicable, output -99.
8.  Mutation Notation (cDNA/protein): Output the notation as described in the strategy (e.g., c.511C>T, p.Gln171*). Standardize if possible, but preserve original if instructed. If none found/not applicable, output -99.
9.  General Text/Comments: Extract the relevant information as described. If none found, output -99.
10. IMPORTANT: Output ONLY the final formatted value. Do NOT include explanations, apologies, or any text like "Formatted answer:", "Based on the text:", "The value is:", etc. JUST the value or -99.
11. If the Raw Text explicitly states the information is missing (e.g., "not reported", "unknown", "N/A"), output -99.

Formatted Value:"""

        formatted_answer = self._make_gemini_request([formatting_prompt], "formatting")

        if formatted_answer is None:
            print(f"  ERROR: Gemini formatting failed for raw answer: '{raw_answer[:50]}...'. Returning -99.", file=sys.stderr)
            # Return the original raw_answer as context even on failure
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

        if not formatted_answer or formatted_answer.lower() in ["unknown", "not stated", "not reported", "n/a", "none", "not applicable", "not mentioned", "-", "null"]:
             print(f"  Formatted answer resulted in an 'unknown' value ('{formatted_answer}'). Returning -99.")
             # Return the original raw_answer as context
             return "-99", raw_answer

        print(f"  Formatted result: '{formatted_answer}'")
        # Return the formatted answer and the original raw_answer as context
        return formatted_answer, raw_answer