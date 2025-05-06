# gemini_processor_logic.py
import os
import sys
import json
import re
import time
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Use google.generativeai for Gemini interaction
from google.api_core import exceptions as google_exceptions # For specific error handling
from google.genai.types import Part
from google import genai

# --- Configuration ---
# It's best practice to load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Default model, can be overridden in constructor
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL") # Or "gemini-1.5-pro", etc.
# Safety settings to allow potentially sensitive medical info extraction (adjust as needed)
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]
# Generation config (optional, adjust temperature etc. if needed)
DEFAULT_GENERATION_CONFIG = genai.types.GenerationConfig(
    # candidate_count=1, # Default is 1
    # stop_sequences=["..."],
    # max_output_tokens=8192, # Adjust if needed
    temperature=0.1, # Lower temperature for more deterministic extraction/formatting
    # top_p=0.9,
    # top_k=40
)


class GeminiProcessorLogic:
    """Handles interactions with the Gemini API for PDF processing."""

    def __init__(self, pdf_filepath: Path, api_key: Optional[str] = None, model_name: str = DEFAULT_GEMINI_MODEL):
        """
        Initializes the Gemini client and loads the PDF document.

        Args:
            pdf_filepath: Path to the input PDF file.
            api_key: Google AI API key. If None, tries to read from GEMINI_API_KEY env var.
            model_name: The specific Gemini model to use (e.g., 'gemini-1.5-flash').
        """
        resolved_api_key = api_key or GEMINI_API_KEY
        if not resolved_api_key:
            raise ValueError("Gemini API key not provided and GEMINI_API_KEY environment variable not set.")

        if not pdf_filepath.exists() or not pdf_filepath.is_file():
             raise FileNotFoundError(f"PDF file not found: {pdf_filepath}")

        try:
            # --- Use genai.Client instead of configure/GenerativeModel ---
            self.client = genai.Client(api_key=resolved_api_key)
            self.model_name = model_name
            # Store config to pass to generate_content later
            self.safety_settings = DEFAULT_SAFETY_SETTINGS
            self.generation_config = DEFAULT_GENERATION_CONFIG
            print(f"Gemini client initialized for model '{model_name}'.")
            # --- End Change ---
        except Exception as e:
            print(f"Error initializing Gemini client: {e}", file=sys.stderr)
            raise

        self.pdf_filepath = pdf_filepath
        self.pdf_parts: Optional[List[Part]] = None
        self._load_pdf() # Load PDF during initialization

    def _load_pdf(self):
        """Reads the PDF file into Gemini Part format."""
        try:
            print(f"Loading PDF: {self.pdf_filepath}")
            pdf_bytes = self.pdf_filepath.read_bytes()
            # Determine MIME type (usually application/pdf)
            mime_type = "application/pdf" # Assuming standard PDF
            self.pdf_parts = [Part.from_bytes(data=pdf_bytes, mime_type=mime_type)]
            print("PDF loaded successfully into Gemini Part.")
        except Exception as e:
            print(f"Error reading PDF file {self.pdf_filepath}: {e}", file=sys.stderr)
            self.pdf_parts = None # Ensure it's None if loading fails

    def _make_gemini_request(self, prompt_parts: List[Any], task_description: str) -> Optional[str]:
        """Makes a request to the Gemini API using the client with retry logic."""
        if not self.pdf_parts and task_description != "formatting":
             print("ERROR: PDF parts not loaded, cannot make request.", file=sys.stderr)
             return None

        if task_description == "formatting":
            contents = [Part.from_text(text=p) if isinstance(p, str) else p for p in prompt_parts]
        else:
            text_prompts = [Part.from_text(text=p) if isinstance(p, str) else p for p in prompt_parts]
            contents = text_prompts + (self.pdf_parts or [])

        max_retries = 3
        delay = 5 # seconds

        for attempt in range(max_retries):
            try:
                print(f"  Attempting Gemini request ({task_description})... (Attempt {attempt + 1}/{max_retries})")

                # --- Call using client.models.generate_content ---
                response = self.client.models.generate_content(
                    model=f"models/{self.model_name}", # Model name needs 'models/' prefix for client API
                    contents=contents,
                    # safety_settings=self.safety_settings,
                    # generation_config=self.generation_config
                )
                # --- End Change ---

                if hasattr(response, 'text'):
                     print(f"  Gemini request successful ({task_description}).")
                     return response.text.strip()
                else:
                     print(f"  WARNING: Gemini response for '{task_description}' has no text. Response details: {response.prompt_feedback}", file=sys.stderr)
                     return None

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
            # Handle potential InvalidArgument errors (e.g., bad model name)
            except google_exceptions.InvalidArgument as e:
                 print(f"  ERROR: Invalid argument for Gemini API call ({task_description}): {e}", file=sys.stderr)
                 print(f"  Check if model name '{self.model_name}' is valid and correctly formatted ('models/...') for the client API.")
                 return None # Don't retry on invalid argument
            except Exception as e:
                print(f"  ERROR: Unhandled exception during Gemini API call ({task_description}): {e}", file=sys.stderr)
                traceback.print_exc()
                return None

        print(f"  ERROR: Gemini request failed after {max_retries} attempts ({task_description}).", file=sys.stderr)
        return None


    def get_patient_identifiers(self) -> List[Dict[str, Optional[str]]]:
        """
        Uses Gemini to extract a list of patient and family identifiers from the PDF.
        """
        print("\nRequesting patient identifiers from Gemini...")
        extraction_prompt = (
            "Based on the provided document, extract a list of all distinct patient cases mentioned. "
            "For each case, identify the specific patient identifier (e.g., 'Patient 1', 'II-1', 'P3') and the family identifier if available (e.g., 'Family A', 'F1'). "
            "Present the result STRICTLY as a JSON array of objects. Each object must have two keys: 'family' (string or null) and 'patient' (string). "
            "Example: [{'family': 'Family 1', 'patient': 'II-1'}, {'family': null, 'patient': 'Patient 2'}]"
            "Output ONLY the JSON array, without any introductory text, code block markers (like ```json), or explanations."
        )

        json_text = self._make_gemini_request([extraction_prompt], "patient identification")

        if not json_text:
            print("ERROR: Failed to get response for patient identifiers.", file=sys.stderr)
            return []

        print(f"  Raw Gemini JSON response:\n---\n{json_text}\n---")

        try:
            match = re.search(r'\[\s*\{.*?\}\s*\]', json_text, re.DOTALL)
            if match:
                json_array_text = match.group(0)
            elif json_text.strip().startswith('[') and json_text.strip().endswith(']'):
                 json_array_text = json_text.strip()
            else:
                 print("ERROR: Could not find a valid JSON array structure in the response.", file=sys.stderr)
                 print(f"Received text: {json_text}")
                 return []

            patients = json.loads(json_array_text)
            if not isinstance(patients, list):
                print("ERROR: Parsed JSON is not a list.", file=sys.stderr)
                return []

            unique_patients = {}
            for entry in patients:
                if not isinstance(entry, dict) or "patient" not in entry:
                    print(f"WARNING: Skipping invalid patient entry format: {entry}", file=sys.stderr)
                    continue
                patient_id_raw = entry.get("patient")
                family_id_raw = entry.get("family")
                patient_id = str(patient_id_raw).strip() if patient_id_raw is not None else None
                family_id = str(family_id_raw).strip() if family_id_raw is not None and str(family_id_raw).strip().lower() != 'null' else None
                if not patient_id:
                    print(f"WARNING: Skipping entry with missing patient identifier: {entry}", file=sys.stderr)
                    continue
                key = (patient_id, family_id)
                if key not in unique_patients:
                    unique_patients[key] = {"patient": patient_id, "family": family_id}

            unique_patient_list = list(unique_patients.values())
            print(f"  Identified {len(unique_patient_list)} unique patient entries.")
            return unique_patient_list

        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON response for patient list: {e}", file=sys.stderr)
            print(f"  Invalid JSON text received:\n---\n{json_array_text or json_text}\n---", file=sys.stderr)
            return []
        except Exception as e:
            print(f"ERROR: Unexpected error processing patient list: {e}", file=sys.stderr)
            traceback.print_exc()
            return []

    def answer_question(self, question_text: str) -> Optional[Tuple[str, str]]:
        """
        Asks a specific question to Gemini based on the loaded PDF context.

        Returns:
            A tuple of (answer, context), or None if the question cannot be answered
        """
        prompt = (
             f"Based *only* on the information contained within the provided document, answer the following question:\n\n"
             f"{question_text}\n\n"
             f"If the information is not found in the document, state 'Information not found'."
        )
        print(f"  Asking Gemini: {question_text[:100]}...")
        raw_answer = self._make_gemini_request([prompt], "question answering")

        if raw_answer and ("not found" in raw_answer.lower() or "not stated" in raw_answer.lower() or "not mentioned" in raw_answer.lower()):
             print(f"  Gemini indicated information not found for: '{question_text[:100]}...'")

        # Return both the answer and the prompt as context
        return (raw_answer, prompt) if raw_answer is not None else None



    def extract_publication_details(self) -> Dict[str, Optional[str]]:
        """
        Uses Gemini to extract key publication details (title, author, year) from the PDF.

        Returns:
            A dictionary with keys 'title', 'first_author_lastname', 'year',
            or None values if details cannot be extracted.
        """
        print("\nRequesting publication details (title, author, year) from Gemini...")
        extraction_prompt = (
            "Based *only* on the provided document, identify the following publication details:\n"
            "1. The full title of the article.\n"
            "2. The last name of the *first* author listed.\n"
            "3. The four-digit publication year.\n\n"
            "Present the result STRICTLY as a JSON object with the keys 'title', 'first_author_lastname', and 'year'.\n"
            "Example: {\"title\": \"Actual Title of Paper\", \"first_author_lastname\": \"Smith\", \"year\": \"2023\"}\n"
            "Output ONLY the JSON object, without any introductory text, code block markers (like ```json), or explanations."
            " If any detail cannot be found, use a null value for that key."
        )

        # This request only needs the PDF context
        json_text = self._make_gemini_request([extraction_prompt], "publication detail extraction")

        details = {'title': None, 'first_author_lastname': None, 'year': None} # Default values

        if not json_text:
            print("ERROR: Failed to get response for publication details.", file=sys.stderr)
            return details # Return defaults

        print(f"  Raw Gemini JSON response for details:\n---\n{json_text}\n---")

        # Attempt to parse the JSON response
        try:
            # More robust JSON finding
            match = re.search(r'\{\s*".*?":.*?\s*\}', json_text, re.DOTALL)
            if match:
                json_object_text = match.group(0)
            elif json_text.strip().startswith('{') and json_text.strip().endswith('}'):
                 json_object_text = json_text.strip()
            else:
                 print("ERROR: Could not find a valid JSON object structure in the response for details.", file=sys.stderr)
                 print(f"Received text: {json_text}")
                 return details # Return defaults

            parsed_json = json.loads(json_object_text)
            if not isinstance(parsed_json, dict):
                print("ERROR: Parsed JSON for details is not a dictionary.", file=sys.stderr)
                return details # Return defaults

            # Extract details, handling potential missing keys or null values
            details['title'] = str(parsed_json.get('title')).strip() if parsed_json.get('title') else None
            details['first_author_lastname'] = str(parsed_json.get('first_author_lastname')).strip() if parsed_json.get('first_author_lastname') else None
            details['year'] = str(parsed_json.get('year')).strip() if parsed_json.get('year') else None

            # Basic validation
            if details['year'] and not re.match(r'^\d{4}$', details['year']):
                print(f"WARNING: Extracted year '{details['year']}' is not 4 digits. Setting to None.")
                details['year'] = None

            print(f"  Extracted publication details: {details}")
            return details

        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON response for publication details: {e}", file=sys.stderr)
            print(f"  Invalid JSON text received:\n---\n{json_object_text or json_text}\n---", file=sys.stderr)
            return details # Return defaults
        except Exception as e:
            print(f"ERROR: Unexpected error processing publication details: {e}", file=sys.stderr)
            traceback.print_exc()
            return details # Return defaults
