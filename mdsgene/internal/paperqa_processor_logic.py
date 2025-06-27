# paperqa_processor_logic.py
import asyncio
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import paperqa
from paperqa import Settings
from paperqa.settings import AgentSettings

from mdsgene.internal.defines import NO_INFORMATION_LIST

# --- Configuration ---
# It's best practice to load API keys from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Default model, can be overridden in constructor
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


class PaperQAProcessorLogic:
    """Handles interactions with the paper-qa library for PDF processing."""

    def __init__(
        self,
        pdf_filepath: Optional[Path] = None,
        *,
        pdf_uri: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
    ):
        """Initializes the PaperQA client and prepares the PDF.

        Either ``pdf_uri`` or ``pdf_filepath`` must be provided.

        Args:
            pdf_filepath: Local path to the PDF file.
            pdf_uri: URI for the PDF (not directly used by paper-qa, but kept for API compatibility).
            model_name: Specific model name to use with paper-qa.
            api_key: Google AI API key. If ``None`` uses ``GEMINI_API_KEY``.
        """
        resolved_api_key = api_key or GEMINI_API_KEY
        if not resolved_api_key:
            raise ValueError(
                "Gemini API key not provided and GEMINI_API_KEY environment variable not set."
            )

        self.model_name = model_name
        self.pdf_filepath = pdf_filepath
        self.pdf_uri = pdf_uri
        self._answer_cache = {}  # Cache for answer results

        if not self.pdf_uri and not self.pdf_filepath:
            raise ValueError("Either pdf_uri or pdf_filepath must be provided")

        try:
            # Set environment variables for Gemini API
            os.environ["GOOGLE_API_KEY"] = resolved_api_key
            if "GEMINI_API_KEY" not in os.environ:
                os.environ["GEMINI_API_KEY"] = resolved_api_key

            # Configure PaperQA with Gemini models using Settings with timeout
            self.settings = Settings(
                llm=f"gemini/{model_name}",
                summary_llm=f"gemini/{model_name}",
                agent=AgentSettings(agent_llm=f"gemini/{model_name}"),
                embedding="gemini/text-embedding-004",
                # Add timeout and chunk settings
                answer_max_sources=3,  # Maximum number of sources to use for an answer
                llm_config={
                    "timeout": 120,  # Increase timeout to 2 minutes
                    "max_retries": 2,
                },
                chunk_size=3000,  # Reduce chunk size for faster processing
                overlap=100
            )

            # Initialize paper-qa with settings passed directly
            try:
                # Try passing settings to constructor
                self.docs = paperqa.Docs(settings=self.settings)
            except Exception as constructor_error:
                print(f"Constructor with settings failed: {constructor_error}")
                # Fallback: initialize without settings then apply
                self.docs = paperqa.Docs()
                if hasattr(self.docs, 'settings'):
                    self.docs.settings = self.settings
                else:
                    print("WARNING: Could not apply settings to Docs object")

            print(f"PaperQA initialized with Gemini model '{model_name}'.")

            # Load the PDF if filepath is provided
            if self.pdf_filepath:
                if not self.pdf_filepath.exists() or not self.pdf_filepath.is_file():
                    raise FileNotFoundError(f"PDF file not found: {self.pdf_filepath}")
                self._load_pdf()

        except Exception as e:
            print(f"Error initializing PaperQA: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def _load_pdf(self) -> None:
        """Load the PDF into paper-qa."""
        try:
            print(f"Loading PDF: {self.pdf_filepath}")
            # Add the document to paper-qa with chunking options
            self.docs.add(
                str(self.pdf_filepath), 
                citation=str(self.pdf_filepath.stem),
                chunk_chars=3000,  # Smaller chunks for faster processing
                overlap=100
            )
            print("PDF loaded successfully into PaperQA.")
        except Exception as e:
            print(f"Error reading PDF file {self.pdf_filepath}: {e}", file=sys.stderr)
            traceback.print_exc()
            raise

    def _run_async_in_thread(self, coro, timeout=180):
        """Run async coroutine in a separate thread to avoid event loop conflicts."""
        import concurrent.futures

        def run_in_new_loop():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                # Create a task and ensure it's properly awaited
                task = asyncio.ensure_future(coro, loop=new_loop)
                try:
                    return new_loop.run_until_complete(asyncio.wait_for(task, timeout=timeout))
                except TypeError as type_error:
                    # Specifically catch TypeError which can happen in paperqa's docs.py
                    # when answer_text is None and it tries to check if something is in it
                    print(f"TypeError in PaperQA execution: {type_error}")
                    print("This is likely due to a None value being used where a string was expected")
                    traceback.print_exc()
                    return None
            except asyncio.TimeoutError:
                print(f"Query timed out after {timeout} seconds")
                return None
            except Exception as e:
                print(f"Error in async execution: {e}")
                traceback.print_exc()
                return None
            finally:
                # Ensure all pending tasks are completed or cancelled
                pending_tasks = asyncio.all_tasks(new_loop)
                for task in pending_tasks:
                    task.cancel()
                if pending_tasks:
                    # Give tasks a chance to cancel properly
                    try:
                        new_loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
                    except Exception:
                        pass
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_new_loop)
            try:
                return future.result(timeout=timeout + 10)  # Extra buffer for thread timeout
            except concurrent.futures.TimeoutError:
                print(f"Thread execution timed out after {timeout + 10} seconds")
                return None
            except Exception as e:
                print(f"Error in thread execution: {e}")
                traceback.print_exc()
                return None

    def get_patient_identifiers(self) -> List[Dict[str, Optional[str]]]:
        """
        Uses PaperQA to extract a list of patient and family identifiers from the PDF.

        Returns:
            List of dictionaries with 'patient' and 'family' keys.
        """
        print("\nRequesting patient identifiers from PaperQA...")
        # Simplified prompt to reduce processing time
        extraction_prompt = (
            "Extract patient identifiers from this document. Return as JSON array: "
            "[{'family': 'family_id_or_null', 'patient': 'patient_id'}]. "
            "Only include actual patient cases mentioned in the text."
        )

        try:
            # Query paper-qa using async method in separate thread with shorter timeout
            answer = self._run_async_in_thread(
                self.docs.aquery(extraction_prompt, settings=self.settings), 
                timeout=90
            )

            # Handle case where answer is None or invalid
            if answer is None:
                print("ERROR: Query timed out or failed for patient identifiers.", file=sys.stderr)
                return []

            # Check if answer has the expected attributes before using it
            if not hasattr(answer, 'answer') or answer.answer is None:
                print("ERROR: Invalid answer object returned for patient identifiers.", file=sys.stderr)
                return []

            # Get the answer text safely
            json_text = str(answer.answer) if answer.answer is not None else ""

            if not json_text:
                print("ERROR: Failed to get response for patient identifiers.", file=sys.stderr)
                return []

            print(f"  Raw PaperQA JSON response:\n---\n{json_text}\n---")

            # Process the response similar to GeminiProcessorLogic
            match = re.search(r'\[\s*\{.*?\}\s*\]', json_text, re.DOTALL)
            if match:
                json_array_text = match.group(0)
            elif json_text.strip().startswith('[') and json_text.strip().endswith(']'):
                json_array_text = json_text.strip()
            else:
                print("ERROR: Could not find a valid JSON array structure in the response.", file=sys.stderr)
                print(f"Received text: {json_text}")
                return []

            try:
                patients = json.loads(json_array_text)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON response for patient list: {e}", file=sys.stderr)
                print(f"  Invalid JSON text received:\n---\n{json_array_text}\n---", file=sys.stderr)
                return []

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
            print(f"  Invalid JSON text received:\n---\n{locals().get('json_array_text', '') or locals().get('json_text', '')}\n---", file=sys.stderr)
            return []
        except Exception as e:
            print(f"ERROR: Unexpected error processing patient list: {e}", file=sys.stderr)
            traceback.print_exc()
            return []

    def answer_question(self, question_text: str) -> Optional[Tuple[str, str]]:
        """
        Asks a specific question to PaperQA based on the loaded PDF context.
        Uses caching to avoid repeated queries for the same question.

        Args:
            question_text: The question to ask.

        Returns:
            A tuple of (answer, context), or None if the question cannot be answered
        """
        # Check cache first
        if question_text in self._answer_cache:
            print(f"  Using cached answer for: '{question_text[:100]}...'")
            return self._answer_cache[question_text]

        # Simplified prompt to reduce processing time
        prompt = (
            f"Answer this question based on the document: {question_text}\n"
            f"If information not found, respond 'Information not found'."
        )
        print(f"  Asking PaperQA: {question_text[:100]}...")

        try:
            # Query paper-qa using async method in separate thread with explicit settings and timeout
            answer = self._run_async_in_thread(
                self.docs.aquery(prompt, settings=self.settings), 
                timeout=120
            )

            # Handle case where answer is None or invalid
            if answer is None:
                print(f"  ERROR: Query timed out or failed for question: '{question_text[:100]}...'", file=sys.stderr)
                return None

            # Check if answer has the expected attributes before using it
            if not hasattr(answer, 'answer') or answer.answer is None:
                print(f"  ERROR: Invalid answer object returned for question: '{question_text[:100]}...'", file=sys.stderr)
                # Create a fallback answer
                raw_answer = "Error: Unable to process this question due to technical issues."
            else:
                # Get the answer text safely
                raw_answer = str(answer.answer) if answer.answer is not None else "Error: No answer was generated."

            if not raw_answer:
                print(f"ERROR: Empty response for question: '{question_text[:100]}...'", file=sys.stderr)
                raw_answer = "Error: Empty response received."

            if raw_answer.lower() in NO_INFORMATION_LIST:
                print(f"PaperQA indicated information not found for: '{question_text[:100]}...'")

            # Create result tuple
            result = (raw_answer, prompt)

            # Cache the result
            self._answer_cache[question_text] = result
            print(f"  Cached answer for future use: '{question_text[:50]}...'")

            # Return both the answer and the prompt as context
            return result

        except Exception as e:
            print(f"ERROR: Failed to answer question: {e}", file=sys.stderr)
            traceback.print_exc()
            # Return a fallback response instead of None
            fallback_answer = (f"Error processing question: {str(e)[:100]}...", prompt)
            return fallback_answer

    def extract_publication_details(self) -> Dict[str, Optional[str]]:
        """
        Uses PaperQA to extract key publication details (title, author, year) from the PDF.

        Returns:
            A dictionary with keys 'title', 'first_author_lastname', 'year',
            or None values if details cannot be extracted.
        """
        print("\nRequesting publication details (title, author, year) from PaperQA...")
        # Simplified prompt to reduce processing time
        extraction_prompt = (
            "Extract publication details as JSON: "
            "{'title': 'paper_title', 'first_author_lastname': 'last_name', 'year': 'YYYY'}. "
            "Use null for missing information."
        )

        details = {'title': None, 'first_author_lastname': None, 'year': None}  # Default values

        try:
            # Query paper-qa using async method in separate thread with explicit settings and timeout
            answer = self._run_async_in_thread(
                self.docs.aquery(extraction_prompt, settings=self.settings), 
                timeout=90
            )

            # Handle case where answer is None or invalid
            if answer is None:
                print("ERROR: Query timed out or failed for publication details.", file=sys.stderr)
                return details

            # Check if answer has the expected attributes before using it
            if not hasattr(answer, 'answer') or answer.answer is None:
                print("ERROR: Invalid answer object returned for publication details.", file=sys.stderr)
                return details

            # Get the answer text safely
            json_text = str(answer.answer) if answer.answer is not None else ""

            if not json_text:
                print("ERROR: Failed to get response for publication details.", file=sys.stderr)
                return details  # Return defaults

            print(f"  Raw PaperQA JSON response for details:\n---\n{json_text}\n---")

            # Process the response similar to GeminiProcessorLogic
            match = re.search(r'\{\s*".*?":.*?\s*\}', json_text, re.DOTALL)
            if match:
                json_object_text = match.group(0)
            elif json_text.strip().startswith('{') and json_text.strip().endswith('}'):
                json_object_text = json_text.strip()
            else:
                print(
                    "ERROR: Could not find a valid JSON object structure in the response for details.",
                    file=sys.stderr
                )
                print(f"Received text: {json_text}")
                return details  # Return defaults

            try:
                parsed_json = json.loads(json_object_text)
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to parse JSON response for publication details: {e}", file=sys.stderr)
                print(f"  Invalid JSON text received:\n---\n{json_object_text}\n---", file=sys.stderr)
                return details  # Return defaults

            if not isinstance(parsed_json, dict):
                print("ERROR: Parsed JSON for details is not a dictionary.", file=sys.stderr)
                return details  # Return defaults

            # Extract details, handling potential missing keys or null values
            details['title'] = str(parsed_json.get('title')).strip() if parsed_json.get('title') else None
            details['first_author_lastname'] = str(parsed_json.get('first_author_lastname')).strip() \
                if parsed_json.get('first_author_lastname') else None
            details['year'] = str(parsed_json.get('year')).strip() if parsed_json.get('year') else None

            # Basic validation
            if details['year'] and not re.match(r'^\d{4}$', details['year']):
                print(f"WARNING: Extracted year '{details['year']}' is not 4 digits. Setting to None.")
                details['year'] = None

            print(f"  Extracted publication details: {details}")
            return details

        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON response for publication details: {e}", file=sys.stderr)
            print(f"  Invalid JSON text received:\n---\n{locals().get('json_object_text', '') or locals().get('json_text', '')}\n---", file=sys.stderr)
            return details  # Return defaults
        except Exception as e:
            print(f"ERROR: Unexpected error processing publication details: {e}", file=sys.stderr)
            traceback.print_exc()
            return details  # Return defaults
