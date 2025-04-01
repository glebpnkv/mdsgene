import sys
import os
from datetime import timedelta
from typing import List, Dict, Optional
from pathlib import Path
import re

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import LLMResult

# Import local modules
from mapping_item import MappingItem, QuestionInfo
from rag_query_engine import RagQueryEngine  # Requires previous setup
from excel_writer import ExcelWriter

# --- Configuration ---
# Use Path objects for better path handling
PDF_FILEPATH = Path(r"D:/000333/fine_tune_deepseek_r1/test_pdf/ando2012-22991136.pdf")
OUTPUT_EXCEL_PATH = Path("./Patient_Data_Appended_Python.xlsx")

# Define which fields are considered "common" vs "patient-specific"
# These are field names from MappingItem
# COMMON_FIELDS = ["pmid", "author, year", "comments_study"] # Example from Java
COMMON_FIELDS = []  # Start with none as per the uncommented Java code

# Ollama config for formatting (can potentially reuse from RAG engine)
FORMATTER_OLLAMA_BASE_URL = "http://localhost:11434"
FORMATTER_MODEL_NAME = "mistral"  # Model used for formatting in Java code
FORMATTER_TIMEOUT = timedelta(seconds=60)  # Shorter timeout for formatting


class ExcelMappingApp:
    """Orchestrates PDF processing, RAG querying, formatting, and Excel writing."""

    def __init__(self):
        """Initializes the formatting LLM."""
        try:
            # Create a separate LLM instance for formatting tasks
            self.formatter_llm = OllamaLLM(
                base_url=FORMATTER_OLLAMA_BASE_URL,
                model=FORMATTER_MODEL_NAME,
                # request_timeout=FORMATTER_TIMEOUT.total_seconds(),
                temperature=0.0  # Make formatting deterministic
            )
            print(f"Formatter LLM ({FORMATTER_MODEL_NAME}) initialized.")
        except Exception as e:
            print(f"Error initializing formatting LLM: {e}", file=sys.stderr)
            self.formatter_llm = None  # Indicate failure

    # --- Populated createMappingData method (Python equivalent) ---
    def create_mapping_data(self) -> List[MappingItem]:
        """Creates the list of fields, questions, and formatting strategies."""
        mapping_data: List[MappingItem] = []
        # Add items using the MappingItem dataclass
        # Copy all the commented and uncommented lines from the Java version

        # Example uncommented item:
        mapping_data.append(MappingItem(
            field="aao",
            question="What was the age at onset of symptoms? (Enter -99 if unknown)",
            mapped_excel_column="Guessed: AAO",
            response_convertion_strategy="Enter the age at onset (AAO) of symptoms, in years, as reported in the publication. If the age at onset is not given, enter -99."
        ))

        # Add all other MappingItem instances from your Java code here...
        # Make sure the 'field' names are consistent (e.g., used in COMMON_FIELDS, results dict keys)

        # Example commented item translated:
        # mapping_data.append(MappingItem(
        #     field="pmid",
        #     question="What is the PubMed ID of the article?",
        #     mapped_excel_column="Guessed: PMID",
        #     response_convertion_strategy="Enter the PubMed ID (PMID) of the publication as a numeric identifier (e.g., 28847615)."
        # ))
        # mapping_data.append(MappingItem(
        #     field="author, year",
        #     question="Who is the first author and what is the publication year?",
        #     mapped_excel_column="Guessed: Author_year",
        #     response_convertion_strategy="Enter the last name of the first author, followed by a comma and the four-digit publication year (e.g., 'Smith, 2018')."
        # ))
        # mapping_data.append(MappingItem(
        #     field="comments_study",
        #     question="Any general comments about the study?",
        #     mapped_excel_column="Guessed: Study Comments",
        #     response_convertion_strategy="Enter any overarching comments regarding the study. For example, if the authors were contacted for missing information, record the date of contact and any relevant response in this field."
        # ))
        # ... add all others ...

        return mapping_data

    # --- Implemented formatAnswer method ---
    def format_answer(self, raw_answer: Optional[str], strategy: str) -> str:
        """Formats the raw RAG answer using an LLM based on the provided strategy."""
        if self.formatter_llm is None:
            print(" -> ERROR: Formatter LLM not available. Returning raw answer.", file=sys.stderr)
            return raw_answer.strip() if raw_answer else "-99"  # Fallback

        if not raw_answer or raw_answer.strip() == "" or "error" in raw_answer.lower():
            # print(f" -> Raw answer is empty or indicates error ('{raw_answer}'). Returning -99.")
            return "-99"  # Default to -99 if query failed or answer is blank

        # Clean up common RAG failure indicators before sending to formatter
        if "don't know" in raw_answer.lower() or "couldn't find" in raw_answer.lower():
            print(f" -> Raw answer indicates info not found ('{raw_answer[:50]}...'). Returning -99.")
            return "-99"

        try:
            # Build formatting prompt (similar to Java)
            formatting_prompt = f"""
Format the following raw answer according to these instructions.
Provide ONLY the final formatted value, with no explanation or preamble.

Raw answer from previous query:
\"\"\"
{raw_answer}
\"\"\"

Formatting strategy / Expected output description:
\"\"\"
{strategy}
\"\"\"

Rules for formatting:
1. For numeric answers (age, counts): Extract just the number. If unknown/not given/not found, return -99. If explicitly zero/none, return 0.
2. For yes/no questions: Return lowercase "yes" or "no". If unknown/not mentioned, return -99.
3. For sex: Return uppercase "M" or "F". If unknown, return -99.
4. For zygosity: Return "heterozygous", "homozygous", or "hemizygous". If unknown/not applicable, return -99.
5. For inclusion decisions (IN/EX): Return uppercase "IN" or "EX". If unclear, return -99.
6. For optional fields or when info is absent: If not applicable/none/blank/not found, return -99.
7. Remove any preamble like "The answer is:", "Based on the context:", etc. Just give the value.
8. If the raw answer clearly indicates the information is missing or unknown (e.g., "not stated", "unknown", "not reported", "N/A", "could not find"), return -99.

Formatted answer:"""  # No trailing """ here

            # Query the formatter model
            response = self.formatter_llm.invoke(formatting_prompt)
            formatted_answer = response.strip()

            # Apply minimal post-processing
            # Remove surrounding quotes if LLM added them
            if formatted_answer.startswith('"') and formatted_answer.endswith('"') and len(formatted_answer) > 1:
                formatted_answer = formatted_answer[1:-1]
            elif formatted_answer.startswith("'") and formatted_answer.endswith("'") and len(formatted_answer) > 1:
                formatted_answer = formatted_answer[1:-1]

            # Sometimes models still add explanations, try to remove common ones
            formatted_answer = re.sub(r'^(Formatted Answer|Answer|The formatted answer is):\s*', '', formatted_answer,
                                      flags=re.IGNORECASE)

            # Fallback for empty responses after stripping
            if not formatted_answer:
                print(
                    f" -> WARNING: LLM returned empty response for formatting: '{raw_answer[:50]}...'. Using cleaned raw answer or -99.")
                # Decide on a better fallback: -99 or cleaned raw answer? Let's use -99 for consistency.
                return "-99"
                # return raw_answer.strip()

            # Final sanity check for common "unknown" phrases missed by LLM
            if formatted_answer.lower() in ["unknown", "not stated", "not reported", "n/a", "none", "not applicable",
                                            "not mentioned"]:
                print(f" -> LLM returned '{formatted_answer}', interpreting as unknown. Returning -99.")
                return "-99"

            return formatted_answer

        except Exception as e:
            # Fallback to basic cleaning if LLM formatting fails
            print(f" -> ERROR using LLM for formatting: {e}. Falling back to basic cleaning or -99.", file=sys.stderr)
            # Decide on fallback: -99 seems safer if formatting failed
            return "-99"
            # return raw_answer.strip()

    # --- Main execution logic ---
    def run(self):
        """Runs the main application workflow."""
        print("Starting Excel Mapping Application...")

        # --- File Path Checks ---
        if not PDF_FILEPATH.exists():
            print(f"CRITICAL ERROR: Input PDF file not found at '{PDF_FILEPATH}'", file=sys.stderr)
            return
        if not PDF_FILEPATH.is_file():
            print(f"CRITICAL ERROR: Input path '{PDF_FILEPATH}' is not a file.", file=sys.stderr)
            return

        # --- Initialization ---
        try:
            print(f"\nInitializing RAG Query Engine for: {PDF_FILEPATH}")
            query_engine = RagQueryEngine(str(PDF_FILEPATH))  # RAG Engine expects string path maybe
            print("RAG Query Engine initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize RAG Query Engine: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return  # Cannot proceed without RAG engine

        # --- Load Mapping Definitions ---
        all_mapping_items = self.create_mapping_data()
        if not all_mapping_items:
            print("Error: No mapping items loaded. Cannot proceed.", file=sys.stderr)
            return
        print(f"\nLoaded {len(all_mapping_items)} field definitions.")

        # --- Get Patient Identifiers ---
        try:
            # This now returns List[List[QuestionInfo]]
            list_of_patient_question_sets = query_engine.get_patient_identifiers(all_mapping_items)
        except Exception as e:
            print(f"Error getting patient identifiers: {e}", file=sys.stderr)
            list_of_patient_question_sets = []

        # --- Prepare Headers for Excel (Dynamically) ---
        # Header order is determined by all_mapping_items order
        # We also need a column for the identified patient ID itself.
        # Let's insert it after potential common fields.
        headers_for_excel: List[MappingItem] = []
        added_patient_id_header = False

        # Add common headers first (if any)
        for item in all_mapping_items:
            if item.field in COMMON_FIELDS:
                headers_for_excel.append(item)

        # Add the patient identifier header if not already a common field
        if "Identified_Patient_ID" not in COMMON_FIELDS:
            # Create a placeholder MappingItem for the header
            headers_for_excel.append(MappingItem("Identified_Patient_ID", "", "", ""))
            added_patient_id_header = True

        # Add patient-specific headers
        for item in all_mapping_items:
            if item.field not in COMMON_FIELDS:
                # Avoid adding if it was the dynamically added patient ID placeholder
                if not (added_patient_id_header and item.field == "Identified_Patient_ID"):
                    headers_for_excel.append(item)

        print(f"\nGenerated {len(headers_for_excel)} headers for Excel: {[h.field for h in headers_for_excel]}")

        # --- Check if patients were found ---
        if not list_of_patient_question_sets:
            print("\nNo patient identifiers found or questions generated. Cannot process patient data.")
            # Optionally: Write common info if needed, or just exit.
        else:
            # --- Process Each Identified Patient ---
            print(f"\nProcessing {len(list_of_patient_question_sets)} identified patient sets...")
            patient_num = 0
            for patient_question_set in list_of_patient_question_sets:
                patient_num += 1
                if not patient_question_set:
                    print(f"\n=== Skipping Patient Set {patient_num} (Empty Question Set) ===")
                    continue

                # Assuming all questions in the set are for the same patient/family
                current_patient_id = patient_question_set[0].patient_id
                current_family_id = patient_question_set[0].family_id
                print(
                    f"\n=== Processing Patient Set {patient_num} (ID: '{current_patient_id}', Family: '{current_family_id}') ===")

                # Dictionary to store results for this patient row
                patient_results: Dict[str, str] = {}
                # Add the identified patient ID to the results
                patient_results["Identified_Patient_ID"] = current_patient_id or "-99"  # Use -99 if somehow None

                # Add common results (if they were fetched - currently COMMON_FIELDS is empty)
                # for common_field in COMMON_FIELDS:
                #     # You would need to query common fields *once* before this loop
                #     # and store them, then add here.
                #     patient_results[common_field] = common_data.get(common_field, "-99")

                # Process patient-specific questions
                for q_obj in patient_question_set:
                    # Skip common fields if they were handled above
                    if q_obj.field in COMMON_FIELDS:
                        continue

                    print(f"--- Querying for field: {q_obj.field} ---")
                    print(f"    Query: {q_obj.query}")

                    # Query using RAG engine
                    raw_answer = query_engine.query(q_obj.query, q_obj.response_convertion_strategy)
                    print(f"    Raw RAG Answer: {raw_answer[:100] if raw_answer else 'None'}...")

                    # Format the answer using LLM
                    print(f"    Formatting Strategy: {q_obj.response_convertion_strategy}")
                    formatted_answer = self.format_answer(raw_answer, q_obj.response_convertion_strategy)
                    print(f"    Formatted Answer: {formatted_answer}")

                    # Store the formatted result using the 'field' name as the key
                    patient_results[q_obj.field] = formatted_answer
                    print("--- Finished query ---")

                # --- Write this patient's row to Excel ---
                print(f"\nPreparing to write/append patient '{current_patient_id}' entry to Excel file...")
                ExcelWriter.write_or_append_excel(OUTPUT_EXCEL_PATH, headers_for_excel, patient_results)

        print("\nApplication finished.")


# --- Entry Point ---
if __name__ == "__main__":
    app = ExcelMappingApp()
    app.run()