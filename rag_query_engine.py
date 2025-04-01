import sys
import json
import re
from datetime import timedelta
from typing import List, Optional, Dict, Any
import requests
from urllib.parse import quote_plus

from langchain_ollama import OllamaLLM # Note: Class name is OllamaLLM now
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import BaseRetriever  # For type hinting

from pdf_text_extractor import PdfTextExtractor
from mapping_item import MappingItem, QuestionInfo  # Import helper classes

# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL_NAME = "deepseek-r1:14b"
EMBEDDING_MODEL_NAME = "mxbai-embed-large"
TIMEOUT_EMBEDDING = timedelta(seconds=60)
TIMEOUT_CHAT = timedelta(seconds=300)  # Increased timeout for complex tasks

# Splitter Configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Retriever Configuration
MAX_RETRIEVER_RESULTS = 3

# External Service
import os
PAPERQA_SERVER_URL = os.getenv("PAPERQA_SERVER_URL", "http://34.147.75.119:8000/ai_prompt")

class RagQueryEngine:
    """Handles RAG-based querying of an ingested PDF document."""

    def __init__(self, pdf_filepath: str):
        """
        Initializes models, ingests the document, and sets up the RAG chain.

        Args:
            pdf_filepath: Path to the PDF document to process.
        """
        print(f"Initializing RAG Query Engine for: {pdf_filepath}")
        self.pdf_filepath = pdf_filepath
        try:
            # --- Initialize models ---
            self.embedding_model = OllamaEmbeddings(
                base_url=OLLAMA_BASE_URL,
                model=EMBEDDING_MODEL_NAME,
                # request_timeout=TIMEOUT_EMBEDDING.total_seconds(), # Timeout might be handled differently or not available directly
            )
            # Change Ollama to OllamaLLM
            self.chat_model = OllamaLLM(
                base_url=OLLAMA_BASE_URL,
                model=CHAT_MODEL_NAME,
                # request_timeout=TIMEOUT_CHAT.total_seconds(),
                temperature=0.1
            )
            print("Ollama models initialized.")

            # --- Ingest document ---
            self.vector_store: Optional[FAISS] = self._ingest_document(pdf_filepath)
            if self.vector_store is None:
                raise ValueError("Document ingestion failed, cannot proceed.")

            # --- Initialize Content Retriever ---
            self.retriever: BaseRetriever = self.vector_store.as_retriever(
                search_kwargs={'k': MAX_RETRIEVER_RESULTS}
            )
            print(f"FAISS Content Retriever initialized (k={MAX_RETRIEVER_RESULTS}).")

            # --- Define RAG Chain using LCEL ---
            self._setup_rag_chain()
            print("RAG chain configured.")

        except Exception as e:
            print(f"Failed to initialize RagQueryEngine: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise  # Propagate error

    def _ingest_document(self, pdf_filepath: str) -> Optional[FAISS]:
        """Extracts text, splits, embeds, and stores it in FAISS."""
        print(f"Ingesting document: {pdf_filepath}")
        extractor = PdfTextExtractor()
        pdf_text = extractor.extract_text(pdf_filepath)
        if not pdf_text or pdf_text.strip() == "":
            print("Failed to extract text. Cannot ingest.", file=sys.stderr)
            return None
        print(f"PDF text extracted ({len(pdf_text)} chars).")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )

        # Create a single document for splitting
        doc = Document(page_content=pdf_text, metadata={"source": pdf_filepath})
        split_docs = splitter.split_documents([doc])
        print(f"Document split into {len(split_docs)} chunks.")

        if not split_docs:
            print("Warning: Document splitting resulted in zero chunks.", file=sys.stderr)
            return None

        try:
            print("Creating FAISS index...")
            vector_store = FAISS.from_documents(split_docs, self.embedding_model)
            print("Document ingestion complete. FAISS index created.")
            return vector_store
        except Exception as e:
            print(f"Error during document ingestion (FAISS creation): {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None

    def _setup_rag_chain(self):
        """Sets up the LangChain Expression Language (LCEL) chain for RAG."""
        # Define the prompt template
        template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Helpful Answer:"""
        rag_prompt_template = PromptTemplate.from_template(template)

        # Define how to format retrieved documents
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # Define the RAG chain
        self.rag_chain = (
                RunnableParallel(
                    # Retrieve context based on question, format it, pass question through
                    {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                )
                | rag_prompt_template  # Apply prompt template
                | self.chat_model  # Query the LLM
                | StrOutputParser()  # Parse the LLM output string
        )

    def _retrieve_from_paperqa(self, paperqa_query: str) -> str:
        """Makes a request to the external PaperQA service."""
        # Ensure the query is URL encoded
        encoded_query = quote_plus(paperqa_query)
        server_url = f"{PAPERQA_SERVER_URL}?prompt={encoded_query}&use_docs=true"
        print(f"Querying PaperQA service: {PAPERQA_SERVER_URL}...")  # Don't print full query usually

        try:
            response = requests.get(server_url, timeout=120)  # Add timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            print(f"PaperQA Response Status: {response.status_code}")
            return response.text.strip()
        except requests.exceptions.RequestException as e:
            print(f"Error querying PaperQA service at {server_url}: {e}", file=sys.stderr)
            # Decide how to handle failure: return empty string, raise exception, etc.
            return f"Error: Could not contact PaperQA service - {e}"  # Return error message

    def get_patient_identifiers(self, all_mapping_items: List[MappingItem]) -> List[List[QuestionInfo]]:
        """
        Attempts to identify distinct patient/family identifiers using LLMs and generate questions.
        """
        print("\nAttempting to identify patient identifiers via LLM...")

        # 1. Query to extract potential identifiers (using external service or internal RAG)
        # Using the external service approach from the Java code
        patient_extraction_query = """
"Please extract a structured list of family-patient pairs from the document. Each line of the output should contain one set of family and corresponding patient identifier (for example, 'Family A: AII-11'). For each family, list all patient identifiers as provided in the text. If a unique numerical identifier is not provided, assign one sequentially within that family (starting with '1i' for the oldest affected patient). This formatted list will be used for subsequent detailed queries on each individual patient."
        """
        try:
            # Use the external PaperQA service as in the Java code
            # raw_identifier_info = self._retrieve_from_paperqa(patient_extraction_query)

            # --- Alternative: Use internal RAG for extraction (might be less reliable) ---
            # This queries *your own* ingested document using the RAG chain
            # The prompt needs to be tailored for extraction based on context.
            extraction_prompt = f"""
Based ONLY on the provided context, extract a list of family and patient identifiers mentioned.
Present the result as a simple list, one identifier or family-patient pair per line (e.g., "Family A: Patient II-1", "Patient 3", "Family B: Proband").
Do not add any explanation or introductory text.

Context:
{{context}}

List of Identifiers:
            """
            extraction_rag_prompt = PromptTemplate.from_template(extraction_prompt)
            extraction_chain = (
                    {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs))}
                    | extraction_rag_prompt
                    | self.chat_model
                    | StrOutputParser()
            )
            print("Querying internal RAG for identifiers...")
            # Provide a dummy question that triggers retrieval related to patients
            raw_identifier_info = extraction_chain.invoke("List all patient or family identifiers.")
            print(f"Raw identifier info from internal RAG:\n---\n{raw_identifier_info}\n---")
            # --- End Alternative ---

            if "Error:" in raw_identifier_info or not raw_identifier_info:
                print("Failed to retrieve identifier info from source.", file=sys.stderr)
                return []

            # Clean <think> tags if present (less likely with direct Ollama call)
            raw_identifier_info = re.sub(r'<think>.*?</think>', '', raw_identifier_info, flags=re.DOTALL)

            # 2. Use LLM to convert the raw list into JSON
            json_conversion_prompt = f"""
Please convert the provided family-patient information into a JSON array.
Each JSON object should include the following fields:
- "family": the family name (e.g., "A", "B"), or null if the patient is unaffiliated or family is not mentioned.
- "patient": the patient identifier (e.g., "AII-11", "Patient 3", "1i").

Follow this structure exactly. If the input is empty or contains no identifiers, return an empty JSON array `[]`.

Input Information:
\"\"\"
{raw_identifier_info}
\"\"\"

JSON Array Output:
            """
            print("Requesting LLM to format identifiers as JSON...")
            # Use the chat model directly for this formatting task
            json_response = self.chat_model.invoke(json_conversion_prompt)
            json_text = json_response.strip()
            print(f"LLM JSON response:\n---\n{json_text}\n---")

            # 3. Parse the JSON and generate questions
            # Extract JSON array cleanly (handle potential LLM preamble/postamble)
            match = re.search(r'(\[.*\])', json_text, re.DOTALL)
            if not match:
                print("Error: Could not find JSON array in the LLM response.", file=sys.stderr)
                # Try a simpler parse if it's just the array
                if json_text.startswith('[') and json_text.endswith(']'):
                    json_array_text = json_text
                else:
                    return []
            else:
                json_array_text = match.group(1)

            return self._generate_questions_from_patient_list(json_array_text, all_mapping_items)

        except Exception as e:
            print(f"Error trying to identify patients: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return []  # Return empty list on error

    def _generate_questions_from_patient_list(
            self, json_array_text: str, mapping_items: List[MappingItem]
    ) -> List[List[QuestionInfo]]:
        """Generates specific questions for each patient based on mapping items."""
        all_patient_questions: List[List[QuestionInfo]] = []
        try:
            patients = json.loads(json_array_text)
            if not isinstance(patients, list):
                print("Error: Parsed JSON is not a list.", file=sys.stderr)
                return []

            print(f"Generating questions for {len(patients)} identified patient entries.")

            for i, patient_entry in enumerate(patients):
                if not isinstance(patient_entry, dict):
                    print(f"Warning: Skipping invalid patient entry at index {i} (not a dict): {patient_entry}",
                          file=sys.stderr)
                    continue

                family_id = patient_entry.get("family")
                patient_id = patient_entry.get("patient")

                # Ensure IDs are strings, handle potential nulls from JSON
                family_id_str = str(family_id).strip() if family_id is not None else None
                patient_id_str = str(patient_id).strip() if patient_id is not None else None

                # Skip if no patient ID was found
                if not patient_id_str:
                    print(f"Warning: Skipping entry at index {i} due to missing patient ID: {patient_entry}",
                          file=sys.stderr)
                    continue

                one_patient_questions: List[QuestionInfo] = []
                print(f"  Generating questions for Patient ID: {patient_id_str}" + (
                    f" (Family: {family_id_str})" if family_id_str else ""))

                for item in mapping_items:
                    # Create the focused query for the RAG system
                    # Include patient and family context directly in the question
                    context_prefix = f"Regarding patient '{patient_id_str}'"
                    if family_id_str and family_id_str.lower() != 'null':
                        context_prefix += f" from family '{family_id_str}'"

                    # Use the original question from MappingItem, prefixed with context
                    specific_query = f"{context_prefix}: {item.question}"

                    q_info = QuestionInfo(
                        field=item.field,
                        query=specific_query,
                        response_convertion_strategy=item.response_convertion_strategy,
                        family_id=family_id_str,
                        patient_id=patient_id_str
                    )
                    one_patient_questions.append(q_info)

                if one_patient_questions:
                    all_patient_questions.append(one_patient_questions)

            return all_patient_questions

        except json.JSONDecodeError as json_err:
            print(f"Error parsing JSON for patient list: {json_err}", file=sys.stderr)
            print(f"Invalid JSON text received:\n---\n{json_array_text}\n---")
            return []
        except Exception as e:
            print(f"Error generating questions from patient list: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return []

    def query(self, question: str, response_conversion_strategy: str = "") -> str:
        """
        Answers a question using the configured RAG chain.
        The response_conversion_strategy is not directly used by the RAG chain itself
        but is passed along for potential use in formatting later.

        Args:
            question: The user's question text.
            response_conversion_strategy: Hint for later formatting (unused here).

        Returns:
            The answer from the LLM based on retrieved context, or an error message.
        """
        print(f"\nReceived RAG query: \"{question}\"")
        if not hasattr(self, 'rag_chain'):
            return "Error: RAG chain is not initialized."

        try:
            # Invoke the RAG chain
            result = self.rag_chain.invoke(question)
            print("LLM RAG response received.")
            # Simple check if the model explicitly stated it doesn't know
            if "don't know" in result.lower() or "not find information" in result.lower():
                print("LLM indicated context might be insufficient.")
                # Return a standardized unknown value, perhaps? Or the LLM's response.
                # For now, return the LLM response directly.
                # return "-99" # Or based on formatting needs later

            return result.strip()

        except Exception as e:
            print(f"Error querying RAG chain: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return f"Error: Failed to get answer via RAG - {e}"


# --- Example Usage ---
if __name__ == "__main__":
    from pathlib import Path

    # IMPORTANT: Replace with the actual path to YOUR PDF file
    # Use raw string (r"...") or forward slashes for paths
    pdf_file_path = r"D:\000333\fine_tune_deepseek_r1\test_pdf\ando2012-22991136.pdf"

    if not Path(pdf_file_path).exists():
        print(f"ERROR: Test PDF file not found at: {pdf_file_path}")
    else:
        try:
            query_engine = RagQueryEngine(pdf_file_path)

            # --- Test Basic RAG Query ---
            test_question = "What is the PubMed ID (PMID) of the study?"
            response_strategy = "Enter the PubMed ID (PMID) of the publication as a numeric identifier (e.g., 28847615)."
            answer = query_engine.query(test_question, response_strategy)

            print("\n====================================")
            print(f"Question: {test_question}")
            print(f"Answer: {answer}")
            print("====================================")

            # --- Test Patient Identification (requires mapping items) ---
            # Create a dummy mapping list for the test
            dummy_mapping_items = [
                MappingItem(field='pmid', question='What is PMID?', mapped_excel_column='PMID',
                            response_convertion_strategy='Number'),
                MappingItem(field='aao', question='What is Age at Onset?', mapped_excel_column='AAO',
                            response_convertion_strategy='Number or -99'),
                MappingItem(field='sex', question='What is sex?', mapped_excel_column='Sex',
                            response_convertion_strategy='M/F or -99'),
            ]
            print("\n--- Testing Patient Identification ---")
            list_of_question_lists = query_engine.get_patient_identifiers(dummy_mapping_items)

            if list_of_question_lists:
                print(f"\nIdentified {len(list_of_question_lists)} potential patient sets.")
                for i, question_set in enumerate(list_of_question_lists):
                    if question_set:
                        first_q = question_set[0]
                        print(
                            f"  Set {i + 1}: Patient ID='{first_q.patient_id}', Family ID='{first_q.family_id}' ({len(question_set)} questions generated)")
                        # print(f"    Example Question: {first_q.query}")
                    else:
                        print(f"  Set {i + 1}: Empty question set generated.")
            else:
                print("\nNo patient sets identified or questions generated.")

        except Exception as main_e:
            print(f"\nAn error occurred during the RagQueryEngine example: {main_e}", file=sys.stderr)
            import traceback

            traceback.print_exc()