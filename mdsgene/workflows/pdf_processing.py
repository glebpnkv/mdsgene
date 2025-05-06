import os
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any

from ai.pmid_extractor import PmidExtractor
from ai.agents.publication_details_agent import PublicationDetailsAgent
from ai.agents.patient_identifiers_agent import PatientIdentifiersAgent
from ai.agents.questions_processing_agent import QuestionsProcessingAgent

def process_pdf_file(pdf_filepath: str) -> Dict[str, Any]:
    """
    Process a PDF file and extract information using agents.
    
    This function uses a sequence of agents to process a PDF file:
    1. PublicationDetailsAgent - Extracts publication details and PMID
    2. PatientIdentifiersAgent - Extracts patient identifiers
    3. QuestionsProcessingAgent - Processes questions for each patient
    
    Args:
        pdf_filepath: Path to the PDF file to process
        
    Returns:
        Dictionary containing the processing results
        
    Raises:
        ValueError: If the PDF file is not found or other validation errors
        Exception: For other processing errors
    """
    # Fix path if it has duplicated filename
    pdf_path = Path(pdf_filepath)
    if str(pdf_path.parent.name) == pdf_path.name:
        # Path has duplicated filename, fix it
        corrected_path = pdf_path.parent
        pdf_filepath = str(corrected_path)
        print(f"Corrected duplicated path to: {pdf_filepath}")
    
    # Validate the PDF filepath
    if not Path(pdf_filepath).exists():
        raise ValueError(f"Error: PDF file not found at {pdf_filepath}")
    
    print(f"Processing PDF: {pdf_filepath}")
    
    try:
        # Step 1: Extract publication details and PMID using PublicationDetailsAgent
        publication_agent = PublicationDetailsAgent()
        publication_agent.setup()
        
        publication_initial_state = {
            "pdf_filepath": pdf_filepath,
            "publication_details": None,
            "pmid": None,
            "messages": [
                {"role": "user", "content": f"Extract publication details and PMID from {Path(pdf_filepath).name}"}
            ]
        }
        
        publication_final_state = publication_agent.run(publication_initial_state)
        
        # Extract PMID and publication details from the result
        pmid = publication_final_state.get("pmid")
        publication_details = publication_final_state.get("publication_details", {})
        
        print(f"Publication details extracted. PMID: {pmid}")
        
        # Step 2: Extract patient identifiers using PatientIdentifiersAgent
        patient_agent = PatientIdentifiersAgent(pmid)
        patient_agent.setup()
        
        patient_initial_state = {
            "pdf_filepath": pdf_filepath,
            "patient_identifiers": [],
            "messages": [
                {"role": "user", "content": f"Extract patient identifiers from {Path(pdf_filepath).name}"}
            ]
        }
        
        try:
            patient_final_state = patient_agent.run(patient_initial_state)
            patient_identifiers = patient_final_state.get("patient_identifiers", [])
            print(f"Patient identifiers extracted: {len(patient_identifiers)} patients")
        except ValueError as e:
            # This will catch the exception thrown when patient identifiers are not found in cache
            print(f"WARNING: {e}")
            patient_identifiers = []
        
        # Step 3: Process questions for each patient using QuestionsProcessingAgent
        if patient_identifiers:
            questions_agent = QuestionsProcessingAgent(pmid)
            questions_agent.setup()
            
            questions_initial_state = {
                "pdf_filepath": pdf_filepath,
                "mapping_items": [],
                "patient_identifiers": patient_identifiers,
                "patient_questions": [],
                "patient_answers": [],
                "vector_store": None,
                "processors": [],
                "messages": [
                    {"role": "user", "content": f"Process questions for patients in {Path(pdf_filepath).name}"}
                ]
            }
            
            try:
                questions_final_state = questions_agent.run(questions_initial_state)
                patient_answers = questions_final_state.get("patient_answers", [])
                print(f"Patient questions processed: {len(patient_answers)} patient data rows")
            except Exception as e:
                print(f"ERROR processing questions: {e}")
                patient_answers = []
        else:
            patient_answers = []
        
        # Prepare results
        results = {
            "pdf_filename": Path(pdf_filepath).name,
            "publication_details": publication_details,
            "pmid": pmid,
            "patient_identifiers": patient_identifiers,
            "patient_answers": patient_answers,
            "messages": []
        }
        
        # Process messages for response
        for message in publication_final_state.get("messages", []):
            # Handle both dict-style messages and LangChain message objects
            if hasattr(message, "type") and hasattr(message, "content"):
                # This is a LangChain message object (like HumanMessage)
                role = message.type
                content = message.content
                results["messages"].append({"role": role, "content": content})
            elif isinstance(message, dict) and "role" in message and "content" in message:
                # This is a dictionary-style message
                results["messages"].append(message)
        
        print("\n=== Results ===")
        print(f"PDF: {results['pdf_filename']}")
        print(f"Publication Details: {results['publication_details']}")
        print(f"PMID: {results['pmid']}")
        print(f"Patient Identifiers: {len(results['patient_identifiers'])} patients")
        print(f"Patient Answers: {len(results['patient_answers'])} rows")
        
        return results
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        traceback.print_exc()
        raise