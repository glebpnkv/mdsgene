import sys
from pathlib import Path

from ai.agents.publication_details_agent import PublicationDetailsAgent
from ai.agents.patient_identifiers_agent import PatientIdentifiersAgent
from ai.agents.questions_processing_agent import QuestionsProcessingAgent

def main():
    """
    Example workflow demonstrating how to use all three agents in sequence.
    
    This script:
    1. Extracts publication details and PMID from a PDF
    2. Extracts patient identifiers from the same PDF
    3. Processes questions for each patient
    """
    # Get PDF filepath from command line argument or use a default
    if len(sys.argv) > 1:
        pdf_filepath = sys.argv[1]
    else:
        # Prompt user for PDF filepath
        pdf_filepath = input("Enter the path to a PDF file: ")
    
    # Validate the PDF filepath
    if not Path(pdf_filepath).exists():
        print(f"Error: PDF file not found at {pdf_filepath}")
        sys.exit(1)
    
    pdf_name = Path(pdf_filepath).name
    print(f"Processing PDF: {pdf_name}")
    
    try:
        # Step 1: Extract publication details and PMID
        print("\n=== Step 1: Extracting Publication Details ===")
        pub_agent = PublicationDetailsAgent()
        pub_agent.setup()
        
        pub_initial_state = {
            "pdf_filepath": pdf_filepath,
            "publication_details": None,
            "pmid": None,
            "messages": [
                {"role": "user", "content": f"Extract publication details and PMID from {pdf_name}"}
            ]
        }
        
        pub_final_state = pub_agent.run(pub_initial_state)
        pub_agent.print_results(pub_final_state)
        
        # Save PDF with PMID as filename if PMID was found
        pub_agent.save_pdf_with_pmid(pub_final_state)
        
        # Step 2: Extract patient identifiers
        print("\n=== Step 2: Extracting Patient Identifiers ===")
        patient_agent = PatientIdentifiersAgent()
        patient_agent.setup()
        
        patient_initial_state = {
            "pdf_filepath": pdf_filepath,
            "patient_identifiers": [],
            "messages": [
                {"role": "user", "content": f"Extract patient identifiers from {pdf_name}"}
            ]
        }
        
        patient_final_state = patient_agent.run(patient_initial_state)
        patient_agent.print_results(patient_final_state)
        
        # Step 3: Process questions for each patient
        print("\n=== Step 3: Processing Questions for Patients ===")
        questions_agent = QuestionsProcessingAgent()
        questions_agent.setup()
        
        questions_initial_state = {
            "pdf_filepath": pdf_filepath,
            "mapping_items": [],
            "patient_identifiers": [],
            "patient_questions": [],
            "patient_answers": [],
            "vector_store": None,
            "messages": [
                {"role": "user", "content": f"Process questions for patients in {pdf_name}"}
            ]
        }
        
        try:
            questions_final_state = questions_agent.run(questions_initial_state)
            questions_agent.print_results(questions_final_state)
        except ValueError as e:
            # This will catch the exception thrown when patient identifiers are not found in cache
            print(f"ERROR: {e}")
            print("Note: The questions processing agent requires that patient identifiers have been cached by the patient identifiers agent.")
        
        print("\n=== Workflow Complete ===")
        print(f"PDF: {pdf_name}")
        print(f"Publication Details: {pub_final_state.get('publication_details')}")
        print(f"PMID: {pub_final_state.get('pmid')}")
        print(f"Patient Identifiers: {len(patient_final_state.get('patient_identifiers', []))} patients")
        
        # If questions processing was successful, print the number of patient answers
        if 'questions_final_state' in locals():
            print(f"Patient Answers: {len(questions_final_state.get('patient_answers', []))} rows")
        
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()