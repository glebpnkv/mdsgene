import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from ai.agents.base_agent import BaseAgent
from ai.mapping_item import MappingItem, QuestionInfo
from ai.pdf_text_extractor import PdfTextExtractor
from ai.document_processor import DocumentProcessor
from langchain_community.vectorstores import FAISS

# Define the state for our LangGraph
class State(TypedDict):
    pdf_filepath: str
    mapping_items: List[MappingItem]
    patient_identifiers: List[Dict[str, Optional[str]]]
    patient_questions: List[List[QuestionInfo]]
    patient_answers: List[Dict[str, str]]
    vector_store: Optional[FAISS]
    messages: Annotated[list, add_messages]

class QuestionsProcessingAgent(BaseAgent[State]):
    """Agent for processing questions mapping data from PDFs."""
    
    def __init__(self):
        """Initialize the questions processing agent."""
        super().__init__("questions_processing", "patient_cache.json")
        self.patient_cache_key = "__patient_identifiers_list_v1__"
        self.questions_dir = Path(".questions")
        self.mapping_data_path = self.questions_dir / "mapping_data.json"
        self.vector_store_dir = Path("vector_store/faiss_index")
        
        # Create vector store directory if it doesn't exist
        if not self.vector_store_dir.exists():
            try:
                self.vector_store_dir.mkdir(parents=True, exist_ok=True)
                print(f"Vector store directory created at: {self.vector_store_dir}")
            except Exception as e:
                print(f"Error creating vector store directory: {e}")
    
    def initialize_vector_store(self, state: State) -> State:
        """Initialize Vector Store with the given PDF filepath."""
        pdf_filepath = state["pdf_filepath"]
        
        try:
            # Extract text from PDF
            print("\nExtracting text from PDF...")
            extractor = PdfTextExtractor()
            pdf_text = extractor.extract_text(pdf_filepath)
            
            if pdf_text is None:
                error_msg = "ERROR: Could not extract text from PDF."
                print(error_msg)
                return {
                    **state,
                    "vector_store": None,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": error_msg}
                    ]
                }
            
            print(f"PDF Text extracted successfully ({len(pdf_text)} chars).")
            
            # Process and store the document
            print("\nProcessing document and creating vector store...")
            processor = DocumentProcessor(storage_path=str(self.vector_store_dir))
            processor.process_document(pdf_text, source_filename=Path(pdf_filepath).name)
            
            vector_store = processor.get_vector_store()
            if vector_store:
                # Optional: Check how many vectors are stored
                try:
                    if hasattr(vector_store, 'index') and vector_store.index:
                        print(f"FAISS index contains {vector_store.index.ntotal} vectors.")
                    else:
                        print("Could not determine the number of vectors in the store.")
                    
                    # Test the vector store with a simple query
                    test_query = "What is this document about?"
                    results = vector_store.similarity_search(test_query, k=1)
                    print(f"\nTest search results for '{test_query}': {len(results)}")
                    if results:
                        print(f"Top result snippet: {results[0].page_content[:100]}...")
                except Exception as vs_check_e:
                    print(f"Could not check vector store details: {vs_check_e}")
                
                print("Vector store created successfully.")
                return {
                    **state,
                    "vector_store": vector_store,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": "Vector store created successfully."}
                    ]
                }
            else:
                error_msg = "ERROR: Vector store was not created."
                print(error_msg)
                return {
                    **state,
                    "vector_store": None,
                    "messages": state["messages"] + [
                        {"role": "assistant", "content": error_msg}
                    ]
                }
        except Exception as e:
            error_msg = f"ERROR: Failed to initialize Vector Store: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {
                **state,
                "vector_store": None,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
    
    def load_mapping_data(self, state: State) -> State:
        """Load mapping data from JSON file."""
        if not self.mapping_data_path.exists():
            error_msg = f"ERROR: Mapping data file not found at {self.mapping_data_path}"
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
        
        try:
            with open(self.mapping_data_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            
            mapping_data = []
            for entry in items:
                mapping_data.append(
                    MappingItem(
                        field=entry["field"],
                        question=entry["question"],
                        mapped_excel_column=entry["mapped_excel_column"],
                        response_convertion_strategy=entry["response_convertion_strategy"],
                        custom_processor=None  # We don't have processors in this simplified version
                    )
                )
            
            print(f"Loaded {len(mapping_data)} mapping items from JSON.")
            
            return {
                **state,
                "mapping_items": mapping_data,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": f"Loaded {len(mapping_data)} mapping items from JSON."}
                ]
            }
        except Exception as e:
            error_msg = f"ERROR: Failed to load mapping data: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
    
    def get_patient_identifiers(self, state: State) -> State:
        """Get patient identifiers from cache or throw an exception."""
        cache = self.load_cache()
        loaded_from_cache = False
        
        print("\nGetting patient identifiers (checking cache)...")
        cached_data = cache.get(self.patient_cache_key)
        
        if cached_data is not None:
            print("  Cache HIT for patient identifiers.")
            try:
                if isinstance(cached_data, list):
                    patient_identifiers = cached_data
                    print(f"  Successfully loaded {len(patient_identifiers)} identifiers from cache.")
                    loaded_from_cache = True
                else:
                    print("  WARNING: Cached data for patient identifiers is not a list.")
            except Exception:
                print("  ERROR: Failed to parse cached patient identifiers.")
        
        if not loaded_from_cache:
            error_msg = "ERROR: Patient identifiers not found in cache. This agent requires pre-cached patient identifiers."
            print(error_msg)
            raise ValueError(error_msg)
        
        print(f"--> Proceeding with {len(patient_identifiers)} patient identifiers.")
        
        return {
            **state,
            "patient_identifiers": patient_identifiers,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Found {len(patient_identifiers)} patient identifiers in cache."}
            ]
        }
    
    def generate_patient_questions(self, state: State) -> State:
        """Generate questions for each patient using mapping items."""
        mapping_items = state["mapping_items"]
        patient_identifiers = state["patient_identifiers"]
        
        if not mapping_items:
            error_msg = "ERROR: No mapping items loaded. Cannot generate questions."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
        
        if not patient_identifiers:
            error_msg = "ERROR: No patient identifiers found. Cannot generate questions."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
        
        list_of_patient_question_sets = []
        print(f"\nGenerating questions for {len(patient_identifiers)} unique patients...")
        
        for entry in patient_identifiers:
            patient_id = entry.get("patient")
            family_id = entry.get("family")
            one_patient_questions = []
            
            for item in mapping_items:
                context_prefix = f"Regarding patient '{patient_id}'"
                if family_id:
                    context_prefix += f" from family '{family_id}'"
                specific_query = f"{context_prefix}: {item.question}"
                
                try:
                    q_info = QuestionInfo(
                        field=item.field,
                        query=specific_query,
                        response_convertion_strategy=item.response_convertion_strategy,
                        family_id=family_id,
                        patient_id=patient_id
                    )
                    one_patient_questions.append(q_info)
                except Exception as e:
                    print(f"ERROR: Failed to create QuestionInfo: {e}")
                    continue
            
            list_of_patient_question_sets.append(one_patient_questions)
        
        print(f"Generated question sets for {len(list_of_patient_question_sets)} patients.")
        
        return {
            **state,
            "patient_questions": list_of_patient_question_sets,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Generated question sets for {len(list_of_patient_question_sets)} patients."}
            ]
        }
    
    def process_patient_questions(self, state: State) -> State:
        """Process questions for each patient and get answers."""
        pdf_filepath = state["pdf_filepath"]
        patient_questions = state["patient_questions"]
        vector_store = state.get("vector_store")
        
        if not patient_questions:
            error_msg = "ERROR: No patient question sets generated. Cannot process patients."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
        
        # Initialize Gemini processor
        gemini_processor = self.initialize_gemini(pdf_filepath)
        if not gemini_processor:
            error_msg = "ERROR: Failed to initialize Gemini Processor."
            print(error_msg)
            return {
                **state,
                "messages": state["messages"] + [
                    {"role": "assistant", "content": error_msg}
                ]
            }
        
        # Process each patient's questions
        all_patient_data_rows = []
        print(f"\nProcessing {len(patient_questions)} identified patient sets...")
        
        patient_num = 0
        for patient_question_set in patient_questions:
            patient_num += 1
            if not patient_question_set:
                print(f"\n=== Skipping Patient Set {patient_num} (Empty Question Set) ===")
                continue
            
            current_patient_id = patient_question_set[0].patient_id or "UnknownPatient"
            current_family_id = patient_question_set[0].family_id
            print(f"\n=== Processing Patient Set {patient_num} (Patient: '{current_patient_id}', Family: '{current_family_id or 'N/A'}') ===")
            
            patient_results = {}
            patient_results["family_id"] = current_family_id or "-99"
            patient_results["individual_id"] = current_patient_id
            
            for q_obj in patient_question_set:
                print(f"--- Querying for field: {q_obj.field} (Patient: {current_patient_id}) ---")
                query_text = q_obj.query
                
                # Try to use vector store first if available
                raw_answer = None
                if vector_store:
                    try:
                        print(f"  Using vector store for query: {query_text[:50]}...")
                        results = vector_store.similarity_search(query_text, k=1)
                        if results:
                            raw_answer = results[0].page_content
                            print(f"  Vector store found answer: {raw_answer[:50]}...")
                    except Exception as vs_err:
                        print(f"  ERROR using vector store: {vs_err}. Falling back to Gemini.")
                        raw_answer = None
                
                # Fall back to Gemini if vector store failed or is not available
                if raw_answer is None:
                    try:
                        print(f"  Using Gemini for query: {query_text[:50]}...")
                        raw_answer = gemini_processor.answer_question(query_text)
                        if raw_answer:
                            print(f"  Gemini found answer: {raw_answer[:50]}...")
                        else:
                            print("  Gemini returned no answer.")
                    except Exception as gemini_err:
                        print(f"  ERROR using Gemini: {gemini_err}")
                        raw_answer = None
                
                # Format the answer if we got one
                if raw_answer:
                    try:
                        formatted_answer = gemini_processor.format_answer(raw_answer, q_obj.response_convertion_strategy)
                        patient_results[q_obj.field] = formatted_answer
                        print(f"  Formatted answer: {formatted_answer}")
                    except Exception as format_err:
                        print(f"  ERROR formatting answer: {format_err}")
                        patient_results[q_obj.field] = "-99_FORMAT_ERROR"
                else:
                    patient_results[q_obj.field] = "-99_NO_ANSWER"
            
            all_patient_data_rows.append(patient_results)
        
        print(f"Processed {len(all_patient_data_rows)} patient data rows.")
        
        return {
            **state,
            "patient_answers": all_patient_data_rows,
            "messages": state["messages"] + [
                {"role": "assistant", "content": f"Processed {len(all_patient_data_rows)} patient data rows."}
            ]
        }
    
    def setup(self):
        """Set up the agent by building the graph."""
        nodes = {
            "load_mapping_data": self.load_mapping_data,
            "initialize_vector_store": self.initialize_vector_store,
            "get_patient_identifiers": self.get_patient_identifiers,
            "generate_patient_questions": self.generate_patient_questions,
            "process_patient_questions": self.process_patient_questions
        }
        return self.build_graph(State, nodes)
    
    def print_results(self, final_state: State):
        """Print the results of running the agent."""
        print("\n=== Results ===")
        print(f"PDF: {Path(final_state['pdf_filepath']).name}")
        print(f"Mapping Items: {len(final_state.get('mapping_items', []))} items")
        print(f"Patient Identifiers: {len(final_state.get('patient_identifiers', []))} patients")
        print(f"Patient Answers: {len(final_state.get('patient_answers', []))} rows")
        
        # Display vector store information
        vector_store = final_state.get('vector_store')
        if vector_store:
            try:
                if hasattr(vector_store, 'index') and vector_store.index:
                    print(f"Vector Store: Active with {vector_store.index.ntotal} vectors")
                else:
                    print("Vector Store: Active (vector count unknown)")
            except Exception:
                print("Vector Store: Active (could not determine vector count)")
        else:
            print("Vector Store: Not available")
        
        # Call the base class method to print the conversation
        super().print_results(final_state)

def main():
    """Run the agent if this file is executed directly."""
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
    
    print(f"Processing PDF: {pdf_filepath}")
    
    # Initialize the agent
    agent = QuestionsProcessingAgent()
    agent.setup()
    
    # Initialize the state
    initial_state = {
        "pdf_filepath": pdf_filepath,
        "mapping_items": [],
        "patient_identifiers": [],
        "patient_questions": [],
        "patient_answers": [],
        "vector_store": None,
        "messages": [
            {"role": "user", "content": f"Process questions for patients in {Path(pdf_filepath).name}"}
        ]
    }
    
    try:
        # Run the agent
        final_state = agent.run(initial_state)
        
        # Display the results
        agent.print_results(final_state)
        
    except ValueError as e:
        # This will catch the exception thrown when patient identifiers are not found in cache
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()