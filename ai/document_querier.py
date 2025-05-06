# Document Querier
# This script queries the database with questions and generates responses

from ai.data_query import DataQuery

# Database path
local_db = r"C:\Users\madoev\Desktop\Nairi-New7\MDSGene_backend\vector_store\faiss_index"
document_name = "ando2012-22991136.pdf"

def main():
    # Initialize the data query
    querier = DataQuery(local_db)
    
    # Define the question
    question = """List all *motor* symptoms observed in the patient and state if they were present (yes/no). 
Format as 'Symptom Name: yes/no', separated by semicolons or newlines. 
Examples: Rigidity: yes; Tremor: no; Bradykinesia: yes."""
    
    # Search for content in the database
    context = querier.search_content(question, document_name)
    
    # Check if context is empty
    if not context:
        print("No relevant documents found.")
        return
    
    # Print the search results
    print("Search results:")
    print(context)
    
    # Generate a response to the question
    response_question = "Provide a list of motor symptoms and their status (e.g., 'Rigidity: yes; Tremor: no'). If none are mentioned or present, state 'None'."
    result = querier.answer_query(context, response_question)
    
    # Print the response
    print("\nResponse:")
    print(result)

if __name__ == "__main__":
    main()