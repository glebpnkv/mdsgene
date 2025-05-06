# Document Loader
# This script loads a document into the database

from pathlib import Path
from ai.data_uploader import DataUploader

# Database and document paths
local_db = r"C:\Users\madoev\Desktop\Nairi-New7\MDSGene_backend\vector_store\faiss_index"
document_path = r"D:\000333\fine_tune_deepseek_r1\test_pdf"  # Change path
document_name = "ando2012-22991136.pdf"

def main():
    # Initialize the data uploader
    uploader = DataUploader(local_db)
    
    # Load the document into the database
    uploader.load_document(document_path, document_name)
    
    print(f"Document '{document_name}' has been loaded into the database.")

if __name__ == "__main__":
    main()