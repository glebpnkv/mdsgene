# Main entry point for the application
# This script demonstrates how to use document_loader.py and document_querier.py

import sys
import os

def main():
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Add the current directory to the Python path
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    # Import the document loader and querier modules
    from document_loader import main as load_document
    from document_querier import main as query_document

    # Load the document into the database
    print("Loading document into the database...")
    load_document()

    # Query the database with questions
    print("\nQuerying the database with questions...")
    query_document()

if __name__ == "__main__":
    main()
