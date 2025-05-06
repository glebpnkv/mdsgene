import os
import sys
import argparse
from pathlib import Path

from ai.document_processor import DocumentProcessor
from ai.pdf_text_extractor import PdfTextExtractor

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

from ai.data_query import DataQuery
from ai import excel_mapping_app
from ai.excel_mapping_app import ExcelMappingApp

def test_excel_mapping(pdf_path, output_dir=None, model_name="gemini-1.5-flash"):
    """
    Test the ExcelMappingApp with a PDF input and generate an Excel file.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the Excel file. Defaults to None.
        model_name (str, optional): Name of the Gemini model to use. Defaults to "gemini-1.5-flash".

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert to Path objects
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            return False

        # Set up output directory
        if output_dir:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                print(f"Creating output directory: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default to a tables directory in the current directory
            output_dir = parent_dir / ".tables"
            if not output_dir.exists():
                print(f"Creating default output directory: {output_dir}")
                output_dir.mkdir(parents=True, exist_ok=True)

        # Set up the output Excel path
        output_excel_path = output_dir / f"{pdf_path.stem}_output.xlsx"

        # Ensure the output directory exists
        output_excel_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Processing PDF: {pdf_path}")
        print(f"Output Excel will be saved to: {output_excel_path}")

        # Initialize DataQuery for full text retrieval
        storage_path = parent_dir / "vector_store" / "faiss_index"
        document_processor = DocumentProcessor(storage_path=str(storage_path)) # Initialize with the storage path
        pdf_text_extractor = PdfTextExtractor()
        pdf_text = pdf_text_extractor.extract_text(pdf_path)

        if pdf_text is None:
            print("Error: Could not extract text from PDF.")
            return False

        # Process the document and store it in the vector store
        document_processor.process_document(pdf_text, source_filename=pdf_path.name)
        vector_store = document_processor.get_vector_store()

        test_query = "List all *motor* symptoms observed in the patient and state if they were present (yes/no). Format as 'Symptom Name: yes/no', separated by semicolons or newlines. Examples: Rigidity: yes; Tremor: no; Bradykinesia: yes."
        results = vector_store.similarity_search(test_query, k=1)
        print(f"\nTest search results for '{test_query}': {len(results)}")

        # Initialize and run ExcelMappingApp
        app = ExcelMappingApp(
            pdf_filepath=pdf_path,
            model_name=model_name,
            vector_store=vector_store
        )

        # Store the original output path
        original_output_path = excel_mapping_app.OUTPUT_EXCEL_PATH

        try:
            # Override the default output path
            excel_mapping_app.OUTPUT_EXCEL_PATH = output_excel_path

            # Run the application
            app.run()
        finally:
            # Restore the original output path
            excel_mapping_app.OUTPUT_EXCEL_PATH = original_output_path

        if output_excel_path.exists():
            print(f"Excel file successfully created: {output_excel_path}")
            return True
        else:
            print(f"Error: Excel file was not created at {output_excel_path}")
            print(f"Current directory: {os.getcwd()}")
            print(f"Output directory exists: {output_dir.exists()}")
            print(f"Output directory is writable: {os.access(str(output_dir), os.W_OK)}")
            print(f"Output path parent exists: {output_excel_path.parent.exists()}")
            print(f"Output path parent is writable: {os.access(str(output_excel_path.parent), os.W_OK)}")
            return False

    except Exception as e:
        import traceback
        print(f"Error in test_excel_mapping: {e}")
        traceback.print_exc()
        return False

def main():
    """
    Main function to parse command-line arguments and run the test.
    """
    parser = argparse.ArgumentParser(description="Test ExcelMappingApp with a PDF input")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output-dir", "-o", help="Directory to save the Excel file")
    parser.add_argument("--model", "-m", default="gemini-1.5-flash", help="Name of the Gemini model to use")

    args = parser.parse_args()

    success = test_excel_mapping(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        model_name=args.model
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
