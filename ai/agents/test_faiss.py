# --- Step 1: Get Text (using previous class) ---
import sys
from pathlib import Path

from ai.document_processor import DocumentProcessor
from ai.pdf_text_extractor import PdfTextExtractor

file_path = r"D:\000333\fine_tune_deepseek_r1\test_pdf\ando2012-22991136.pdf" # Change path
local_db = r"C:\Users\madoev\Desktop\Nairi-New7\MDSGene_backend\vector_store\faiss_index" # Change path

document_name = Path(file_path).name

if not Path(file_path).exists():
     print(f"ERROR: Test PDF file not found at: {file_path}")
else:
    extractor = PdfTextExtractor()
    pdf_text = extractor.extract_text(file_path)

    if pdf_text is None:
        print("Exiting: Could not extract text from PDF.", file=sys.stderr)
    else:
        print(f"PDF Text extracted successfully ({len(pdf_text)} chars).")

        # --- Step 2: Process and Store ---
        try:
            processor = DocumentProcessor(storage_path=local_db)
            processor.process_document(pdf_text, source_filename=Path(file_path).name)

            vector_store = processor.get_vector_store()
            if vector_store:
                # Optional: Check how many vectors are stored
                try:
                    # FAISS index might be accessible via `vector_store.index`
                    if hasattr(vector_store, 'index') and vector_store.index:
                         print(f"FAISS index contains {vector_store.index.ntotal} vectors.")
                    else:
                         print("Could not determine the number of vectors in the store.")
                    # You can perform a test similarity search
                    test_query = "List all *motor* symptoms observed in the patient and state if they were present (yes/no). Format as 'Symptom Name: yes/no', separated by semicolons or newlines. Examples: Rigidity: yes; Tremor: no; Bradykinesia: yes."
                    results = vector_store.similarity_search(test_query, k=1)
                    print(f"\nTest search results for '{test_query}': {len(results)}")
                    if results:
                        print(f"Top result metadata: {results[0].metadata}")
                        print(f"Top result snippet: {results[0].page_content[:200]}...")

                except Exception as vs_check_e:
                     print(f"Could not check vector store details: {vs_check_e}")

                print("Processing finished. Embeddings should be in the in-vector store.")
            else:
                print("Processing finished, but vector store was not created.", file=sys.stderr)

        except Exception as proc_e:
             print(f"Failed to initialize or run DocumentProcessor: {proc_e}", file=sys.stderr)
