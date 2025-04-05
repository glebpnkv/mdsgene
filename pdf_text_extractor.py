import sys
from pathlib import Path

from pypdf import PdfReader


class PdfTextExtractor:
    """Extracts text content from a PDF file using pypdf."""

    def extract_text(self, pdf_filepath: str | Path) -> str | None:
        """
        Extracts text content from a PDF file.

        Args:
            pdf_filepath: The path to the PDF file.

        Returns:
            The extracted text content as a single string, or None if an error occurs.
        """
        pdf_file = Path(pdf_filepath)
        if not pdf_file.exists():
            print(f"Error: PDF file not found at {pdf_filepath}", file=sys.stderr)
            return None
        if not pdf_file.is_file():
            print(f"Error: Path {pdf_filepath} is not a file.", file=sys.stderr)
            return None

        try:
            reader = PdfReader(pdf_file)

            if reader.is_encrypted:
                # pypdf can sometimes handle basic encryption without a password,
                # but will raise an error if a password is required.
                # Attempting decryption might be needed for some files.
                try:
                    # Try decrypting with an empty password, might work for some
                    reader.decrypt("")
                except Exception as decrypt_error:
                    print(
                        f"Error: Cannot process encrypted PDF without password: {decrypt_error}",
                        file=sys.stderr
                    )
                    # Or try password handling here if needed
                    return None

            full_text = ""
            for page in reader.pages:
                try:
                    full_text += page.extract_text() + "\n"  # Add newline between pages
                except Exception as page_error:
                    print(f"Warning: Could not extract text from a page: {page_error}", file=sys.stderr)

            # Basic cleaning (optional)
            full_text = '\n'.join(line.strip() for line in full_text.splitlines() if line.strip())

            return full_text

        except Exception as e:
            print(f"Error loading or reading PDF: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None


# --- Example Usage ---
if __name__ == "__main__":
    # IMPORTANT: Replace with the actual path to YOUR PDF file
    # Use raw string (r"...") or forward slashes for paths
    file_path = r"D:\000333\fine_tune_deepseek_r1\test_pdf\ando2012-22991136.pdf"

    if not Path(file_path).exists():
        print(f"ERROR: Test PDF file not found at: {file_path}")
    else:
        extractor = PdfTextExtractor()
        extracted_text = extractor.extract_text(file_path)

        if extracted_text:
            print("Successfully extracted text:")
            print("------------------------------------")
            # Print only the first 1000 characters
            limit = min(len(extracted_text), 1000)
            print(extracted_text[:limit] + "...")
            print("------------------------------------")
            print(f"(Text extraction complete - {len(extracted_text)} characters)")
        else:
            print("Failed to extract text from PDF.", file=sys.stderr)
