import sys
from pathlib import Path
from pypdf import PdfReader
import re


class PdfTextExtractorLogic:
    """Extracts text content from a PDF file using pypdf with enhanced table handling."""

    # pip install tabula-py pdfplumber

    def extract_text(self, pdf_filepath: str | Path) -> str | None:
        """
        Extracts text content from a PDF file with improved handling for tables and special characters.

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
                try:
                    reader.decrypt("")
                except Exception as decrypt_error:
                    print(
                        f"Error: Cannot process encrypted PDF without password: {decrypt_error}",
                        file=sys.stderr
                    )
                    return None

            full_text = ""
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    # Preserve table structures by normalizing whitespace
                    page_text = self._normalize_table_text(page_text)
                    full_text += page_text + "\n"  # Add newline between pages
                except Exception as page_error:
                    print(f"Warning: Could not extract text from a page: {page_error}", file=sys.stderr)

            # Process special characters
            full_text = self._process_special_characters(full_text)

            return full_text

        except Exception as e:
            print(f"Error loading or reading PDF: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return None

    def _normalize_table_text(self, text: str) -> str:
        """
        Normalize text from tables to better preserve structure.
        """
        # Split by lines
        lines = text.splitlines()
        processed_lines = []

        for line in lines:
            if line.strip():
                # Normalize spaces in each line (replace multiple spaces with single)
                normalized = re.sub(r'\s{2,}', ' ', line.strip())
                processed_lines.append(normalized)

        # Join lines preserving empty lines that might indicate table row separations
        return '\n'.join(processed_lines)

    def _process_special_characters(self, text: str) -> str:
        """
        Convert special characters to more standard representation.
        Focus on medical symbols commonly used in clinical documents.
        """
        # Replace medical symbols with human-readable text
        text = text.replace("þ", "YES")
        text = text.replace("/C0", "NO")

        # Additional character replacements for medical notation
        text = re.sub(r'[^\x00-\x7F]+', lambda m: self._replace_unicode(m.group(0)), text)

        return text

    def _replace_unicode(self, char: str) -> str:
        """Map known unicode characters to their meaning in medical context"""
        chars_map = {
            'þ': ' YES ',
            '±': ' PARTIAL ',
            '−': ' NEGATIVE ',
            '✓': ' YES ',
            '✗': ' NO ',
            '✘': ' NO ',
            '⊕': ' POSITIVE ',
            '⊖': ' NEGATIVE '
            # Add more mappings as needed
        }
        return chars_map.get(char, char)