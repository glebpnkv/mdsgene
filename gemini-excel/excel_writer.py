# excel_writer.py
import pandas as pd
from pathlib import Path
from typing import List, Dict

class ExcelWriter:
    """Utility class for writing data to Excel."""

    @staticmethod
    def write_all_data(filepath: Path, headers: List[str], all_data_rows: List[Dict[str, str]]):
        """
        Writes all collected data rows to an Excel file using pandas.

        Args:
            filepath: The path to the output Excel file.
            headers: The ordered list of column headers.
            all_data_rows: A list of dictionaries, where each dictionary represents a row.
        """
        if not all_data_rows:
            print("ExcelWriter: No data rows provided. Nothing to write.")
            return

        try:
            # Create DataFrame ensuring columns are in the specified order
            # Fill missing values with a placeholder (e.g., empty string or '-99')
            df = pd.DataFrame(all_data_rows)

            # Reindex to ensure all headers are present and in order
            # Use a fill_value suitable for your data, maybe '' or '-99'
            df = df.reindex(columns=headers, fill_value='-99')

            # Write to Excel
            # Make sure the directory exists if filepath includes directories
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(filepath, index=False, engine='openpyxl')

            print(f"ExcelWriter: Data successfully written to {filepath}")

        except ImportError:
            print("ERROR: pandas or openpyxl library not found. Please install them: pip install pandas openpyxl", file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Failed to write data to Excel file {filepath}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()