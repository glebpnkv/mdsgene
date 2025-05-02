import sys

import pandas as pd
from pathlib import Path
from typing import List, Dict


class ExcelWriter:
    """Utility class for writing data to Excel."""

    @staticmethod
    def write_all_data(filepath: Path, headers: List[str], all_data_rows: List[Dict[str, str]]):
        """
        Writes all collected data rows to an Excel file using pandas.
        If the file exists, new data is appended, otherwise the file is created.

        Args:
            filepath: The path to the output Excel file.
            headers: The ordered list of column headers.
            all_data_rows: A list of dictionaries, where each dictionary represents a row.
        """
        if not all_data_rows:
            print("ExcelWriter: No data rows provided. Nothing to write.")
            return

        try:
            # Check if the file exists
            if filepath.exists():

                # Read the existing data from the file
                existing_data = pd.read_excel(filepath)

                # Create a DataFrame for new data
                new_data = pd.DataFrame(all_data_rows, columns=headers)

                # Combine old and new data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)

                # Write the combined data back to the Excel file
                # Using a simple overwrite approach instead of append mode
                combined_data.to_excel(filepath, index=False, engine='openpyxl')
            else:
                # If the file doesn't exist, create it and write the data
                df = pd.DataFrame(all_data_rows, columns=headers)
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df.to_excel(filepath, index=False, engine='openpyxl')

            print(f"Successfully wrote data to {filepath}")

        except ImportError:
            print("ERROR: pandas or openpyxl library not found. Please install them: pip install pandas openpyxl",
                  file=sys.stderr)
        except Exception as e:
            print(f"ERROR: Failed to write data to Excel file {filepath}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
