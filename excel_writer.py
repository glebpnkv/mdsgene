import sys
import os
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

from mapping_item import MappingItem # Assuming MappingItem is defined here or imported

class ExcelWriter:
    """Writes or appends data to an Excel file using pandas."""

    @staticmethod
    def write_or_append_excel(
        filepath: str | Path,
        mapping_items: List[MappingItem],
        results_map: Dict[str, str]
    ):
        """
        Writes or appends a row of results to an Excel sheet.

        The sheet uses field names from mappingItems as ordered headers.
        If the file doesn't exist, it's created with headers and the first data row.
        If the file exists, a new data row is appended, ensuring columns match.

        Args:
            filepath: The path to the Excel file (.xlsx).
            mapping_items: The list of mapping items, used for ordered headers.
            results_map: A dictionary where keys are field names (String)
                         and values are the corresponding answers (String) for the current row.
        """
        filepath = Path(filepath)
        print(f"Attempting to write/append Excel file: {filepath}")

        # Extract ordered headers from mappingItems based on the 'field' attribute
        headers = [item.field for item in mapping_items]
        if not headers:
             print("Error: No headers provided from mapping items. Cannot write Excel.", file=sys.stderr)
             return

        # Prepare the data for the new row as a DataFrame
        # Ensure the DataFrame has columns in the same order as headers
        new_row_data = {header: [results_map.get(header, "")] for header in headers} # Use list for single row DataFrame
        new_row_df = pd.DataFrame(new_row_data, columns=headers) # Explicitly set column order

        try:
            if filepath.exists():
                # --- File Exists: Read, Append, Write ---
                print("File exists. Reading existing data...")
                try:
                    # Read the first sheet by default
                    existing_df = pd.read_excel(filepath, sheet_name=0, engine='openpyxl')
                    print(f"Read {len(existing_df)} existing rows.")

                    # Basic header validation (optional but recommended)
                    if list(existing_df.columns) != headers:
                        print("Warning: Existing file header doesn't match expected fields.", file=sys.stderr)
                        print(f"  Existing: {list(existing_df.columns)}")
                        print(f"  Expected: {headers}")
                        print("  Appending data based on expected headers. Check file for consistency.")
                        # Force columns to match headers, filling missing with NaN/blank
                        # This can be risky if columns are fundamentally different.
                        # A safer approach might be to raise an error or create a new file.
                        # For now, we proceed, aligning the new row only.

                    # Append the new row
                    # Ensure consistent data types if possible (pandas often handles this)
                    # Use concat as append is deprecated
                    combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
                    next_row_num = len(combined_df) # Row number is index + 1

                except Exception as read_err:
                     print(f"Error reading existing Excel file '{filepath}': {read_err}", file=sys.stderr)
                     print("Attempting to overwrite the file with new data.")
                     # Fallback to writing new file if read fails
                     combined_df = new_row_df
                     next_row_num = 1 # First data row

                print(f"Appending data to row index: {next_row_num -1} (Excel row {next_row_num+1} incl. header)")
                # Write the combined data back
                combined_df.to_excel(filepath, index=False, engine='openpyxl')

            else:
                # --- File Doesn't Exist: Create new file with header and data ---
                print("File does not exist. Creating new workbook with headers...")
                # Write the new DataFrame (which includes the first data row)
                new_row_df.to_excel(filepath, index=False, engine='openpyxl')
                print("Created header row and added first data row at index 0.")

            print("Excel file saved successfully!")

        except ImportError:
             print("Error: 'openpyxl' library not found. Please install it: pip install openpyxl", file=sys.stderr)
        except PermissionError:
             print(f"Error: Permission denied to write file '{filepath}'. Is it open in another program?", file=sys.stderr)
        except Exception as e:
            print(f"Error writing Excel file: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

# --- Example Usage ---
if __name__ == "__main__":
    output_path = Path("./Patient_Data_Appended_Python.xlsx")

    # Example Mapping Items (ensure 'field' matches keys in results)
    example_headers = [
        MappingItem(field='pmid', question='q1', mapped_excel_column='A', response_convertion_strategy='s1'),
        MappingItem(field='Identified_Patient_ID', question='q_id', mapped_excel_column='B', response_convertion_strategy='s_id'),
        MappingItem(field='aao', question='q2', mapped_excel_column='C', response_convertion_strategy='s2'),
        MappingItem(field='sex', question='q3', mapped_excel_column='D', response_convertion_strategy='s3'),
    ]

    # Example data for two rows
    results1 = {
        'pmid': '12345',
        'Identified_Patient_ID': 'Patient_1',
        'aao': '45',
        'sex': 'M',
        'extra_field': 'ignore_me' # Field not in headers
    }
    results2 = {
        'pmid': '12345', # Same study
        'Identified_Patient_ID': 'Patient_II-b',
        'aao': '60',
        'sex': 'F'
        # 'sex': None # Example of missing data
    }

    # Clean up previous test file if it exists
    if output_path.exists():
        print(f"Removing existing test file: {output_path}")
        try:
             output_path.unlink()
        except OSError as e:
             print(f"Could not remove existing file: {e}", file=sys.stderr)


    print("\n--- Writing first row ---")
    ExcelWriter.write_or_append_excel(output_path, example_headers, results1)

    print("\n--- Appending second row ---")
    ExcelWriter.write_or_append_excel(output_path, example_headers, results2)

    # Verify content (optional)
    if output_path.exists():
        try:
             df = pd.read_excel(output_path)
             print(f"\n--- Verifying Excel Content ({output_path}) ---")
             print(df.to_string())
             print("------------------------------------------")
        except Exception as read_e:
            print(f"Could not read back Excel file for verification: {read_e}")
    else:
        print(f"Excel file '{output_path}' was not created.")