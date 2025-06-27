# qc/api/files.py
import os


def get_excel_files_list(directory: str = 'excel') -> list[str]:
    # Get list of files in the directory
    files = os.listdir(directory)

    # Filter files: keep only those that end with .xlsx and don't start with '~'
    return [file for file in files if file.endswith(".xlsx") and not file.startswith("~") and not file.startswith(".")]
