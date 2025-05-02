import os
import sys
from pathlib import Path

import excel_mapping_app
from excel_mapping_app import GEMINI_MODEL_NAME

# --- Entry Point ---
if __name__ == "__main__":
    print(f"Script starting execution at {Path(__file__).parent.resolve()}")  # Show working directory

    # Проверка переменной окружения FOLDER_PATH
    folder_path = os.getenv("FOLDER_PATH")
    if not folder_path or not folder_path.strip():
        print("CRITICAL ERROR: FOLDER_PATH environment variable not set.", file=sys.stderr)
        sys.exit(1)

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"CRITICAL ERROR: Specified folder does not exist or is not a directory: '{folder_path}'",
              file=sys.stderr)
        sys.exit(1)

    # Удаление старого файла Excel (опционально)
    output_excel_path = Path(os.getenv("OUTPUT_EXCEL_PATH", ""))
    if output_excel_path.exists():
        print(f"Removing existing output file: {output_excel_path}")
        try:
            output_excel_path.unlink()
        except OSError as e:
            print(f"Could not remove existing file: {e}. Please close it if it's open before running.", file=sys.stderr)
            # Решаем, прерывать ли выполнение - пока не будем
            # sys.exit(1)

    # Обработка файлов в папке
    files = list(folder.iterdir())
    if not files:
        print(f"No files found in the directory: '{folder_path}'")

    for file in files:
        # Process only PDF files
        if not file.suffix.lower() == ".pdf":
            continue

        # Check if the file is readable
        if not os.access(file, os.R_OK):
            print(f"Skipping unreadable file: {file}", file=sys.stderr)
            continue

        print(f"Creating ExcelMappingApp for PDF: {file}")
        app = excel_mapping_app.ExcelMappingApp(pdf_filepath=file, model_name=GEMINI_MODEL_NAME)
        app.run()
