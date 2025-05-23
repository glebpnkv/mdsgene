# qc/file_operations.py
import os
from fastapi import UploadFile

async def update_excel_file_content(fileId: str, newFile: UploadFile):
    directory = "excel"
    file_path = os.path.join(directory, fileId)

    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    # Write the new file content
    with open(file_path, "wb") as f:
        content = await newFile.read()
        f.write(content)
