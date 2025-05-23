# qc/api/files.py
import os
from typing import List

def get_excel_files_list(directory: str = 'excel') -> List[str]:
    # Получаем список файлов в каталоге
    files = os.listdir(directory)

    # Фильтруем файлы: оставляем только те, что заканчиваются на .xlsx и не начинаются с '~'
    return [file for file in files if file.endswith(".xlsx") and not file.startswith("~") and not file.startswith(".")]
