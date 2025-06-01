import os
import json
from typing import List, Optional
from qc.config import properties_directory


def get_symptom_gene_mapping_files(directory: str = None) -> List[str]:
    if directory is None:
        directory = properties_directory

    try:
        # Получаем список всех файлов в директории
        files = os.listdir(directory)
        return sorted(files)
    except Exception as e:
        print(f"Error reading directory: {str(e)}")
        return []


def get_file_id_from_disease_gene(disease_gene_pair: str, file_name: str = "disease_gene_mapping.json",
                                  directory: str = None) -> Optional[str]:
    if directory is None:
        directory = properties_directory

    file_path = os.path.join(directory, file_name)

    with open(file_path, "r") as f:
        data = json.load(f)

    for pair_id, pair_data in data["pairs"].items():
        disease = pair_data["disease"]
        gene = pair_data["gene"]
        if f"{disease}|{gene}" == disease_gene_pair:
            return pair_data["id"]

    return None
