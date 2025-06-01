import json
import pandas as pd
import os
from typing import Dict, Set, Tuple
from qc.logging_config import logger
from qc.api.gene.utils import get_cached_dataframe
from qc.config import properties_directory


def get_or_create_categories() -> Dict:
    """Creates the basic structure of categories."""
    return {
        "Development in childhood/adolescence": {},
        "Imaging features": {},
        "Laboratory results": {},
        "Motor ictal": {},
        "Motor interictal": {},
        "Motor signs and symptoms": {},
        "Non-motor ictal": {},
        "Non-motor interictal": {},
        "Non-motor signs and symptoms": {},
        "Other ictal": {},
        "Other interictal": {},
        "Other signs and symptoms": {},
        "Paroxysmal movements": {},
        "Therapy": {},
        "Triggers": {},
        "Unknown": {}
    }


def update_symptom_categories(file_path: str, properties_dir: str = None) -> None:
    """
    Updates symptom categories for a given file.

    Args:
        file_path (str): Path to the Excel file.
        properties_dir (str): Directory for storing category files.
    """
    if properties_dir is None:
        properties_dir = properties_directory

    os.makedirs(properties_dir, exist_ok=True)

    disease_gene_symptoms: Dict[str, Set[Tuple[str, str]]] = {}

    try:
        df = get_cached_dataframe(file_path)
        logger.info(f"Processing file: {os.path.basename(file_path)}")
        print(f"\nProcessing file: {os.path.basename(file_path)}")
        logger.info(f"Initial DataFrame shape: {df.shape}")
        print(f"Initial DataFrame shape: {df.shape}")

        # Filter rows with "mdsgene_decision" == "IN"
        df = df[df["mdsgene_decision"] == "IN"]

        # Identify relevant symptom columns
        symptom_cols = df.columns[
            df.columns.str.contains('_sympt$|_HP:', case=False, na=False)
        ]

        for _, row in df.iterrows():
            disease = row.get('disease_abbrev')
            if not isinstance(disease, str) or not disease:
                continue

            for gene_col in ['gene1', 'gene2', 'gene3']:
                gene = row.get(gene_col)
                if not isinstance(gene, str) or not gene:
                    continue

                disease_gene_key = f"{disease}_{gene}"
                file_id = f"symptom_categories_{disease}_{gene}.json"

                if disease_gene_key not in disease_gene_symptoms:
                    disease_gene_symptoms[disease_gene_key] = set()

                for col in symptom_cols:
                    symptom_name = col.replace('_sympt', '').replace('_HP:', '')
                    display_name = ' '.join(word.capitalize() for word in symptom_name.split('_'))
                    disease_gene_symptoms[disease_gene_key].add((symptom_name, display_name))

    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        logger.exception("Detailed error:")
        return

    for disease_gene_key, symptoms in disease_gene_symptoms.items():
        file_name = f"symptom_categories_{disease_gene_key}.json"
        file_path = os.path.join(properties_dir, file_name)

        categories = get_or_create_categories()
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                categories = json.load(f)
            logger.info(f"Existing categories loaded from: {file_path}")
            print(f"Existing categories loaded from: {file_path}")

        for symptom_name, display_name in symptoms:
            if not any(symptom_name in cat for cat in categories.values()):
                categories["Unknown"][symptom_name] = display_name

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(categories, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated categories file: {file_path}; Total symptoms: {len(symptoms)}")
            print(f"Updated categories file: {file_path}")
            print(f"Total symptoms: {len(symptoms)}")
        except Exception as e:
            logger.error(f"Failed to write JSON file {file_path}: {e}")
            print(f"Failed to write JSON file {file_path}: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python update_symptom_categories.py <excel_file_path>")
        sys.exit(1)

    excel_file_path = sys.argv[1]
    update_symptom_categories(excel_file_path)
    logger.info(f"Finished updating symptom categories for file: {excel_file_path}")
    print(f"Finished updating symptom categories for file: {excel_file_path}")
