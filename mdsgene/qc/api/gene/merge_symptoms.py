import json
import os

import pandas as pd
from pydantic import BaseModel

from mdsgene.qc.api.gene.list_genes import get_file_id_from_disease_gene
from mdsgene.qc.config import properties_directory
from mdsgene.qc.logging_config import logger


class MergeSymptomRequest(BaseModel):
    geneName: str
    mergedSymptomName: str
    symptomsToMerge: list[str]


def merge_symptoms(data: MergeSymptomRequest) -> bool:
    try:
        # Get the file path
        file_id = get_file_id_from_disease_gene(data.geneName)
        if not file_id:
            logger.error(f"No file ID found for gene '{data.geneName}'")
            return False

        # Use properties_directory to determine the path
        json_file = os.path.join(properties_directory, f"symptom_categories_{file_id}.json")

        # Read current data
        with open(json_file, 'r') as file:
            categories_data = json.load(file)

        # Convert data to DataFrame for easier processing
        rows = []
        for category, symptoms in categories_data.items():
            for symptom, value in symptoms.items():
                rows.append({
                    'category': category,
                    'symptom': symptom,
                    'value': value
                })
        df = pd.DataFrame(rows)

        # Find the first category where any of the symptoms to merge appears
        target_category = None
        for symptom in data.symptomsToMerge:
            mask = df['symptom'].isin([symptom])
            if mask.any():
                target_category = df.loc[mask.idxmax(), 'category']
                break

        if not target_category:
            target_category = next(iter(categories_data.keys()))

        # Create new data structure
        new_categories = {category: {} for category in categories_data.keys()}

        # Copy all symptoms except those being merged
        for index, row in df.iterrows():
            if row['symptom'] not in data.symptomsToMerge:
                new_categories[row['category']][row['symptom']] = row['value']

        # Add new merged symptom
        # Use the first symptom's value as the base for the new one
        base_value = df[df['symptom'].isin(data.symptomsToMerge)]['value'].iloc[0]
        new_categories[target_category][data.mergedSymptomName] = base_value

        # Save updated data
        with open(json_file, 'w') as file:
            json.dump(new_categories, file, indent=2)

        return True
    except Exception as e:
        logger.error(f"Error in merge_symptoms: {str(e)}")
        return False
