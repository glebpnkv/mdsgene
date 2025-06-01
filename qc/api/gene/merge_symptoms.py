from typing import List

from fastapi import HTTPException
from pydantic import BaseModel
import pandas as pd
import json
import os

from qc.api.gene.list_genes import get_file_id_from_disease_gene
from qc.logging_config import logger
from qc.config import properties_directory

class MergeSymptomRequest(BaseModel):
    geneName: str
    mergedSymptomName: str
    symptomsToMerge: List[str]


def merge_symptoms(data: MergeSymptomRequest) -> bool:
    try:
        # Получаем путь к файлу
        file_id = get_file_id_from_disease_gene(data.geneName)
        if not file_id:
            logger.error(f"No file ID found for gene '{data.geneName}'")
            return False

        # Use properties_directory to determine the path
        json_file = os.path.join(properties_directory, f"symptom_categories_{file_id}.json")

        # Читаем текущие данные
        with open(json_file, 'r') as file:
            categories_data = json.load(file)

        # Преобразуем данные в DataFrame для удобства обработки
        rows = []
        for category, symptoms in categories_data.items():
            for symptom, value in symptoms.items():
                rows.append({
                    'category': category,
                    'symptom': symptom,
                    'value': value
                })
        df = pd.DataFrame(rows)

        # Находим первую категорию, где встречается любой из объединяемых симптомов
        target_category = None
        for symptom in data.symptomsToMerge:
            mask = df['symptom'].isin([symptom])
            if mask.any():
                target_category = df.loc[mask.idxmax(), 'category']
                break

        if not target_category:
            target_category = next(iter(categories_data.keys()))

        # Создаем новую структуру данных
        new_categories = {category: {} for category in categories_data.keys()}

        # Копируем все симптомы, кроме тех, что объединяются
        for index, row in df.iterrows():
            if row['symptom'] not in data.symptomsToMerge:
                new_categories[row['category']][row['symptom']] = row['value']

        # Добавляем новый объединенный симптом
        # Используем значение первого симптома как основу для нового
        base_value = df[df['symptom'].isin(data.symptomsToMerge)]['value'].iloc[0]
        new_categories[target_category][data.mergedSymptomName] = base_value

        # Сохраняем обновленные данные
        with open(json_file, 'w') as file:
            json.dump(new_categories, file, indent=2)

        return True
    except Exception as e:
        logger.error(f"Error in merge_symptoms: {str(e)}")
        return False
