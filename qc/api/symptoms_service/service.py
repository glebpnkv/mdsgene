from typing import List, Dict, Any, Optional, final
from pydantic import BaseModel
import json
import os
from qc.logging_config import logger
from qc.api.gene.list_genes import get_file_id_from_disease_gene
from qc.config import properties_directory


class SymptomOrder(BaseModel):
    geneName: str
    symptomName: str
    categoryName: str
    order: int


def update_symptom_order(symptoms: List[SymptomOrder]) -> bool:
    """
    Обновляет порядок симптомов в JSON файле.
    """
    try:
        if not symptoms:
            return True

        # Используем имя файла из первого симптома, так как файл один
        file_name = symptoms[0].geneName
        json_file = os.path.join(properties_directory, file_name)

        # Читаем текущее содержимое файла
        try:
            with open(json_file, 'r') as file:
                current_data = json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {json_file}")
            return False

        # Создаем новую структуру данных
        updated_data = {category: {} for category in current_data.keys()}

        # Создаем маппинг старых значений для симптомов
        symptom_values = {}
        for category, symptoms_dict in current_data.items():
            for symptom_name, value in symptoms_dict.items():
                symptom_values[symptom_name] = value

        # Обрабатываем симптомы с заданным порядком
        symptoms_processed = set()
        for category in current_data.keys():
            category_symptoms = []

            # Собираем симптомы для текущей категории
            for symptom in symptoms:
                if symptom.categoryName == category:
                    category_symptoms.append((
                        symptom.symptomName,
                        symptom_values.get(symptom.symptomName, symptom.symptomName),
                        symptom.order
                    ))
                    symptoms_processed.add(symptom.symptomName)

            # Сортируем по порядку
            category_symptoms.sort(key=lambda x: x[2])

            # Добавляем в обновленные данные
            for name, value, _ in category_symptoms:
                updated_data[category][name] = value

        # Добавляем оставшиеся симптомы
        for category, symptoms_dict in current_data.items():
            for symptom_name, value in symptoms_dict.items():
                if symptom_name not in symptoms_processed:
                    # Проверяем, был ли симптом перемещен
                    moved = False
                    for symptom in symptoms:
                        if symptom.symptomName == symptom_name:
                            updated_data[symptom.categoryName][symptom_name] = value
                            moved = True
                            break
                    if not moved:
                        updated_data[category][symptom_name] = value

        # Записываем обновленные данные
        try:
            with open(json_file, 'w') as file:
                json.dump(updated_data, file, indent=2)
        except Exception as e:
            logger.error(f"Error writing to file {json_file}: {str(e)}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error in update_symptom_order: {str(e)}")
        return False

def get_symptoms_for_gene(file_name: str) -> Dict[str, Dict[str, str]]:
    """
    Читает JSON файл с симптомами
    """
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_name}")
        raise Exception(f"File not found: {file_name}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in file: {file_name}")
        raise Exception(f"Invalid JSON format in file: {file_name}")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise Exception("Internal server error")


def get_categories_with_nested_keys(file_name: str) -> Dict[str, List[str]]:
    """
    Возвращает словарь с категориями и их ключами
    """
    try:
        data = get_symptoms_for_gene(file_name)
        result = {}
        for category, nested_dict in data.items():
            result[category] = list(nested_dict.keys())
        return result
    except Exception as e:
        logger.error(f"Error in get_categories_with_nested_keys: {str(e)}")
        raise