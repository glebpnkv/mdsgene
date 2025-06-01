import json
import os

from pydantic import BaseModel

from mdsgene.qc.config import properties_directory
from mdsgene.qc.logging_config import logger


class SymptomOrder(BaseModel):
    geneName: str
    symptomName: str
    categoryName: str
    order: int


def update_symptom_order(symptoms: list[SymptomOrder]) -> bool:
    """
    Updates the order of symptoms in the JSON file.
    """
    try:
        if not symptoms:
            return True

        # Using file name from the first symptom as there is only one file
        file_name = symptoms[0].geneName
        json_file = os.path.join(properties_directory, file_name)

        # Read current file content
        try:
            with open(json_file, 'r') as file:
                current_data = json.load(file)
        except FileNotFoundError:
            logger.error(f"File not found: {json_file}")
            return False

        # Create new data structure
        updated_data = {category: {} for category in current_data.keys()}

        # Create mapping of old values for symptoms
        symptom_values = {}
        for category, symptoms_dict in current_data.items():
            for symptom_name, value in symptoms_dict.items():
                symptom_values[symptom_name] = value

        # Process symptoms with specified order
        symptoms_processed = set()
        for category in current_data.keys():
            category_symptoms = []

            # Collect symptoms for current category
            for symptom in symptoms:
                if symptom.categoryName == category:
                    category_symptoms.append((
                        symptom.symptomName,
                        symptom_values.get(symptom.symptomName, symptom.symptomName),
                        symptom.order
                    ))
                    symptoms_processed.add(symptom.symptomName)

            # Sort by order
            category_symptoms.sort(key=lambda x: x[2])

            # Add to updated data
            for name, value, _ in category_symptoms:
                updated_data[category][name] = value

        # Add remaining symptoms
        for category, symptoms_dict in current_data.items():
            for symptom_name, value in symptoms_dict.items():
                if symptom_name not in symptoms_processed:
                    # Check if symptom was moved
                    moved = False
                    for symptom in symptoms:
                        if symptom.symptomName == symptom_name:
                            updated_data[symptom.categoryName][symptom_name] = value
                            moved = True
                            break
                    if not moved:
                        updated_data[category][symptom_name] = value

        # Write updated data
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

def get_symptoms_for_gene(file_name: str) -> dict[str, dict[str, str]]:
    """
    Reads JSON file with symptoms
    """
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"File not found: {file_name}")
        raise Exception(f"File not found: {file_name}")
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in file: {file_name}: {e}")
        raise Exception(f"Invalid JSON format in file: {file_name}: {e}")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise Exception("Internal server error")


def get_categories_with_nested_keys(file_name: str) -> dict[str, list[str]]:
    """
    Returns a dictionary with categories and their keys
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
