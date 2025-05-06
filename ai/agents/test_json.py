import json
import os


# load json data
def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Loads JSON data from a file.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


#