import os
import json

def read_json_to_dict(file_path: str) -> dict:
    """
    Reads a JSON file and converts it to a Python dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The dictionary representation of the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return {}