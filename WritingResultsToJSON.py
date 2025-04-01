import json
import os

def save_dict_to_json(data, file_path):
    """
    Saves a Python dictionary to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (str): The path to the output JSON file.

    Returns:
        bool: True if the save was successful, False otherwise.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)  # Use indent for better readability
        return True
    except Exception as e:
        print(f"Error saving dictionary to JSON: {e}")
        return False