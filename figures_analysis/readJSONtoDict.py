import json
import os
from typing import List, Dict

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
        # print(f"Error reading JSON file: {e}")
        # print(file_path)
        return {}


def read_whole_folders_data(folder_path: str) -> List[Dict]:
    """
    Reads all JSON files in a folder and converts them to a list of dictionaries.
    :param folder_path: The path to the folder containing JSON files.
    """
    data_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            data = read_json_to_dict(file_path)
            if len(data) > 0:
                try:
                    if type(data["Preprocess Runtime"]) == float:
                        data_list.append(data)
                except KeyError:
                    pass
    return data_list

def read_folder_of_folders_to_one_list(folder_path: str) -> List[Dict]:
    """
    Reads all JSON files in a folder and its data subfolders and converts them to a list of dictionaries.
    :param folder_path: The path to the folder containing the JSON files.
    """
    data_list = []
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(root, file_name)
                data = read_json_to_dict(file_path)
                if len(data) > 0:
                    try:
                        if type(data["Preprocessing Runtime"]) == float:
                            data_list.append(data)
                    except KeyError:
                        pass
    return data_list