from typing import List, Dict, Any
import os
from readJSONtoDict import read_whole_folders_data, read_json_to_dict

def find_agg_function_on_data(data: List[Dict[str, Any]], key: str, agg_function: Any) -> Any:
    """
    Find the aggregation function on the data (in the format collected in RunningOnData.py).

    :param data: List of dictionaries containing the data.
    :param key: The key in the dictionary to apply the aggregation function to its value.
    :param agg_function: The aggregation function to apply.
    :return: The aggregation result.
    """
    if not data:
        return -1

    data_for_aggregation = []

    for item in data:
        data_for_aggregation.append(item[key])

    return agg_function(data_for_aggregation)

def find_agg_function_on_data_from_path_to_data_folder(folder_path: str, key: str, agg_function: Any) -> Any:
    """
    Find the aggregation function on the data (in the format collected in RunningOnData.py) from a folder path.

    :param folder_path: Path to the folder containing the data files.
    :param key: The key in the dictionary to apply the aggregation function to its value.
    :param agg_function: The aggregation function to apply.
    :return: The aggregation result.
    """
    data_folder = os.path.join(folder_path, "data")
    all_data = read_whole_folders_data(data_folder)
    return find_agg_function_on_data(all_data, key, agg_function)

if __name__ == "__main__":
    big_folder_path = "/Users/dan/Desktop/FinalGeneratedData"

    for folder in sorted(os.listdir(big_folder_path)):
        if folder.startswith("."):
            continue
        folder_path = os.path.join(big_folder_path, folder)
        data_path = os.path.join(folder_path, "data")
        key = "Treewidth"
        if os.path.isdir(data_path):
            print(f"Folder: {folder}")
            print(f"Length of data: {len(read_whole_folders_data(data_path))}")
            print("Mean:", find_agg_function_on_data_from_path_to_data_folder(folder_path, key, lambda x: sum(x) / len(x)))
            print("Median:", find_agg_function_on_data_from_path_to_data_folder(folder_path, key, lambda x: sorted(x)[len(x) // 2]))
            print("Max:", find_agg_function_on_data_from_path_to_data_folder(folder_path, key, max))
            print("Min:", find_agg_function_on_data_from_path_to_data_folder(folder_path, key, min))
            print()