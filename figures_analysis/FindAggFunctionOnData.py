from typing import List, Dict, Any
import os
from readJSONtoDict import read_whole_folders_data, read_json_to_dict
import math

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



def calculate_correlation_covariance(X, Y):
    """
    Computes the covariance and Pearson correlation coefficient between two lists of numbers.

    Args:
        X (list): The first list of numbers.
        Y (list): The second list of numbers.

    Returns:
        tuple: A tuple containing (covariance, correlation).
               Returns (None, None) if lists are of different lengths
               or if standard deviation is zero for correlation calculation.
    """
    if len(X) != len(Y):
        print("Error: Lists must have the same length.")
        return None, None

    n = len(X)
    if n == 0:
        print("Error: Lists cannot be empty.")
        return None, None

    # Calculate means
    mean_X = sum(X) / n
    mean_Y = sum(Y) / n

    # Calculate covariance
    covariance = sum((X[i] - mean_X) * (Y[i] - mean_Y) for i in range(n)) / n

    # Calculate standard deviations for correlation
    std_dev_X = math.sqrt(sum((x - mean_X) ** 2 for x in X) / n)
    std_dev_Y = math.sqrt(sum((y - mean_Y) ** 2 for y in Y) / n)

    # Calculate correlation
    correlation = None
    if std_dev_X != 0 and std_dev_Y != 0:
        correlation = covariance / (std_dev_X * std_dev_Y)
    elif covariance == 0:  # If covariance is 0 and std dev is 0, correlation is undefined but often treated as 0
        correlation = 0
    else:
        print("Warning: Cannot compute correlation due to zero standard deviation in one or both lists.")

    return covariance, correlation

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