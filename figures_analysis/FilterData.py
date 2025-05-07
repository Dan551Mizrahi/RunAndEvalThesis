from typing import List, Dict, Any

def filter_data(data: List[Dict], filter1=None):
    """
    Filter the data based on the provided filter function.
    :param data: List of dictionaries containing the data.
    :param filter1: Function to filter the data.
    :return: Filtered data.
    """
    if filter1 is None:
        return data
    return [item for item in data if filter1(item)]

def filter_by_tw(data, tw):
    """
    Filter the data by a specific treewidth value.
    :param data: List of dictionaries containing the data.
    :param tw: The treewidth value to filter by.
    :return: Filtered data.
    """
    return filter_data(data, lambda x: x["Treewidth"] == tw)

def filter_by_max_join(data, max_join):
    """
    Filter the data by a maximum max join node size value.
    :param data: List of dictionaries containing the data.
    :param max_join: The value to filter by.
    :return: Filtered data.
    """
    return filter_data(data, lambda x: x["Max Join Node Size"] <= max_join)