import os
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from PreprocessEnumRunEval import running_times
from PyMELib.TreeDecompositions import RootedDisjointBranchNiceTreeDecomposition
from PyMELib.utils.readHypergraphFromFile import read_hypergraph, MAX_CHR
from PyMELib.TreeDecompositions import NodeType
from memoryManagement import memory_limit_p
from WritingResultsToJSON import save_dict_to_json
import networkx as nx
import json
import sys

sys.setrecursionlimit(1500)

def Hypergraph_features(file_path: str):

    def count_join_nodes(tree_decomposition: RootedDisjointBranchNiceTreeDecomposition) -> dict:
        """Counts the number of vertices in the bag of each join node in a rooting of a nice tree decomposition.
        :param (Disjoint Nice) tree_decomposition: RootedDisjointBranchNiceTreeDecomposition
        :return: dict
        """
        return_dict = dict()
        for node in tree_decomposition.nodes:
            if tree_decomposition.nodes[node]['type'] == NodeType.JOIN:
                return_dict[node] = len(tree_decomposition.nodes[node]['bag'])
        return return_dict

    def count_branching(tree_decomposition: RootedDisjointBranchNiceTreeDecomposition) -> dict:
        """Counts the number of branches for each vertex in a rooting of a nice tree decomposition.
        :param (Disjoint Nice) tree_decomposition: RootedDisjointBranchNiceTreeDecomposition
        :return: dict
        """
        help_dict = {chr(vertex): set() for vertex in tree_decomposition.original_graph.nodes}
        for node in tree_decomposition.nodes:
            for vertex in tree_decomposition.nodes[node]['bag']:
                help_dict[vertex[0]].add(tree_decomposition.nodes[node]['br'])

        return {vertex: len(help_dict[vertex]) for vertex in help_dict}

    hypergraph = read_hypergraph(file_path)
    try:
        rooted_dntd = RootedDisjointBranchNiceTreeDecomposition(hypergraph)
    except RecursionError:
        return {
            "Num of Vertices": "R",
            "Num of Hyperedges": "R",
            "n + m": "R",
            "Max Hyperedge Size": "R",
            "Min Hyperedge Size": "R",
            "Avg Hyperedge Size": "R",
            "Size of Hypergraph": "R",
            "Number of Connected Components": "R",
            "Treewidth": "R",
            "Number of Join Nodes": "R",
            "Size of Join Nodes": "R",
            "Special Join Measure": "R",
            "Number of Branching": "R",
            "Max Branching": "R",
        }

    no_reduction_graph = hypergraph.copy()
    no_reduction_graph.remove_node(MAX_CHR)

    # Number of lines in the file is the number of the hyperedges
    with open(file_path, 'r') as f:
        hyperedges_size = []
        num_hyperedges = 0
        for line in f:
            if line.strip():
                hyperedges_size.append(len(line.split()))
                # Increase the number of hyperedges
                num_hyperedges += 1

    num_vertices = len(no_reduction_graph.nodes) - num_hyperedges

    # Get the number of connected components
    num_connected_components = nx.number_connected_components(no_reduction_graph)

    # Root measures
    dict_of_join_nodes = count_join_nodes(rooted_dntd)
    number_of_join_nodes = len(dict_of_join_nodes)
    size_of_join_nodes = sum(dict_of_join_nodes.values())
    special_join_measure = sum({5 ** l for l in dict_of_join_nodes.values()})
    dict_of_branches = count_branching(rooted_dntd)
    number_of_branching = sum(dict_of_branches.values())
    max_branching = max(dict_of_branches.values())

    return {
        "Num of Vertices": num_vertices,
        "Num of Hyperedges": num_hyperedges,
        "n + m": num_vertices + num_hyperedges,
        "Max Hyperedge Size": max(hyperedges_size),
        "Min Hyperedge Size": min(hyperedges_size),
        "Avg Hyperedge Size": sum(hyperedges_size) / len(hyperedges_size),
        "Size of Hypergraph": num_vertices + sum(hyperedges_size),
        "Number of Connected Components": num_connected_components,
        "Treewidth": rooted_dntd.width,
        "Number of Join Nodes": number_of_join_nodes,
        "Size of Join Nodes": size_of_join_nodes,
        "Special Join Measure": special_join_measure,
        "Number of Branching": number_of_branching,
        "Max Branching": max_branching,
    }

def timeout_memoryout_handler(what_happened: str, file_path: str):
    parent_path = os.path.dirname(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    data_folder = os.path.join(parent_path, "data")
    data_file_path = os.path.join(data_folder, file_name.split('.')[0] + "_data.json")

    try:
        with open(data_file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}
    except json.JSONDecodeError:
        existing_data = {}


def help_pool_server(file_path: str):
    memory_limit_p(0.1)

    parent_path = os.path.dirname(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    data_folder = os.path.join(parent_path, "data")
    data_file_path = os.path.join(data_folder, file_name.split('.')[0] + "_data.json")

    try:
        with open(data_file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}
    except json.JSONDecodeError:
        existing_data = {}

    if "Treewidth" not in existing_data.keys():
        existing_data.update(Hypergraph_features(file_path))

    if "Preprocessing Runtime" not in existing_data.keys():
        existing_data.update(running_times(file_path))

    # Write the updated dictionary to the file
    save_dict_to_json(existing_data, data_file_path)


def writing_of_an_entire_folder_server(folder_path: str, max_for_one = 5):


    with ProcessPool(9, max_tasks=2) as pool:
        # Create a data folder
        data_folder = os.path.join(folder_path, "data")
        os.makedirs(data_folder, exist_ok=True)

        list1 = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".dat") or file.endswith(".graph")][:max_for_one]
        futures = []
        for i, file_path in enumerate(list1):  # Use enumerate to get indices
            future = pool.schedule(help_pool_server, args=[file_path], timeout=6000)
            futures.append((i, future))  # Store index with future

        for i, future in tqdm(futures):
            file_path = list1[i]
            file_name = os.path.basename(file_path)
            data_file_path = os.path.join(data_folder, file_name.split('.')[0] + "_data.json")
            try:
                print(future.result())
            except TimeoutError:
                # Handle timeout
                print(f"Process {file_path} timed out!")
            except MemoryError:
                # Handle memory error
                print(f"Process {file_path}_id.txt Memory!")

if __name__ == "__main__":
    print("ok")