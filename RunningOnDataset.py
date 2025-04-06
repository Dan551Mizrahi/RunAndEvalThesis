import os
from tqdm import tqdm
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from PreprocessEnumRunEval import running_times_in_dict
from PyMELib.utils.readHypergraphFromFile import read_hypergraph, MAX_CHR
from memoryManagement import memory_limit_p
from RootMeasures import *
from WritingResultsToJSON import save_dict_to_json
import networkx as nx
import json
import sys
import argparse

sys.setrecursionlimit(1700)

def timeout_memoryout_recursion_handler(what_happened: str, file_path: str):
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

    hypergraph = read_hypergraph(file_path)

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

    existing_data.update({
        "Num of Vertices": num_vertices,
        "Num of Hyperedges": num_hyperedges,
        "n + m": num_vertices + num_hyperedges,
        "Max Hyperedge Size": max(hyperedges_size),
        "Min Hyperedge Size": min(hyperedges_size),
        "Avg Hyperedge Size": sum(hyperedges_size) / len(hyperedges_size),
        "Size of Hypergraph": num_vertices + sum(hyperedges_size),
        "Number of Connected Components": num_connected_components})
    if not what_happened != "Recursion":
        td = RootedDisjointBranchNiceTreeDecomposition(hypergraph)
        # Root measures
        dict_of_join_nodes = count_join_nodes(td)
        number_of_join_nodes = len(dict_of_join_nodes)
        size_of_join_nodes = sum(dict_of_join_nodes.values())
        special_join_measure = sum({5 ** l for l in dict_of_join_nodes.values()})
        real_effective_width = max(dict_of_join_nodes.values()) - 1
        dict_of_branches = count_branching(td)
        number_of_branching = sum(dict_of_branches.values())
        max_branching = max(dict_of_branches.values())
        existing_data.update({"Treewidth": td.width,
            "Number of Join Nodes": number_of_join_nodes,
            "Size of Join Nodes": size_of_join_nodes,
            "Special Join Measure": special_join_measure,
            "Number of Branching": number_of_branching,
            "Max Branching": max_branching,
            "Real Effective Width": real_effective_width})
    else:
        existing_data.update({"Treewidth": what_happened,
            "Number of Join Nodes": what_happened,
            "Size of Join Nodes": what_happened,
            "Special Join Measure": what_happened,
            "Number of Branching": what_happened,
            "Max Branching": what_happened,
            "Real Effective Width": what_happened})

    existing_data["Preprocessing Runtime"] = what_happened
    existing_data["Number of Minimal Hitting Sets"] = what_happened
    existing_data["Delays"] = what_happened
    existing_data["Average delay"] = what_happened

    save_dict_to_json(existing_data, data_file_path)


def help_pool_server(file_path: str, memory_limit = True):

    if memory_limit:
        memory_limit_p(0.175)
    try:
        parent_path = os.path.dirname(file_path)
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

        if "Preprocessing Runtime" not in existing_data.keys():
            existing_data.update(running_times_in_dict(file_path))

        # Write the updated dictionary to the file
        save_dict_to_json(existing_data, data_file_path)
    except TimeoutError:
        # Handle timeout
        timeout_memoryout_recursion_handler("Timeout", file_path)
        print(f"Process {file_path} timed out!")
    except MemoryError:
        # Handle memory error
        timeout_memoryout_recursion_handler("Memory", file_path)
        print(f"Process {file_path}_id.txt Memory!")
    except RecursionError:
        # Handle recursion error
        timeout_memoryout_recursion_handler("Recursion", file_path)
        print(f"Process {file_path} Recursion!")


def writing_of_an_entire_folder_server(folder_path: str, multiprocessing: bool = True):
    list1 = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if
             file.endswith(".dat") or file.endswith(".graph")]
    data_folder = os.path.join(folder_path, "data")
    os.makedirs(data_folder, exist_ok=True)
    if multiprocessing:
        with ProcessPool(5, max_tasks=2) as pool:
            # Create a data folder

            futures = []

            for i, file_path in enumerate(list1):  # Use enumerate to get indices
                future = pool.schedule(help_pool_server, args=[file_path], timeout=6000)
                futures.append((i, future))  # Store index with future

            for i, future in tqdm(futures):
                file_path = list1[i]
                try:
                    result = future.result()
                except TimeoutError:
                    # Handle timeout
                    timeout_memoryout_recursion_handler("Timeout", file_path)
                    print(f"Process {file_path} timed out!")
                except MemoryError:
                    # Handle memory error
                    timeout_memoryout_recursion_handler("Memory", file_path)
                    print(f"Process {file_path}_id.txt Memory!")
                except RecursionError:
                    # Handle recursion error
                    timeout_memoryout_recursion_handler("Recursion", file_path)
                    print(f"Process {file_path} Recursion!")
    else:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing the algorithms on a folder of files, extracting features and performance metrics.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the files.")
    parser.add_argument("-d", "--folder_of_datasets", action="store_true", help="")
    args = parser.parse_args()
    if args.folder_of_datasets:
        all_dirs = os.listdir(args.folder_path)
        for dir1 in all_dirs:
            folder_path = os.path.join(args.folder_path, dir1)
            if os.path.isdir(folder_path):
                writing_of_an_entire_folder_server(folder_path, multiprocessing=True)
    else:
        writing_of_an_entire_folder_server(args.folder_path, multiprocessing=True)