import time
from PyMELib.utils.readHypergraphFromFile import read_hypergraph
from PyMELib.PreprocessingAlgorithms import *
from PyMELib.EnumerationAlgorithms import *
from RootMeasures import *
import networkx as nx

def Hypergraph_features(file_path: str, td: RootedDisjointBranchNiceTreeDecomposition):

    hypergraph = read_hypergraph(file_path)

    no_reduction_graph = hypergraph.copy()
    no_reduction_graph.remove_node(MAX_CHR)

    # Number of lines in the file is the number of the hyperedges
    with open(file_path, 'r') as f:
        hyperedges_size = []
        num_hyperedges = 0
        for line in f:
            if line.strip():
                if file_path.split('.')[-1] == 'dat':
                    hyperedges_size.append(len(line.split()))
                elif file_path.split('.')[-1] == 'graph':
                    hyperedges_size.append(len(line.split(',')[1:]))
                else:
                    print('Unsupported file format')
                    return
                # Increase the number of hyperedges
                num_hyperedges += 1

    num_vertices = len(no_reduction_graph.nodes) - num_hyperedges

    # Get the number of connected components
    num_connected_components = nx.number_connected_components(no_reduction_graph)

    # Root measures
    dict_of_join_nodes = count_join_nodes(td)
    number_of_join_nodes = len(dict_of_join_nodes)
    size_of_join_nodes = sum(dict_of_join_nodes.values())
    special_join_measure = sum({5 ** l for l in dict_of_join_nodes.values()})
    if len(dict_of_join_nodes) > 0:
        max_join_node_size = max(dict_of_join_nodes.values())
        min_join_node_size = min(dict_of_join_nodes.values())
    else:
        max_join_node_size = 0
        min_join_node_size = 0
    real_effective_width = width(td)
    dict_of_branches = count_branching(td)
    number_of_branching = sum(dict_of_branches.values())
    max_branching = max(dict_of_branches.values())

    return {
        "Num of Vertices": num_vertices,
        "Num of Hyperedges": num_hyperedges,
        "n + m": num_vertices + num_hyperedges,
        "Max Hyperedge Size": max(hyperedges_size),
        "Min Hyperedge Size": min(hyperedges_size),
        "Avg Hyperedge Size": round(sum(hyperedges_size) / len(hyperedges_size), 2),
        "Size of Hypergraph": num_vertices + sum(hyperedges_size),
        "Number of Connected Components": num_connected_components,
        'Diameter': nx.diameter(hypergraph),
        'Sparsity': round(nx.density(no_reduction_graph), 4),
        "Treewidth": td.width,
        "(m + n) * tw": (num_vertices + num_hyperedges) * td.width,
        "Number of Join Nodes": number_of_join_nodes,
        "Size of Join Nodes": size_of_join_nodes,
        "Special Join Measure": special_join_measure,
        "Max Join Node Size": max_join_node_size,
        "Min Join Node Size": min_join_node_size,
        "Number of Branching": number_of_branching,
        "Max Branching": max_branching,
        "Real Effective Width": real_effective_width
    }


def running_times_plus_features(path: str, first_k = None, iterative = True):

    hypergraph = read_hypergraph(path)

    rooted_dntd = RootedDisjointBranchNiceTreeDecomposition(hypergraph)

    # Create features dict
    features_dict = Hypergraph_features(path, rooted_dntd)

    Y = []

    # preprocessing phase
    first_time = time.time()
    create_factors(rooted_dntd)
    if iterative:
        calculate_factors_for_mds_enum_iterative(rooted_dntd, options_for_labels=True)
    else:
        calculate_factors_for_mds_enum(rooted_dntd, rooted_dntd.get_root(), options_for_labels=True)
    second_time = time.time()
    preprocess_runtime = second_time - first_time

    # enumeration phase
    i = 0
    first_time = time.time()
    for mhs in EnumMHS_iterative(rooted_dntd):
        if first_k is not None and i >= first_k:
            break
        next_time = time.time()
        Y.append(next_time - first_time)
        i +=1

    return preprocess_runtime, Y, features_dict

def running_times_in_dict(path: str, **kwargs) -> dict:
    preprocess_runtime, Y, features = running_times_plus_features(path, **kwargs)
    return_dict = features

    return_dict["Preprocess Runtime"] = preprocess_runtime
    return_dict["Number of Minimal Hitting Sets"] = len(Y)
    return_dict["Delays"] = Y
    if len(Y) > 0:
        return_dict["Average delay"] = round(Y[-1]/len(Y), 8)
    else:
        return_dict["Average delay"] = 0
    return return_dict

if __name__ == '__main__':

    pass
