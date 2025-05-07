from PyMELib.TreeDecompositions import RootedDisjointBranchNiceTreeDecomposition, NodeType


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

def width(tree_decomposition: RootedDisjointBranchNiceTreeDecomposition) -> int:
    """Returns the width of a rooted (disjoint) nice tree decomposition.
    :param (Disjoint Nice) tree_decomposition: RootedDisjointBranchNiceTreeDecomposition
    :return: int
    """
    max_width = 0
    for node in tree_decomposition.nodes:
        max_width = max(max_width, len(tree_decomposition.nodes[node]['bag']))
    return max_width - 1