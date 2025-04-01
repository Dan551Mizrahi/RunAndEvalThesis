import time
import os
from PyMELib.TreeDecompositions import RootedDisjointBranchNiceTreeDecomposition
from PyMELib.utils.readHypergraphFromFile import read_hypergraph
from PyMELib.PreprocessingAlgorithms import *
from PyMELib.EnumerationAlgorithms import *


def running_times(path: str, first_k = None, iterative = True):

    hypergraph = read_hypergraph(path)

    rooted_dntd = RootedDisjointBranchNiceTreeDecomposition(hypergraph)


    Y = []

    # preprocessing phase
    first_time = time.time()
    create_factors(rooted_dntd)
    if iterative:
        calculate_factors_for_mds_enum_iterative(rooted_dntd)
    else:
        calculate_factors_for_mds_enum(rooted_dntd, rooted_dntd.get_root())
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

    return preprocess_runtime, Y

def running_times_in_dict(path: str) -> dict:
    preprocess_runtime, Y = running_times(path)
    return_dict = dict()

    return_dict["Preprocess Runtime"] = preprocess_runtime
    return_dict["Number of Minimal Hitting Sets"] = len(Y)
    return_dict["Delays"] = Y
    return_dict["Average delay"] = sum(Y)/len(Y)
    return return_dict