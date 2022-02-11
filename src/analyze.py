import copy
import os, inspect
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt


def get_hyperedge_pairs(hypergraph:dict):
    out = []
    keys = list(hypergraph.keys())
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            intersection = set(hypergraph[a]) & set(hypergraph[b])
            if len(intersection) == len(hypergraph[a]) or len(intersection) == len(hypergraph[b]):
                continue
            if len(intersection) > 0:
                out.append((a, b))
    return out


def intra_distance(hypergraph:dict, hyperedge_pairs:list, dists:np.ndarray) -> list:
    '''
    return list of intra_distance within hyperedges
    :param hyperedge_pair: list of hyperedge pairs
    :type hyperedge_pair:
    :param dists: distances between all pairs of nodes
    :type dists: numpy.ndarray
    :return:
    :rtype:
    '''
    out = []
    for pair in hyperedge_pairs:
        a, b = pair
        a_nodes = hypergraph[a]
        b_nodes = hypergraph[b]

        a_pairwise = combinations(a_nodes, 2)
        a_avg = np.mean([dists[pair[0]][pair[1]] for pair in a_pairwise])

        b_pairwise = combinations(b_nodes, 2)
        b_avg = np.mean([dists[pair[0]][pair[1]] for pair in b_pairwise])

        out.append(np.mean([a_avg, b_avg]))
    return out


def inter_distance(hypergraph:dict, hyperedge_pairs:list, dists:np.ndarray) -> list:
    '''
    return list of inter_distance within
    :param hyperedge_pairs:
    :type hyperedge_pairs:
    :param dists:
    :type dists:
    :return:
    :rtype:
    '''
    out = []
    for pair in hyperedge_pairs:
        a, b = pair

        a_nodes = set(hypergraph[a])
        b_nodes = set(hypergraph[b])

        intersect = a_nodes & b_nodes

        a = a_nodes - intersect
        b = b_nodes - intersect

        pairwise = combinations(a.union(b), 2)
        out.append(np.mean([dists[pair[0]][pair[1]] for pair in pairwise]))
    return out
