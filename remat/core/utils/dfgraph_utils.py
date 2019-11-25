from collections import defaultdict

from remat.core.utils.definitions import EdgeList, AdjList

import numpy as np


def toposort(edge_list: EdgeList):
    """
    Compute topological ordering of this graph given graph structure
    :param edge_list: A mapping of (source, dest) pairs
    :return: list of nodes in topological order
    """
    def _toposort_helper(adj_list_, v, visited_, stack_):
        visited_.add(v)
        for i in adj_list_[v]:
            if i not in visited_:
                _toposort_helper(adj_list_, i, visited_, stack_)
        stack_.insert(0, v)

    adj_list = edge_to_adj_list(edge_list, convert_undirected=True)
    visited = set()
    stack = []
    for i in adj_list.keys():
        if i not in visited:
            _toposort_helper(adj_list, i, visited, stack)
    return stack


def edge_to_adj_list(E: EdgeList, convert_undirected=False):
    """Returns an (undirected / bidirectional) adjacency list"""
    adj_list = defaultdict(set)
    for (i, j) in E:
        adj_list[i].add(j)
        if convert_undirected:
            adj_list[j].add(i)
    return dict(adj_list)


def adj_to_edge_list(E: AdjList, convert_undirected=False, reverse_edge=False):
    """Returns an edge list
    :param E: AdjList -- input graph
    :param convert_undirected: bool -- if true, add u -> v and v -> u to output graph
    :param reverse_edge: bool -- if true, reverse edge direction prior to conversion
    :return:
    """
    edge_list = []
    for u, deps in E.items():
        for v in deps:
            edge = (u, v) if not reverse_edge else (v, u)
            edge_list.append(edge)
            if convert_undirected:
                edge_list.append(tuple(reversed(edge)))
    return edge_list


def gcd(*args):
    values = np.array(args)
    intvalues = np.array(values, dtype=int)
    if not np.allclose(intvalues, values):  # GCD is 1 if values are not integral
        return 1
    return np.gcd.reduce(intvalues)
