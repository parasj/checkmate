from collections import defaultdict

from remat.core.utils.definitions import EdgeList, AdjList


def toposort(edge_list: EdgeList):
    E = edge_list

    def helper(adj_list_, v, visited_, stack_):
        visited_[v] = True
        for i in adj_list_[v]:
            if not visited_[i]:
                helper(adj_list_, i, visited_, stack_)
        stack_.insert(0, v)

    adj_list = edge_to_adj_list(E, convert_undirected=True)
    num_nodes = len(adj_list.keys())

    visited = [False] * num_nodes
    stack = []
    for i in range(num_nodes):
        if not visited[i]:
            helper(adj_list, i, visited, stack)
    return stack


def edge_to_adj_list(E: EdgeList, convert_undirected=False):
    """Returns an (undirected / bidirectional) adjacency list"""
    adj_list = defaultdict(set)
    for (i, j) in list(E):
        adj_list[i].add(j)
        if convert_undirected:
            adj_list[j].add(i)
    return adj_list


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