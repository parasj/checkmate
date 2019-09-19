from collections import defaultdict
from functools import lru_cache
from typing import Tuple, Iterable, Dict, List, Set

import numpy as np
from graphviz import Digraph

Vertex = int
EdgeList = Iterable[Tuple[Vertex, Vertex]]
AdjList = Dict[Vertex, List[Vertex]]


class DFGraph:
    def __init__(self, args: AdjList, v: Iterable[Vertex], vfwd_map: Dict[Vertex, Vertex], vloss: Vertex,
                 cost_cpu: Dict[Vertex, int] = None, cost_ram: Dict[Vertex, int] = None,
                 node_names: Dict[Vertex, str] = None, cost_ram_parameters: int = 0):
        """
        Graph defines the forward and backward graph for a neural network
        :param deps (Dict[int, List[int]]): Dependency listing, where arguments ordered
        :param v: List of nodes in the graph
        :param vfwd_map: Mapping from forward graph to corresponding nodes in the reverse graph
        :param cost_cpu: Dictionary mapping nodes to respective integral runtime costs (for forward and backward operators)
        :param cost_ram: Dictionary mapping nodes to respective integral memory costs (for forward and backward operators)
        """
        self.args = defaultdict(list, args)
        self.v = list(sorted(v))
        self.size = len(self.v)
        self.edge_list = adj_to_edge_list(self.args, reverse_edge=True)
        self.vfwd_map = vfwd_map
        self.vfwd = list(sorted(vfwd_map.keys())) if vfwd_map else self.v
        self.vloss = vloss
        self.node_names = node_names if node_names is not None else {}
        if self.vloss not in self.node_names:
            self.node_names[self.vloss] = "Loss"
        self.size_fwd = len(self.vfwd)
        self.edge_list_fwd = self.induce_subgraph(self.vfwd)
        self.cost_cpu = cost_cpu if cost_cpu else {v: 1 for v in set(self.v + [self.vloss])}
        self.cost_ram = cost_ram if cost_ram else {v: 1 for v in set(self.v + [self.vloss])}
        self.cost_cpu_np = np.array([cost for _v, cost in sorted(self.cost_cpu.items())])
        self.cost_ram_np = np.array([ram for _v, ram in sorted(self.cost_ram.items())])
        self.cost_cpu_fwd = {v: cost_cpu[v] for v in self.vfwd} if cost_cpu else {v: 1 for v in self.vfwd}
        self.cost_ram_fwd = {v: cost_ram[v] for v in self.vfwd} if cost_ram else {v: 1 for v in self.vfwd}
        self.cost_ram_parameters = cost_ram_parameters

    @property
    def cost_ram_fixed(self):
        """Get fixed memory costs for the model (parameters and their gradients)"""
        return int(2 * self.cost_ram_parameters)

    def ram_gcd(self, *othervals):
        values = list(self.cost_ram.values())
        # vals.append(self.cost_ram_fixed)
        values.extend(othervals)
        values = np.array(values)
        intvalues = np.array(values, dtype=int)

        # GCD is 1 if values are not integral
        if not np.allclose(intvalues, values):
            return 1

        return np.gcd.reduce(intvalues)

    def cpu_gcd(self, *othervals):
        values = list(self.cost_cpu.values())
        values.extend(othervals)
        values = np.array(values)
        intvalues = np.array(values, dtype=int)

        # GCD is 1 if values are not integral
        if not np.allclose(intvalues, values):
            return 1

        return np.gcd.reduce(intvalues)

    def write_graphviz(self, directory, format='pdf', quiet=True, name=""):
        """
        Generate Graphviz-formatted edge list for visualization
        :param directory: str -- where to write source and rendered graph
        :param format: str -- file format for output
        :param quiet: bool -- whether or not to print debug information
        """
        dot = Digraph("!ExtractedGraph" + str(name))
        dot.attr('graph', rankdir='LR')
        for u in self.vfwd:
            with dot.subgraph() as s:
                s.attr(rank='same')
                node_name = self.node_names.get(u)
                node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
                s.node(str(u), node_name)

                v = self.forward_to_backward(u)
                node_name = "&nabla;{}".format(self.node_names.get(u, u))
                node_name = node_name if node_name is None else "{} ({})".format(node_name, str(v))
                s.node(str(v), node_name, style='filled')

        for u in self.v:
            if u not in self.vfwd_map.values() and u not in self.vfwd_map.keys():
                node_name = self.node_names.get(u)
                node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
                dot.node(str(u), node_name)

        for edge in self.edge_list:
            dep_order = str(self.args[edge[-1]].index(edge[0]))
            if edge not in self.edge_list_fwd and self.vloss not in edge:
                dot.edge(*map(str, edge), constraint='false', label=dep_order)
            else:
                dot.edge(*map(str, edge), label=dep_order)
        try:
            dot.render(directory=directory, format=format, quiet=quiet)
        except TypeError:
            dot.render(directory=directory, format=format)

    @property
    @lru_cache(maxsize=None)
    def max_degree(self):
        counts = defaultdict(int)
        for (i, j) in self.edge_list:
            counts[j] += 1
            counts[i] += 1
        return max(counts.values())

    @property
    @lru_cache(maxsize=None)
    def checkpoint_set(self) -> Set[Vertex]:
        """Determine checkpointable nodes in a forward graph (not backward)"""

        def connected_components(V: Iterable[int], E: EdgeList):
            def dfs(graph, node, visited):
                if node not in visited:
                    visited.append(node)
                    for n in graph[node]:
                        dfs(graph, n, visited)
                return visited

            unvisited = set(V)
            adj_list = edge_to_adj_list(E, convert_undirected=True)
            components = 0
            while len(unvisited) > 0:
                v = unvisited.pop()
                visited = dfs(adj_list, v, [])
                for v in visited:
                    unvisited.discard(v)
                components += 1
            return components

        E = list(self.edge_list_fwd)
        V = set([i for (i, j) in E] + [j for (i, j) in E]).union({-1, -2})  # directed to undirected graph
        E = [(i, j) for (i, j) in E if i in V and j in V] + [(-1, 0), (max(V), -2)]

        checkpoint_ok = set()
        for v in filter(lambda v: v >= 0, V):  # ignore placeholders for input and output
            # count connected components in induced subgraph F = G / v
            Vprime = {x for x in V if x != v}
            Eprime = {e for e in E if v not in e}
            n_components = connected_components(Vprime, Eprime)
            if n_components > 1:
                checkpoint_ok.add(v)
        return checkpoint_ok

    @property
    def checkpoint_set_all(self) -> set:
        E = list(self.edge_list_fwd)
        V = set([i for (i, j) in E] + [j for (i, j) in E])
        return V

    @property
    @lru_cache(maxsize=None)
    def topological_order_fwd(self):
        E = self.edge_list_fwd

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

    @property
    @lru_cache(maxsize=None)
    def predecessor_dict(self):
        preds = defaultdict(list)
        for eidx, (u, v) in enumerate(self.edge_list):
            preds[v].append((eidx, u))
        return preds

    @property
    @lru_cache(maxsize=None)
    def successor_dict(self):
        sucs = defaultdict(list)
        for eidx, (u, v) in enumerate(self.edge_list):
            sucs[u].append((eidx, v))
        return sucs

    def predecessors(self, node) -> Set[Vertex]:
        return {u for (_, u) in self.predecessor_dict[node]}

    def successors(self, node) -> Set[Vertex]:
        return {u for (_, u) in self.successor_dict[node]}

    def predecessors_indexed(self, node):
        return self.predecessor_dict[node]

    def successors_indexed(self, node):
        return self.successor_dict[node]

    def induce_subgraph(self, nodes: List[int]):
        return [e for e in self.edge_list if all(map(lambda x: x in nodes, e))]

    def is_forward_node(self, node: int):
        return not self.is_loss_node(node) and node in self.vfwd

    def is_backward_node(self, node: int):
        return not self.is_loss_node(node) and not self.is_forward_node(node)

    def is_loss_node(self, node: int):
        return node == self.vloss

    def forward_to_backward(self, node: int):
        if self.is_loss_node(node):
            return None
        assert self.is_forward_node(node)
        return self.vfwd_map[node]

    @lru_cache(maxsize=None)
    def backward_to_forward(self, node: int):
        assert self.is_backward_node(node)
        return next(key for key, value in self.vfwd_map.items() if value == node)

    def dependency_order(self, node: int):
        return self.args[node]

    def max_degree_ram(self):
        """compute minimum memory needed for any single node (ie inputs and outputs)"""
        return max([sum([self.cost_ram[u] for u in self.predecessors(v)]) + self.cost_ram[v] for v in self.vfwd])


def gen_linear_graph(forward_node_count, **kwargs):
    """
    gen_linear_graph will generate linear-style graphs like VGG and AlexNet.
    Method returns forward and backward graphs. Pass cost_ram and cost_cpu as kwargs.
    :param forward_node_count: number of forward (not backward nodes)
    :return: Graph object containing linear graph
    """
    args = defaultdict(list)
    vfwd_map = {}
    loss_node_idx = forward_node_count
    for i in range(forward_node_count * 2):
        args[i + 1].append(i)
        if i < forward_node_count:
            corresponding_bwd = (forward_node_count * 2) - i
            args[corresponding_bwd].append(i)
            vfwd_map[i] = corresponding_bwd
    v = list(vfwd_map.keys()) + list(vfwd_map.values()) + [loss_node_idx]
    return DFGraph(args=args, v=v, vfwd_map=vfwd_map, vloss=loss_node_idx, **kwargs)


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
