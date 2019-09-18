from collections import defaultdict

from graphviz import Digraph

from remat.core.graph import Graph, EdgeList, AdjList
from remat.core.schedule import Schedule, OperatorEvaluation


def tensor_plot(g: Graph, sched: Schedule, directory, tag=None, format='pdf', quiet=True):
    dot = Digraph(f"!TensorPlot_{tag}", engine="dot")
    if sched is None:
        return
    for op in sched:
        if isinstance(op, OperatorEvaluation):
            if g.is_loss_node(op.id):
                node_name = "Loss"
            elif g.is_forward_node(op.id):
                node_name = g.node_names.get(op.id)
                node_name = node_name if node_name is None else f"{node_name} ({str(op.id)})"
            elif g.is_backward_node(op.id):
                fwd_node = g.backward_to_forward(op.id)
                node_name = "Grad<{}> {} {}".format(g.node_names.get(fwd_node), fwd_node, op.id)
            else:
                raise ValueError("Unknown operation")
            # dot.node("op{}".format(op.id), node_name, shape="diamond")
            # dot.edge("op{}".format(op.id), "reg{}".format(op.out_register))
            dot.node(f"reg{op.out_register}", f"Register {op.out_register} for {node_name}", shape="box")
            for dep_op, dep_reg in op.arg_regs.items():
                dot.edge("reg{}".format(dep_reg), "reg{}".format(op.out_register),
                         style="dashed", label=str(g.args[op.id].index(dep_op)))

    try:
        dot.render(directory=directory, format=format, quiet=quiet)
    except TypeError:
        dot.render(directory=directory, format=format)


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
    return Graph(args=args, v=v, vfwd_map=vfwd_map, vloss=loss_node_idx, **kwargs)


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
