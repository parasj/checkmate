from graphviz import Digraph

from remat.core.dfgraph import DFGraph
from remat.core.schedule import Schedule, OperatorEvaluation


# deprecated
def tensor_plot(g: DFGraph, sched: Schedule, directory, tag=None, format='pdf', quiet=True):
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


# deprecated
def write_graphviz(g: DFGraph, directory, format='pdf', quiet=True, name=""):
    """
    Generate Graphviz-formatted edge list for visualization
    """
    dot = Digraph("!ExtractedGraph" + str(name))
    dot.attr('graph', rankdir='LR')
    for u in g.vfwd:
        with dot.subgraph() as s:
            s.attr(rank='same')
            node_name = g.node_names.get(u)
            node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
            s.node(str(u), node_name)

            v = g.forward_to_backward(u)
            node_name = "&nabla;{}".format(g.node_names.get(u, u))
            node_name = node_name if node_name is None else "{} ({})".format(node_name, str(v))
            s.node(str(v), node_name, style='filled')

    for u in g.v:
        if u not in g.vfwd_map.values() and u not in g.vfwd_map.keys():
            node_name = g.node_names.get(u)
            node_name = node_name if node_name is None else "{} ({})".format(node_name, str(u))
            dot.node(str(u), node_name)

    for edge in g.edge_list:
        dep_order = str(g.args[edge[-1]].index(edge[0]))
        if edge not in g.edge_list_fwd and g.vloss not in edge:
            dot.edge(*map(str, edge), constraint='false', label=dep_order)
        else:
            dot.edge(*map(str, edge), label=dep_order)
    try:
        dot.render(directory=directory, format=format, quiet=quiet)
    except TypeError:
        dot.render(directory=directory, format=format)
