from graphviz import Digraph

from remat.core.dfgraph import DFGraph
from remat.core.schedule import Schedule, OperatorEvaluation


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
