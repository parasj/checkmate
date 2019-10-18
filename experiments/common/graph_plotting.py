from typing import Optional

import numpy as np
from graphviz import Digraph

from remat.core.dfgraph import DFGraph
from remat.core.schedule import Schedule, OperatorEvaluation, ScheduledResult
from remat.core.utils.definitions import PathLike


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


def render_dfgraph(g: DFGraph, directory, format='pdf', quiet=True, name=""):
    """Generate Graphviz-formatted edge list for visualization, and write pdf"""
    dot = Digraph("render_dfgraph" + str(name))
    dot.attr('graph', ratio='compress', size="8.5")  # rankdir='LR',
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


def plot(sched_result: ScheduledResult, plot_mem_usage=False, save_file: Optional[PathLike] = None, show=False,
         plt=None):
    assert sched_result.feasible
    R = sched_result.schedule_aux_data.R
    S = sched_result.schedule_aux_data.S

    if plt is None:
        import matplotlib.pyplot as plt

    if plot_mem_usage:
        fig, axs = plt.subplots(1, 4)
        vmax = sched_result.schedule_aux_data.mem_grid
        if sched_result.ilp_aux_data is not None:
            U = sched_result.ilp_aux_data.U
            vmax = vmax if U is None else max(vmax, np.max(U))
        else:
            U = None

        # Plot slow verifier memory usage
        axs[2].invert_yaxis()
        axs[2].pcolormesh(sched_result.schedule_aux_data.mem_grid, cmap="Greys", vmin=0, vmax=vmax)
        axs[2].set_title("Memory usage (verifier)")

        # Plot solver memory usage variables
        axs[3].invert_yaxis()
        axs[3].set_title("Memory usage (solved)")
        if U is not None:
            axs[3].pcolormesh(U, cmap="Greys", vmin=0, vmax=vmax)

        fig.set_size_inches(28, 6)
    else:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 6)

    axs[0].invert_yaxis()
    axs[0].pcolormesh(R, cmap="Greys")
    axs[0].set_title("R")

    axs[1].invert_yaxis()
    axs[1].pcolormesh(S, cmap="Greys")
    axs[1].set_title("S")

    if show:
        plt.show()
    if save_file:
        fig.savefig(save_file)
        plt.close(fig)
