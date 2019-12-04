from checkmate.core.dfgraph import DFGraph
from checkmate.core.graph_builder import GraphBuilder

# operations to ignore due to non-determinism or impurity
from checkmate.core.utils.timer import Timer

TFOPS_IGNORE = ["Placeholder", "ReadVariableOp", "Const", "BroadcastGradientArgs", "Fill"]


def dfgraph_from_tf_function(fn) -> DFGraph:
    """
    Given a TensorFlow 2.0 ConcreteFunction, this function will profile and return a DFGraph representation of your
    problem that can be optimized.
    :param fn: ConcreteFunction to extract
    :return: DFGraph
    """
    # todo cost-model
    # todo type assertions for concrete function
    assert fn.__class__.__name__ == "ConcreteFunction", "Can only compile concrete functions"
    gb = GraphBuilder()
    ops = {op for op in fn.graph.get_operations() if op.type not in TFOPS_IGNORE}

    # initialize nodes
    for op in ops:
        gb.add_node(op.name, cpu_cost=1, ram_cost=1, backward="gradients" not in op.name)

    # build dependency graph
    for op in ops:
        for out in op.outputs:
            for cons in out.consumers():
                if cons in ops:
                    gb.add_deps(cons.name, op.name)
    return gb.make_graph()
