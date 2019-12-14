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
    
    #TODO: Make this a class that inherits
    op_dict = {}  #specific for tensorflow, make an dict from uuid to operation

    # initialize nodes
    i = 0
    for op in ops:
        gb.add_node(op.name, cpu_cost=1, ram_cost=1, backward=op.name.startswith("gradients/"))
        op_dict[i] = op
        i+= 1

    # build dependency graph
    for op in ops:
        for out in op.outputs:
            for cons in out.consumers():
                if cons in ops:
                    gb.add_deps(cons.name, op.name)
    g = gb.make_graph()
    g.op_dict = op_dict
    return g
