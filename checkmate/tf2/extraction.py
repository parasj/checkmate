from checkmate.core.dfgraph import DFGraph
from checkmate.core.graph_builder import GraphBuilder

# operations to ignore due to non-determinism or impurity
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
    ops = {op for op in fn.graph.get_operations()}  # if op.type not in TFOPS_IGNORE
    op_dict = {op.name: op for op in ops}  # used to build topoid -> op dictionary

    # propogate gradient nodes down, this will eliminate the "Identity" nodes at the end
    # run over and over until no more gradient nodes are added, thereby fully propogated
    grad_nodes = set()
    last_grad_node_count = -1
    while len(grad_nodes) > last_grad_node_count:
        last_grad_node_count = len(grad_nodes)
        for op in ops:
            if op.name.startswith("gradients/") or op.name in grad_nodes:
                grad_nodes.add(op.name)
                for out in op.outputs:
                    for cons in out.consumers():
                        grad_nodes.add(cons.name)

    # add nodes to dependency graph
    for op in ops:
        gb.add_node(op.name, cpu_cost=1, ram_cost=1, backward=op.name in grad_nodes)

    # build dependency graph
    for op in ops:
        for out in op.outputs:
            for cons in out.consumers():
                if cons in ops:
                    gb.add_deps(cons.name, op.name)
    g = gb.make_graph()
    name_to_topoid = {v: k for k, v in g.node_names.items()}
    id_dict = {name_to_topoid[name]: op for name, op in op_dict.items()}
    g.op_dict = id_dict
    return g
