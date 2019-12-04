import tensorflow as tf

from checkmate.core.dfgraph import DFGraph
from checkmate.core.graph_builder import GraphBuilder
from experiments.common.graph_plotting import plot_dfgraph


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
    return gb.make_graph()


if __name__ == "__main__":
    @tf.function
    def fn(x):
        return 2 * x


    x = tf.constant([[2.0, 3.0]])
    conc_fn = fn.get_concrete_function(x)

    g = dfgraph_from_tf_function(conc_fn)
    plot_dfgraph(g, directory="/tmp/test_remat/tf")
