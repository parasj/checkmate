import tensorflow as tf
import tensorflow.keras as keras

from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.definitions import checkmate_data_dir
from experiments.common.graph_plotting import plot_dfgraph
from experiments.common.load_keras_model import get_keras_model


def test_linear16_execution():
    input_shape = (224, 224, 3)
    inputs = keras.layers.Input(shape=(224, 224, 3))
    vgg16 = get_keras_model("linear16", input_shape=input_shape)
    fn = tf.function(vgg16).get_concrete_function(inputs=inputs)

    g = dfgraph_from_tf_function(fn)
    plot_dfgraph(g, checkmate_data_dir() / "test_exec" / "vgg16" / "base")
    sqrtn_result = solve_chen_sqrtn(g, False)
    assert sqrtn_result.feasible

    sqrtn_fn = edit_graph(fn, g.op_dict, sqrtn_result.schedule)
    g_new = dfgraph_from_tf_function(sqrtn_fn)
    plot_dfgraph(g_new, checkmate_data_dir() / "test_exec" / "vgg16" / "sqrtn")


if __name__ == "__main__":
    test_linear16_execution()
