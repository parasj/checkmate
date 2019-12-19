import tensorflow as tf
import tensorflow.keras as keras

from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.definitions import checkmate_data_dir
from experiments.common.graph_plotting import plot_dfgraph, plot_schedule
from experiments.common.load_keras_model import get_keras_model


def test_vgg16_execution():
    vgg = get_keras_model("VGG16")

    @tf.function
    def get_grads(inputs):
        y = vgg(inputs)
        y = tf.reduce_mean(y)
        return tf.gradients(y, vgg.trainable_variables)

    x = tf.ones(shape=(1, 224, 224, 3), name="input")
    fn = get_grads.get_concrete_function(x)
    g = dfgraph_from_tf_function(fn)

    plot_dfgraph(g, checkmate_data_dir() / "test_exec" / "vgg16" / "base")
    # sqrtn_result = solve_chen_sqrtn(g, False)
    sqrtn_result = solve_checkpoint_all(g)
    assert sqrtn_result.feasible
    plot_schedule(sqrtn_result, save_file=checkmate_data_dir() / "test_exec" / "vgg16" / "checkpoint_last.png")

    sqrtn_fn = edit_graph(fn, g.op_dict, sqrtn_result.schedule)
    g_new = dfgraph_from_tf_function(sqrtn_fn)
    plot_dfgraph(g_new, checkmate_data_dir() / "test_exec" / "vgg16" / "checkpoint_last")


if __name__ == "__main__":
    test_vgg16_execution()
