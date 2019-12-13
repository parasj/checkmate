import tensorflow as tf

from checkmate.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_05_threshold
from checkmate.core.utils.timer import Timer
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.load_keras_model import get_keras_model
from experiments.common.profile.platforms import platform_memory


def test_mlpblock_extract():
    vgg = get_keras_model("ResNet50")
    with Timer("Load model", print_results=True):
        @tf.function
        def get_grads(inputs):
            y = vgg(inputs)
            y = tf.reduce_mean(y)
            return tf.gradients(y, vgg.trainable_variables)

        x = tf.ones(shape=(1, 224, 224, 3), name="input")
        grad_conc = get_grads.get_concrete_function(x)

    with Timer("Extract DFGraph", print_results=True):
        g = dfgraph_from_tf_function(grad_conc)

    with Timer("Schedule", print_results=True):
        sched = solve_approx_lp_deterministic_05_threshold(g, platform_memory("p32xlarge") - g.cost_ram_fixed)
    print(sched.feasible)
    # todo actually run some kind of tests here...


if __name__ == "__main__":
    test_mlpblock_extract()
