import tensorflow as tf
import logging

from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.core.utils.timer import Timer
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.load_keras_model import get_keras_model

BS = 128

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.info("building graph")
    with Timer("build_graph", print_results=True):
        model = get_keras_model("ResNet50")

        def grads(images, labels):
            with tf.GradientTape() as tape:
                pred = model(images)
                loss = tf.reduce_mean(pred - labels)
            gradient = tape.gradient(loss, model.trainable_variables)
            return loss, gradient

        grad_fn = tf.function(grads).get_concrete_function(
            tf.TensorSpec(shape=(BS, 224, 224, 3)), tf.TensorSpec(shape=(BS, 1000))
        )
    logging.info("tracing graph")
    with Timer("trace_graph", print_results=True):
        g = dfgraph_from_tf_function(grad_fn)
    # sched_result = solve_ilp_gurobi(g, budget=platform_memory("p2xlarge"), approx=False, eps_noise=0.0)
    # sched_result = solve_approx_lp_deterministic_05_threshold(g, budget=platform_memory("p2xlarge"))
    logging.info("solving graph")
    with Timer("sched_graph", print_results=True):
        sched_result = solve_chen_sqrtn(g, True)
    # logging.info("rebuilding graph")
    # new_graph = edit_graph(grad_fn, g.op_dict, sched_result.schedule)

    # plot_path = checkmate_data_dir() / "exec"
    # plot_path.mkdir(parents=True, exist_ok=True)
    # plot_schedule(sched_result, save_file=plot_path / "optimal_vgg16.png")
