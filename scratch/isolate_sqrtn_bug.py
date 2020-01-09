import tensorflow as tf

from checkmate.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_05_threshold
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.definitions import checkmate_data_dir
from experiments.common.graph_plotting import plot_schedule
from experiments.common.load_keras_model import get_keras_model
from experiments.common.profile.platforms import platform_memory

BS = 128

if __name__ == "__main__":
    model = get_keras_model("VGG16")

    def grads(images, labels):
        with tf.GradientTape() as tape:
            pred = model(images)
            loss = tf.reduce_mean(pred - labels)
        gradient = tape.gradient(loss, model.trainable_variables)
        return loss, gradient

    grad_fn = tf.function(grads).get_concrete_function(tf.TensorSpec(shape=(BS, 224, 224, 3)), tf.TensorSpec(shape=(BS, 1000)))
    g = dfgraph_from_tf_function(grad_fn)
    # sched_result = solve_ilp_gurobi(g, budget=platform_memory("p2xlarge"), approx=False, eps_noise=0.0)
    # sched_result = solve_approx_lp_deterministic_05_threshold(g, budget=platform_memory("p2xlarge"))
    sched_result = solve_chen_sqrtn(g, True)

    plot_path = checkmate_data_dir() / "exec"
    plot_path.mkdir(parents=True, exist_ok=True)
    plot_schedule(sched_result, save_file=plot_path / "optimal_vgg16.png")
