import pytest


def get_testnet_graph():
    import tensorflow as tf
    from checkmate.tf2.extraction import dfgraph_from_tf_function
    from checkmate.tf2.util.load_keras_model import get_keras_model
    model = get_keras_model("test")

    @tf.function
    def get_grads(inputs):
        y = model(inputs)
        y = tf.reduce_mean(y)
        return tf.gradients(y, model.trainable_variables)

    x = tf.ones(shape=(1, 224, 224, 3), name="input")
    grad_conc = get_grads.get_concrete_function(x)
    return dfgraph_from_tf_function(grad_conc)


@pytest.mark.parametrize("budget_threshold", [(1.2,), (1.0,), (0.9,), (0.5,), (0.1,), (0.05,)])
def test_cvxpy(budget_threshold):
    from checkmate.core.solvers.cvxpy_solver import solve_checkmate_cvxpy
    g = get_testnet_graph()
    budget = sum(g.cost_ram.values()) * budget_threshold
    solve_checkmate_cvxpy(g, budget, solver_override="CBC")
