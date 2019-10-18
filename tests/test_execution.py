import logging

from experiments.common.execution_utils import random_batch
from experiments.common.graph_plotting import render_dfgraph
from experiments.common.load_keras_model import get_keras_model
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.tensorflow2.execution import tfgraph_from_schedule
from remat.tensorflow2.extraction import dfgraph_from_keras
from remat.tensorflow2.tf_losses import categorical_cross_entropy

import tensorflow as tf


def test_exec_vgg16_checkpointall():
    try:
        import gurobipy as _
    except ImportError as e:
        logging.exception(e)
        logging.warning("Continuing with tests, gurobi not installed")
        return
    model = get_keras_model("VGG16")
    g = dfgraph_from_keras(mod=model)
    schedule = solve_checkpoint_all(g)
    assert schedule.feasible
    assert schedule.schedule_aux_data.cpu <= sum(g.cost_cpu.values())
    assert schedule.schedule_aux_data.activation_ram <= sum(g.cost_cpu.values())

    loss = categorical_cross_entropy
    checkmate_graph = tfgraph_from_schedule(model, g, schedule, loss=loss, debug=True)

    test_batch, test_labels = random_batch(1)
    with tf.GradientTape(persistent=True) as tape:
        tf_pred = model(test_batch)
        tf_loss = loss(tf_pred, test_labels)
    tf_grads = tape.gradient(tf_loss, model.trainable_variables)

    our_loss, our_grads = checkmate_graph(test_batch, test_labels)

    logging.info(f"TF baseline loss = {tf_loss}")
    logging.info(f"Checkmate loss = {our_loss}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_exec_vgg16_checkpointall()
