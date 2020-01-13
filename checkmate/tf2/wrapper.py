import logging
import subprocess
from typing import Iterable

import psutil
import tensorflow as tf
from tensorflow.python.client import device_lib

from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.extraction import dfgraph_from_tf_function


def _using_gpu_check():
    return tf.test.is_gpu_available() and tf.test.is_built_with_cuda()


def _get_gpu_memory_map_mb():
    # from https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    mem = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"], encoding="utf-8")
    gpu_memory = [int(x) for x in mem.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def compile_tf2(
    model: tf.keras.Model,
    loss: tf.losses.Loss,
    optimizer: tf.optimizers.Optimizer,
    input_spec=None,
    label_spec=None,
    budget="auto",
):
    """
    Checkmate optimizes your DNN graphs to consume less GPU memory. Call this function using a tf.function
    :param model: a keras Model to optimize
    :param loss: loss function to use when training
    :param input_spec: tf.TensorSpec list that corresponds to model inputs
    :param budget:
    """
    # set input, output shapes
    if model.input_spec is None and input_spec is None:
        raise ValueError(
            "Keras model has not been compiled yet! If model.input_spec is not defined, then input_spec "
            "parameter must be set in the call to checkmate.tensorflow2.compile."
        )
    if label_spec is None:
        raise ValueError(
            "Checkmate needs the shape of the label in order to calculate the size of all operations. Pass in"
            "an example input or tf.TensorSpec object representing the shape of the label."
        )
    input_spec = model.input_spec if input_spec is None else input_spec

    # query budget if not specified
    if budget == "auto":
        if _using_gpu_check():  # choose based on available GPU RAM
            gpu_ram = _get_gpu_memory_map_mb()
            budget = min(gpu_ram.values()) * 0.9
            logging.info(
                "[checkmate] No budget specified; defaulting to the minimum amount of free GPU RAM on any single "
                "GPU, {0:.2f}MB".format(budget)
            )
        else:  # choose based available system memory
            budget = psutil.virtual_memory().available * 0.8 / 1000000
            logging.debug("[checkmate] No GPU detected, using system DRAM on CPU")
            logging.info("[checkmate] No budget specified; defaulting to {0:.2f}MB".format(budget))

    # build gradient function for model
    @tf.function
    def grads_check(data, label):
        with tf.GradientTape() as check_tape:
            predictions = model(data)
            loss_val = loss(label, predictions)
        gradients = check_tape.gradient(loss_val, model.trainable_variables)
        return predictions, loss_val, gradients

    fn = grads_check.get_concrete_function(input_spec, label_spec)
    g = dfgraph_from_tf_function(fn)

    # choose solver and calculate solver
    logging.error(
        "[checkmate] At the moment, Checkmate does not guarentee scheduling under the specified budget. "
        "This feature will appear soon."
    )
    logging.debug("[checkmate] Solving for recomputation schedule, may take a while")
    logging.debug("[checkmate] Using Chen et al. (2016) sqrt(n) algorithm")
    sched_result = solve_chen_sqrtn(g, True)
    logging.debug("[checkmate] Schedule solved")

    # create recomputed gradient function
    def clean_bs(tensorspec):
        newshape = list(tensorspec.shape)
        newshape[0] = None
        return tf.TensorSpec(shape=newshape, dtype=tensorspec.dtype)

    fn_nobatchsize = grads_check.get_concrete_function(clean_bs(input_spec), clean_bs(label_spec))
    grad_fn_check = edit_graph(fn_nobatchsize, g.op_dict, sched_result.schedule)

    @tf.function
    def train_step_check(data, labels):
        predictions, loss_val, gradients = grad_fn_check(data, labels)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return predictions, loss_val

    return train_step_check
