from collections import defaultdict
from typing import Optional, List

import keras_segmentation
import numpy as np
import tensorflow.compat.v2 as tf

from remat.core import graph
from utils.setup_logger import setup_logger

try:
    from tensorflow.python.keras.utils.layer_utils import count_params  # TF r2.0
except ImportError as e:
    from tensorflow.keras.backend import count_params  # TF r1.14

from integration.tf2.hooks import op_hook, MEMORY_MULTIPLIER

KERAS_APPLICATION_MODEL_NAMES = ['InceptionV3', 'VGG16', 'VGG19', 'ResNet50',
                                 'Xception', 'MobileNet', 'MobileNetV2', 'DenseNet121',
                                 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                                 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2',
                                 'ResNet152V2']
SEGMENTATION_MODEL_NAMES = list(keras_segmentation.models.model_from_name.keys())
MODEL_NAMES = KERAS_APPLICATION_MODEL_NAMES + SEGMENTATION_MODEL_NAMES + ["test"]

CHAIN_GRAPH_MODELS = ["VGG16", "VGG19", "MobileNet"]

NUM_SEGMENTATION_CLASSES = 19  # Cityscapes has 19 evaluation classes


def pretty_model_name(model_name: str):
    mapping = {
        "vgg_unet": "U-Net with VGG16",
    }
    if model_name in mapping:
        return mapping[model_name]
    return model_name


def pretty_platform_name(platform: str):
    mapping = {
        "p32xlarge": "V100",
        "p32xlarge_fp16": "V100, fp16",
        "p2xlarge": "K80",
        "flops": "FLOPs",
    }
    if platform in mapping:
        return mapping[platform]
    return platform


def platform_memory(platform: str):
    mapping = {
        "p32xlarge": 16 * 1000 * 1000 * 1000,
        "p32xlarge_fp16": 16 * 1000 * 1000 * 1000,
        "p2xlarge": 12 * 1000 * 1000 * 1000,
        "flops": 12 * 1000 * 1000 * 1000,
    }
    if platform in mapping:
        return mapping[platform]
    return platform


def simple_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='in_conv')(inputs)
    a = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', name='conv1')(x)
    b = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', name='conv2')(x)
    c = tf.keras.layers.Add(name='addc1c2')([a, b])
    d = tf.keras.layers.GlobalAvgPool2D(name='flatten')(c)
    predictions = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(d)
    return tf.keras.Model(inputs=inputs, outputs=predictions)


def get_keras_model(model_name: str, input_shape: Optional[List[int]] = None):
    if model_name is "test":
        model = simple_model()
    elif model_name in KERAS_APPLICATION_MODEL_NAMES:
        # Pre-trained Keras applications
        model = eval("tf.keras.applications.{}".format(model_name))
        model = model(input_shape=input_shape)
    elif model_name in SEGMENTATION_MODEL_NAMES:
        # Segmentation models
        model = keras_segmentation.models.model_from_name[model_name]
        if input_shape is not None:
            assert input_shape[2] == 3, "Can only segment 3-channel, channel-last images"

            model = model(n_classes=NUM_SEGMENTATION_CLASSES,
                          input_height=input_shape[0],
                          input_width=input_shape[1])
        else:
            model = model(n_classes=NUM_SEGMENTATION_CLASSES)
    else:
        raise NotImplementedError("Model {} not available".format(model_name))

    return model


def get_input_shape(model_name: str, batch_size: Optional[int] = None):
    model = get_keras_model(model_name, input_shape=None)
    shape = model.layers[0].input_shape
    if batch_size is not None:
        shape[0] = batch_size
    return shape


def count_params_keras(model: tf.keras.models.Model):
    model._check_trainable_weights_consistency()
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    elif hasattr(model, '_unique_trainable_weights'):
        trainable_count = count_params(model._unique_trainable_weights)  # TF r2.0
    else:
        trainable_count = count_params(model.trainable_weights)  # TF r1.14
    # print("Trainable params:", trainable_count)

    non_trainable_count = count_params(model.non_trainable_weights)
    # print("Non-trainable params:", non_trainable_count)
    return trainable_count, non_trainable_count


def extract_graph_from_keras(mod: tf.keras.models.Model,
                             include_prev_node=True,
                             batch_size=1,
                             loss_cpu_cost=0,
                             loss_ram_cost=0,
                             costs_np=Optional[np.ndarray]):
    """
    Given a Keras model, this method extracts a graph to be utilized by the solver
    :param mod: tf.keras.models.Model -- A Keras model
    :param include_prev_node: bool -- If true, insert nodes from y to dy/dx (required by some autodiff engines)
    :param batch_size: int -- batch size for generated model
    :param loss_cpu_cost: int -- CPU cost to evaluate loss node
    :param loss_ram_cost: int -- RAM cost to store loss node output
    :param costs_np:
    :return: Graph -- graph generated from the Keras model
    """
    logger = setup_logger("KerasGraphExtractor")
    assert batch_size > 0
    layers = mod.layers[1:]
    loss_node_idx = len(layers)
    size = len(layers) + 1 + len(layers)  # loss node plus corresponding back nodes

    fwd_to_bwd = lambda idx: (size - 1) - idx
    name_to_idx = {mod.layers[0].name: -1}
    relevant_nodes = sum(mod._nodes_by_depth.values(), [])
    # build argument list in order of dependencies
    dep_list_fwd = defaultdict(list)
    dep_list_bwd = defaultdict(list)  # joined with dep_list_fwd in order to ensure backward nodes are last
    for layer_idx, layer in enumerate(layers):
        name_to_idx[layer.name] = layer_idx
        inbound_idx = [name_to_idx[t[0].name] for node in layer._inbound_nodes for t in node.iterate_inbound() if
                       node in relevant_nodes]
        for inbound_position, inbound_node in enumerate(filter(lambda x: x != -1, inbound_idx)):
            dep_list_fwd[layer_idx].append(inbound_node)
            dep_list_bwd[fwd_to_bwd(inbound_node)].append(fwd_to_bwd(layer_idx))
            if include_prev_node:  # hopefully not needed: this is using y to compute dy/dx
                dep_list_fwd[fwd_to_bwd(inbound_node)].append(layer_idx)
        if layer_idx == loss_node_idx - 1:  # inject loss node assuming we are at output node
            dep_list_fwd[loss_node_idx].append(layer_idx)
            dep_list_fwd[fwd_to_bwd(layer_idx)].append(loss_node_idx)
        dep_list_fwd[fwd_to_bwd(layer_idx)].append(layer_idx)
    args = {i: dep_list_fwd[i] + dep_list_bwd[i] for i in set(dep_list_fwd.keys()).union(set(dep_list_bwd.keys()))}

    # Get per-node compute costs and activation memory usages
    costs = {loss_node_idx: loss_cpu_cost}
    mems = {loss_node_idx: loss_ram_cost}
    for i, layer in enumerate(layers):
        c, m = op_hook(layer, batch_size=batch_size)
        costs[i] = c
        costs[fwd_to_bwd(i)] = 2 * c
        mems[i] = m
        mems[fwd_to_bwd(i)] = m

    # Get per-node compute costs and activation memory usages
    for i, layer in enumerate(layers):
        c, m = op_hook(layer, batch_size=batch_size)
        costs[i] = c
        costs[fwd_to_bwd(i)] = 2 * c
        mems[i] = m
        mems[fwd_to_bwd(i)] = m

    if costs_np is not None:
        if len(costs_np) == len(costs):
            logger.info(f"Costs imported successfully with {len(costs)} items")
            costs = dict(enumerate(costs_np))
        else:
            logger.error("Wrong cost file!")
            logger.error("Num operations in original:\t{}".format(len(costs)))
            logger.error("Num operations in cost file:\t{}".format(len(costs_np)))

    vfwd = list(range(len(layers)))
    vfwd_map = {u_fwd: fwd_to_bwd(u_fwd) for u_fwd in vfwd}
    vback = [fwd_to_bwd(u_fwd) for u_fwd in vfwd]
    idx_to_name = {v: u for u, v in name_to_idx.items()}
    names = {u: idx_to_name[u] for u in vfwd}

    # Get parameter and gradient momentum memory usage
    total_params = sum(count_params_keras(mod))
    total_mem_params = total_params * MEMORY_MULTIPLIER

    return graph.Graph(args=args, v=vfwd + [loss_node_idx] + vback, vfwd_map=vfwd_map,
                       vloss=loss_node_idx, cost_cpu=costs, cost_ram=mems, node_names=names,
                       cost_ram_parameters=total_mem_params)
