import logging
from collections import defaultdict
from typing import Optional

import tensorflow as tf

from experiments.common.profile.cost_model import CostModel
from remat.core import dfgraph
from remat.tensorflow2.extraction_hooks import op_hook, MEMORY_MULTIPLIER

try:
    from tensorflow.python.keras.utils.layer_utils import count_params  # TF r2.0
except ImportError as e:
    # noinspection PyUnresolvedReferences
    from tensorflow.keras.backend import count_params  # TF r1.14


def dfgraph_from_keras(mod: tf.keras.models.Model, input_dep=False, output_dep=True, next_outputs_deps=True,
                       batch_size=1, loss_cpu_cost=0, loss_ram_cost=4, cost_model: Optional[CostModel] = None):
    """
    Given a Keras model, this method extracts a graph to be utilized by the solver
    :param mod: tf.keras.models.Model -- A Keras model
    :param next_outputs_deps: bool -- graph dependency on outputs of nodes that consume this node
    :param batch_size: int -- batch size for generated model
    :param loss_cpu_cost: int -- CPU cost to evaluate loss node
    :param loss_ram_cost: int -- RAM cost to store loss node output
    :param cost_model: CostModel object representing the costs loaded from disk
    :return: Graph -- graph generated from the Keras model
    """
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
        print(layer.name, [key for key, value in name_to_idx.items() if value in inbound_idx])
        for inbound_position, inbound_node in enumerate(filter(lambda x: x != -1, inbound_idx)):
            dep_list_fwd[layer_idx].append(inbound_node)  # forward dependency
            dep_list_bwd[fwd_to_bwd(inbound_node)].append(fwd_to_bwd(layer_idx))  # connect grad node to previous backward node
            if next_outputs_deps:  # connect output of node to the inbound node's gradient node
                dep_list_fwd[fwd_to_bwd(inbound_node)].append(layer_idx)
            if input_dep:
                dep_list_fwd[fwd_to_bwd(layer_idx)].append(inbound_node)
        if layer_idx == loss_node_idx - 1:  # inject loss node assuming we are at output node
            dep_list_fwd[loss_node_idx].append(layer_idx)
            dep_list_fwd[fwd_to_bwd(layer_idx)].append(loss_node_idx)
        if output_dep:  # connect output of node to corresponding backwards node
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

    if cost_model is not None:
        costs_np = cost_model.get_costs(batch_size)
        if len(costs_np) == len(costs):
            costs = dict(enumerate(costs_np))
        else:
            logging.error("Wrong cost file!")

    vfwd = list(range(len(layers)))
    vfwd_map = {u_fwd: fwd_to_bwd(u_fwd) for u_fwd in vfwd}
    vback = [fwd_to_bwd(u_fwd) for u_fwd in vfwd]
    idx_to_name = {v: u for u, v in name_to_idx.items()}
    names = {u: idx_to_name[u] for u in vfwd}

    # Get parameter and gradient momentum memory usage
    total_params = sum(count_params_keras(mod))
    total_mem_params = total_params * MEMORY_MULTIPLIER

    return dfgraph.DFGraph(args=args, v=vfwd + [loss_node_idx] + vback, vfwd_map=vfwd_map,
                           vloss=loss_node_idx, cost_cpu=costs, cost_ram=mems, node_names=names,
                           cost_ram_parameters=total_mem_params)


# noinspection PyProtectedMember
def count_params_keras(mod: tf.keras.models.Model):
    mod._check_trainable_weights_consistency()
    if hasattr(mod, '_collected_trainable_weights'):
        trainable_count = count_params(mod._collected_trainable_weights)
    elif hasattr(mod, '_unique_trainable_weights'):
        trainable_count = count_params(mod._unique_trainable_weights)  # TF r2.0
    else:
        trainable_count = count_params(mod.trainable_weights)  # TF r1.14

    non_trainable_count = count_params(mod.non_trainable_weights)
    return trainable_count, non_trainable_count
