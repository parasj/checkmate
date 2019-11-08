import itertools
import logging

from remat.core.dfgraph import DFGraph
from remat.core.schedule import Schedule, ScheduledResult, AllocateRegister, DeallocateRegister, OperatorEvaluation
from remat.tensorflow2.tf_losses import categorical_cross_entropy

import tensorflow as tf

logger = logging.getLogger(__name__)


def match_variables(_param_grads_dict, _model):
    for grads_idx, layer in enumerate(_model.layers[1:]):
        grads = _param_grads_dict.get(grads_idx, [])
        for grad, variable in zip(grads, layer.trainable_variables):
            yield variable.name, grad


def sort_by_dep_order(nodes, deps_list):
    layers_to_dep_position = {layer_id: position for position, layer_id in enumerate(deps_list)}
    return list(sorted(nodes, key=layers_to_dep_position.get))


def tfgraph_from_schedule(model, g: DFGraph, scheduled_result: ScheduledResult,
                          loss=categorical_cross_entropy):
    def ids_to_names(ids):
        names = []
        for id in ids:
            if g.is_loss_node(id):
                names.append("∇Loss")
            elif g.is_backward_node(id):
                names.append("∇" + g.node_names.get(g.backward_to_forward(id)))
            else:
                names.append(g.node_names.get(id))
        return names

    def _eager_eval(input_val: tf.Tensor, label_val: tf.Tensor):
        layers = model.layers[1:]  # ignore input layer
        param_grads = {}
        tapes = {}
        regs = {}
        our_loss = -1
        for op in scheduled_result.schedule:
            if isinstance(op, AllocateRegister):
                pass
            elif isinstance(op, DeallocateRegister):
                if op.register_id in regs:
                    del regs[op.register_id]
                if op.register_id in tapes:
                    del tapes[op.register_id]
            elif isinstance(op, OperatorEvaluation) and g.is_forward_node(op.id):
                idx = op.id
                layer = layers[idx]
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    input_layers = sort_by_dep_order(op.arg_regs.keys(), g.args[idx])
                    inputs = [regs[op.arg_regs[arg_layer_id]] for arg_layer_id in input_layers]
                    inputs = inputs if len(inputs) > 0 else [input_val]  # if first node
                    logger.debug(f"reg[{op.out_register}] ⟵ {layer.name}({[op.arg_regs[x] for x in input_layers]}) id#{op.id}")
                    for var in itertools.chain(inputs, layer.variables):
                        tape.watch(var)
                    if len(inputs) > 1:
                        regs[op.out_register] = tf.stop_gradient(layer(inputs))
                    else:
                        regs[op.out_register] = tf.stop_gradient(layer(*inputs))
                tapes[op.out_register] = tape
            elif isinstance(op, OperatorEvaluation) and g.is_loss_node(op.id):
                source_inputs = [regs[op.arg_regs[arg_idx]] for arg_idx in sorted(op.arg_regs.keys())]
                inputs = source_inputs + [label_val]
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    for x in inputs:
                        tape.watch(x)
                    regs[op.out_register] = loss(*inputs)
                tapes[op.out_register] = tape
                our_loss = regs[op.out_register]
                regs[op.out_register] = tape.gradient(our_loss, source_inputs)
                logger.debug(f"reg[{op.out_register}] ⟵ loss_fn ({our_loss})")
                # TODO: fix memory accounting to consider the size of the loss node gradient rather than the actual kiss
            elif isinstance(op, OperatorEvaluation) and g.is_backward_node(op.id):
                idx = g.backward_to_forward(op.id)
                layer = layers[idx]
                # assert not set(op.arg_regs.values()) - set(regs.keys()), "Missing dependency"

                logger.debug(f"reg[{op.out_register}] ⟵ ∇ {layer.name}")
                fwd_nodes = list(filter(lambda x: g.is_forward_node(x), op.arg_regs.keys()))
                bwd_nodes = list(filter(lambda x: g.is_loss_node(x) or g.is_backward_node(x), op.arg_regs.keys()))
                in_idxs = sort_by_dep_order(fwd_nodes, g.args[idx])
                out_grad_idxs = sort_by_dep_order(bwd_nodes, [g.forward_to_backward(x) for x in g.args[idx]])
                logger.debug(f"\targnodes {op.arg_regs}")
                logger.debug(f"\tForward deps {ids_to_names(in_idxs)}")
                logger.debug(f"\tBackprop deps {ids_to_names(bwd_nodes)}")

                # in_tensors = [regs[op.arg_regs[x]] for x in in_idxs]
                # out_tensors = []
                # out_grad_tensors = [regs[op.arg_regs[x]] for x in out_grad_idxs]

                # X_idx, *Y_list_idx = sorted(fwd_nodes)  # yank max node as output node
                # Y_list_idx = sort_by_dep_order(Y_list_idx, g.args[op.id], is_backward=True)
                # X_reg = regs[op.arg_regs[X_idx]]
                # Y_reg_list = [regs[op.arg_regs[y_idx]] for y_idx in Y_list_idx]
                # if debug: logging.debug(f"\t⮑ X {op.arg_regs[X_idx]} ({X_reg})")
                # if debug: logging.debug(f"\t⮑ Y {[op.arg_regs[y] for y in Y_list_idx]} ({Y_reg_list})")

                # dLdY_list = []
                # for x in Y_list_idx:
                #     if not g.is_loss_node(x):
                #         backward = g.forward_to_backward(x)
                #         dldy_reg_idx = op.arg_regs[backward]
                #         dldy = regs[dldy_reg_idx]
                #         if debug: logging.debug(f"\t⮑ Y {op.arg_regs[x]} ⟶\tdLdY {dldy_reg_idx} ({dldy})")
                #     else:
                #         dldy = None
                #     dLdY_list.append(dldy)
                #
                # param_grad_list = []
                # act_grad_list = []
                # param_tape = tapes[op.arg_regs[X_idx]]
                # for x_idx, y, dLdY in zip(Y_list_idx, Y_reg_list, dLdY_list):
                #     loss_tape = tapes[op.arg_regs[x_idx]]
                #     dLdX = loss_tape.gradient(y, X_reg, output_gradients=dLdY)
                #     act_grad_list.append(dLdX)
                #     if layer.trainable_variables:
                #         param_grad_list.append(
                #             param_tape.gradient(X_reg, layer.trainable_variables, output_gradients=dLdX))
                # if debug: logging.debug(f"\t⮑ Param grad list: {param_grad_list}")
                # if debug: logging.debug(f"\t⮑ Activation grad list: {act_grad_list}")
                # if layer.trainable_variables:
                #     param_grads[idx] = list(map(tf.add_n, [list(items) for items in zip(*param_grad_list)]))
                # regs[op.out_register] = tf.math.add_n(act_grad_list)
            else:
                raise ValueError("Unknown operator " + str(op))
        # grad_dict = dict(match_variables(param_grads, model))
        # out_grads = [grad_dict[v.name] for v in model.trainable_variables]
        out_grads = []
        return our_loss, out_grads
    return _eager_eval
