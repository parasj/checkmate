import os

import tensorflow as tf

from integration.tf2.misc import categorical_cross_entropy
from remat.core.schedule import OperatorEvaluation, AllocateRegister, DeallocateRegister, Schedule
from remat.core.graph import Graph
from utils.setup_logger import setup_logger


class TF2Runner:
    def __init__(self, keras_model: tf.keras.models.Model, g: Graph, schedule: Schedule,
                 loss_fn=categorical_cross_entropy, eager: bool = True, log_base: str = None, debug=False, batch_size=None):
        self.log_base = log_base
        self.logger = setup_logger("TF2Runner", os.path.join(log_base, 'TF2Runner.log'))
        self.debug = debug
        self.schedule = schedule
        self.eager = eager
        self.batch_size = batch_size

        self.loss_fn = loss_fn
        self.keras_model = keras_model
        self.g = g

        self.tf_graph = self._generate_tf_graph()

    def _generate_tf_graph(self):
        if self.eager:
            #raise NotImplementedError("Disable eager")
            return self._tf_eager_eval
        #raise NotImplementedError("Disable static")
        model = self.keras_model
        in_shape = list(model.input_shape)
        out_shape=list(model.output_shape)
        in_shape[0] = self.batch_size
        out_shape[0] = self.batch_size
        input_sig = [tf.TensorSpec(in_shape), tf.TensorSpec(out_shape)]
        #return self._tf_static_eval
        static_tf_graph = tf.function(self._tf_static_eval, autograph=False)
        return static_tf_graph.get_concrete_function(*input_sig)

    def _tf_eager_eval(self, input_val: tf.Tensor, label_val: tf.Tensor):
        #raise NotImplementedError("Disable eager")
        def match_variables(_param_grads_dict, _model):
            for grads_idx, layer in enumerate(_model.layers[1:]):
                grads = _param_grads_dict.get(grads_idx, [])
                for grad, variable in zip(grads, layer.trainable_variables):
                    yield variable.name, grad

        def sort_by_dep_order(nodes, deps_list, is_backward=False):
            output_nodes = [] if not is_backward else [i for i in nodes if i not in deps_list]
            input_nodes = [i for i in nodes if i not in output_nodes]
            layers_to_dep_position = {layer_id: position for position, layer_id in enumerate(deps_list)}
            return list(sorted(input_nodes, key=layers_to_dep_position.get)) + output_nodes

        layers = self.keras_model.layers[1:]  # ignore input layer
        param_grads = {}
        tapes = {}
        regs = {}
        our_loss = -1
        print('\nreee\n' * 20)
        for op in self.schedule:
            if isinstance(op, AllocateRegister):
                pass
            elif isinstance(op, DeallocateRegister):
                if op.register_id in regs:
                    del regs[op.register_id]
                if op.register_id in tapes:
                    del tapes[op.register_id]
            elif isinstance(op, OperatorEvaluation) and self.g.is_forward_node(op.id):
                idx = op.id
                layer = layers[idx]
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    input_layers = sort_by_dep_order(op.arg_regs.keys(), self.g.args[idx])
                    if self.debug: self.logger.debug(f"\t⮑ {[op.arg_regs[x] for x in input_layers]}")
                    inputs = [regs[op.arg_regs[arg_layer_id]] for arg_layer_id in input_layers]
                    inputs = inputs if len(inputs) > 0 else [input_val]  # if first node
                    if self.debug: self.logger.info(f"reg[{op.out_register}] ⟵ {layer.name}")
                    if self.debug: self.logger.debug(f"\t⮑ Inputs = {inputs}")
                    #for var in itertools.chain(inputs, layer.variables):
                    #    if self.debug: self.logger.debug(f"\t⮑ Watching variable {var}")
                    #    tape.watch(var)
                    if len(inputs) > 1:
                        regs[op.out_register] = tf.stop_gradient(layer(inputs))
                    else:
                        regs[op.out_register] = tf.stop_gradient(layer(*inputs))
                if self.debug: self.logger.debug(f"\t⮑ Wrote {regs[op.out_register]} to {op.out_register}")
                tapes[op.out_register] = tape
            elif isinstance(op, OperatorEvaluation) and self.g.is_loss_node(op.id):
                with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                    inputs = [regs[op.arg_regs[arg_idx]] for arg_idx in sorted(op.arg_regs.keys())]
                    inputs += [label_val]
                    #for x in inputs:
                    #    tape.watch(x)
                    regs[op.out_register] = self.loss_fn(*inputs)
                tapes[op.out_register] = tape
                our_loss = regs[op.out_register]
                if self.debug: self.logger.info(f"reg[{op.out_register}] ⟵ loss_fn ({our_loss})")
            elif isinstance(op, OperatorEvaluation):
                idx = self.g.backward_to_forward(op.id)
                layer = layers[idx]
                assert not set(op.arg_regs.values()) - set(regs.keys()), "Missing dependency"

                if self.debug: self.logger.info(f"reg[{op.out_register}] ⟵ ∇ {layer.name}")
                fwd_nodes = list(filter(lambda x: not self.g.is_backward_node(x), op.arg_regs.keys()))
                X_idx, *Y_list_idx = sorted(fwd_nodes)  # yank max node as output node
                Y_list_idx = sort_by_dep_order(Y_list_idx, self.g.args[op.id], is_backward=True)
                X_reg = regs[op.arg_regs[X_idx]]
                Y_reg_list = [regs[op.arg_regs[y_idx]] for y_idx in Y_list_idx]
                if self.debug: self.logger.debug(f"\t⮑ X {op.arg_regs[X_idx]} ({X_reg})")
                if self.debug: self.logger.debug(f"\t⮑ Y {[op.arg_regs[y] for y in Y_list_idx]} ({Y_reg_list})")

                dLdY_list = []
                for x in Y_list_idx:
                    if not self.g.is_loss_node(x):
                        backward = self.g.forward_to_backward(x)
                        dldy_reg_idx = op.arg_regs[backward]
                        dldy = regs[dldy_reg_idx]
                        if self.debug: self.logger.debug(f"\t⮑ Y {op.arg_regs[x]} ⟶\tdLdY {dldy_reg_idx} ({dldy})")
                    else:
                        dldy = None
                    dLdY_list.append(dldy)

                param_grad_list = []
                act_grad_list = []
                param_tape = tapes[op.arg_regs[X_idx]]
                for x_idx, y, dLdY in zip(Y_list_idx, Y_reg_list, dLdY_list):
                    loss_tape = tapes[op.arg_regs[x_idx]]
                    dLdX = loss_tape.gradient(y, X_reg, output_gradients=dLdY,
                                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
                    
                    act_grad_list.append(dLdX)
                    if layer.trainable_variables:
                        param_grad_list.append(param_tape.gradient(X_reg, layer.trainable_variables, output_gradients=dLdX, unconnected_gradients=tf.UnconnectedGradients.ZERO))
                if self.debug: self.logger.debug(f"\t⮑ Param grad list: {param_grad_list}")
                if self.debug: self.logger.debug(f"\t⮑ Activation grad list: {act_grad_list}")
                if layer.trainable_variables:
                    param_grads[idx] = list(map(tf.add_n, [list(items) for items in zip(*param_grad_list)]))
                regs[op.out_register] = tf.math.add_n(act_grad_list)
            else:
                raise ValueError("Unknown operator " + str(op))
        grad_dict = dict(match_variables(param_grads, self.keras_model))
        out_grads = [grad_dict[v.name] for v in self.keras_model.trainable_variables]
        return our_loss, out_grads
    
    def _tf_static_eval(self, input_val: tf.Tensor, label_val: tf.Tensor):
        def match_variables(_param_grads_dict, _model):
            for grads_idx, layer in enumerate(_model.layers[1:]):
                grads = _param_grads_dict.get(grads_idx, [])
                for grad, variable in zip(grads, layer.trainable_variables):
                    yield variable.name, grad

        def sort_by_dep_order(nodes, deps_list, is_backward=False):
            output_nodes = [] if not is_backward else [i for i in nodes if i not in deps_list]
            input_nodes = [i for i in nodes if i not in output_nodes]
            layers_to_dep_position = {layer_id: position for position, layer_id in enumerate(deps_list)}
            return list(sorted(input_nodes, key=layers_to_dep_position.get)) + output_nodes

        layers = self.keras_model.layers[1:]  # ignore input layer
        param_grads = {}
        regs = {}
        our_loss = -1
        for op in self.schedule:
            if isinstance(op, DeallocateRegister):
                if op.register_id in regs:
                    del regs[op.register_id]
            elif isinstance(op, OperatorEvaluation) and self.g.is_forward_node(op.id):
                idx = op.id
                layer = layers[idx]
                input_layers = sort_by_dep_order(op.arg_regs.keys(), self.g.args[idx])
                if self.debug: self.logger.debug(f"\t⮑ {[op.arg_regs[x] for x in input_layers]}")
                inputs = [regs[op.arg_regs[arg_layer_id]] for arg_layer_id in input_layers]
                inputs = inputs if len(inputs) > 0 else [input_val]  # if first node
                if len(inputs) > 1:
                    regs[op.out_register] = tf.stop_gradient(layer(inputs))
                else:
                    regs[op.out_register] = tf.stop_gradient(layer(*inputs))
            elif isinstance(op, OperatorEvaluation) and self.g.is_loss_node(op.id):
                inputs = [regs[op.arg_regs[arg_idx]] for arg_idx in sorted(op.arg_regs.keys())]
                inputs += [label_val]
                regs[op.out_register] = self.loss_fn(*inputs)
                our_loss = regs[op.out_register]
            elif isinstance(op, OperatorEvaluation):
                idx = self.g.backward_to_forward(op.id)
                layer = layers[idx]
                fwd_nodes = list(filter(lambda x: not self.g.is_backward_node(x), op.arg_regs.keys()))
                X_idx, *Y_list_idx = sorted(fwd_nodes)  # yank max node as output node
                Y_list_idx = sort_by_dep_order(Y_list_idx, self.g.args[op.id], is_backward=True)
                X_reg = regs[op.arg_regs[X_idx]]
                Y_reg_list = [regs[op.arg_regs[y_idx]] for y_idx in Y_list_idx]
                if self.debug: self.logger.debug(f"\t⮑ X {op.arg_regs[X_idx]} ({X_reg})")
                if self.debug: self.logger.debug(f"\t⮑ Y {[op.arg_regs[y] for y in Y_list_idx]} ({Y_reg_list})")

                dLdY_list = []
                for x in Y_list_idx:
                    if not self.g.is_loss_node(x):
                        backward = self.g.forward_to_backward(x)
                        dldy_reg_idx = op.arg_regs[backward]
                        dldy = regs[dldy_reg_idx]
                    else:
                        dldy = None
                    dLdY_list.append(dldy)

                param_grad_list = []
                act_grad_list = []
                for x_idx, y, dLdY in zip(Y_list_idx, Y_reg_list, dLdY_list):
                    if(dLdY is not None):
                        dLdY = tf.squeeze(dLdY)

                    dLdX = tf.gradients(y, X_reg, grad_ys=dLdY, name=f"grad_{layer.name}", unconnected_gradients=tf.UnconnectedGradients.ZERO)

                    act_grad_list.append(dLdX)
                    if layer.trainable_variables:
                        param_grad_list.append(tf.gradients(X_reg, layer.trainable_variables, grad_ys=dLdX, unconnected_gradients=tf.UnconnectedGradients.ZERO))
                if self.debug: self.logger.debug(f"\t⮑ Param grad list: {param_grad_list}")
                if self.debug: self.logger.debug(f"\t⮑ Activation grad list: {act_grad_list}")
                if layer.trainable_variables:
                    param_grads[idx] = list(map(tf.add_n, [list(items) for items in zip(*param_grad_list)]))
                regs[op.out_register] = tf.math.add_n(act_grad_list)
        grad_dict = dict(match_variables(param_grads, self.keras_model))
        out_grads = [grad_dict[v.name] for v in self.keras_model.trainable_variables]
        return our_loss, out_grads

