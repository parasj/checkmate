import logging

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from copy import deepcopy

# from tensorflow.python.ops import gradients_util as tfg
from checkmate.core.schedule import (
    OperatorEvaluation,
    AllocateRegister,
    DeallocateRegister,
    Schedule,
)

# tf.compat.v1.disable_eager_execution()
logging.basicConfig(level=logging.DEBUG)


class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MLPBlock(layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        # self.linear_2 = Linear(32)
        # self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        z = tf.nn.relu(x)
        # y = tf.matmul(z,x)
        y = tf.reduce_mean(z)
        return y  # loss


@tf.function
def get_grads(f, inputs, tvars):
    y = f(inputs)
    return tf.gradients(y, tvars)


mlp = MLPBlock()

x = tf.ones(shape=(32, 32), name="x")
spec = tf.TensorSpec((32, 32))
conc = tf.function(mlp).get_concrete_function(x)
grad_conc = get_grads.get_concrete_function(mlp, x, mlp.trainable_variables)

op_list = grad_conc.graph.get_operations()
exclude_list = [
    "Placeholder",
    "ReadVariableOp",
    "Const",
    "BroadcastGradientArgs",
    "Fill",
]


def copy_op(
    op, new_name
):  # taken from "tensorflow/contrib/copy_graph/python/util/copy_elements.py"
    nnd = deepcopy(op.node_def)
    nnd.name = new_name
    nod = deepcopy(op.op_def)
    output_types = op._output_types[:]
    input_types = op._input_types[:]
    control_inputs = op.control_inputs[:]
    new_op = tf.Operation(
        nnd,
        op.graph,
        list(op.inputs),
        output_types,
        control_inputs,
        input_types,
        op,
        nod,
    )
    return new_op


op_list = [op for op in op_list if op.type not in exclude_list]

op_dict = dict(zip(op_list, range(len(op_list))))

adj_list = {}
# what should we not mess with
for i in range(len(op_list)):
    adj_list[i] = []
for op in op_list:
    in_id = op_dict[op]
    for out in op.outputs:
        for cons in out.consumers():
            if cons in op_list:
                # adj_list[in_id].append(op_dict[cons])
                adj_list[op_dict[cons]].append(in_id)

# all forward schedule
schedule = []
reg_counter = 0
needed = {k: len(v) for k, v in adj_list.items()}
# print(needed)

for op in op_list:
    iid = op_dict[op]
    schedule.append(AllocateRegister(iid, iid, np.prod(op.outputs[0].shape)))

    schedule.append(OperatorEvaluation(iid, adj_list[iid], iid, 9, False, True))

    for inp in op.inputs:
        if inp.op in op_list:
            needed[op_dict[inp.op]] -= 1
            if needed[op_dict[inp.op]] == 0:
                schedule.append(DeallocateRegister(iid, iid))


# print(schedule)


def can_replace(orig, replace):
    # if t1.op._original_op != None:
    #    o1 = t1.op._original_op
    # else:
    #    o1 = t1.op
    if replace.op._original_op != None:
        replace_op = replace.op._original_op
    else:
        replace_op = replace.op
    return orig.op == replace_op


# destructive oops
def execute(fxn, op_list, op_dict, schedule, samp_inputs):
    registers = [None] * len(op_list)
    output_ops = [t.op for t in fxn.outputs]
    # run the schedule
    name_fmt = "{}_Copy_{}"
    for i, inst in enumerate(schedule):
        if type(inst) == OperatorEvaluation:
            args = [registers[i] for i in inst.arg_regs]
            op = op_list[inst.id]
            assert (
                len(op.outputs) == 1
            ), "ops which output two tensors not yet supported"

            if op in output_ops:
                new_op = op  #
            else:
                new_op = copy_op(op, name_fmt.format(op.name, i))
            # match up args
            for arg in args:
                for j, inp in enumerate(new_op.inputs):
                    if can_replace(inp, arg):
                        new_op._update_input(j, arg)
            registers[inst.out_register] = new_op.outputs[0]
