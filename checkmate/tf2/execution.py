import tensorflow as tf
from copy import deepcopy

from checkmate.core.schedule import (
    OperatorEvaluation,
    AllocateRegister,
    DeallocateRegister,
    Schedule
)


def can_replace(orig, replace):
    # if t1.op._original_op != None:
    #    o1 = t1.op._original_op
    # else:
    #    o1 = t1.op
    if replace.op._original_op is not None:
        replace_op = replace.op._original_op
    else:
        replace_op = replace.op
    return orig.op == replace_op


def copy_op(op, new_name):
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


def edit_graph(fxn, op_list,  schedule):
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


