from copy import deepcopy

import tensorflow as tf

from checkmate.core.schedule import OperatorEvaluation, Schedule


def can_replace(orig, replace):
    # if t1.op._original_op != None:
    #    o1 = t1.op._original_op
    # else:
    #    o1 = t1.op
    replace_op = replace.op if replace.op._original_op is None else replace.op._original_op
    return orig.op == replace_op and orig.value_index == replace.value_index


def copy_op(op, new_name):
    nnd = deepcopy(op.node_def)
    nnd.name = new_name
    nod = deepcopy(op.op_def)
    output_types = op._output_types[:]
    input_types = op._input_types[:]
    control_inputs = op.control_inputs[:]
    new_op = tf.Operation(nnd, op.graph, list(op.inputs), output_types, control_inputs, input_types, op, nod)
    return new_op


def edit_graph(fxn, op_dict, schedule: Schedule):
    registers = dict()
    output_ops = [t.op for t in fxn.outputs]
    for i, inst in enumerate(schedule):
        if type(inst) == OperatorEvaluation:
            args = [arg for i in inst.arg_regs for arg in registers[i]]
            op = op_dict[inst.id]
            # assert len(op.outputs) == 1, "op {} which output two tensors not yet supported".format(op.name)

            # duplicate rematerialized operation
            new_op = op if op in output_ops else copy_op(op, "{}_copy_at_sched_idx{}".format(op.name, i))

            # match up args, and assign outputs
            for arg in args:
                for j, inp in enumerate(new_op.inputs):
                    if can_replace(inp, arg):
                        new_op._update_input(j, arg)
            registers[inst.out_register] = new_op.outputs
    return fxn
