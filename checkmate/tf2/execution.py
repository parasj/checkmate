from copy import deepcopy

import tensorflow as tf

from checkmate.core.schedule import OperatorEvaluation, Schedule


def can_replace(orig, replace):
    replace_op = replace.op if replace.op._original_op is None else replace.op._original_op
    return orig.op == replace_op and orig.value_index == replace.value_index


def copy_op(op, new_name):
    nnd = deepcopy(op.node_def)  # ~7% of test script runtime
    nnd.name = new_name
    op_def = deepcopy(op.op_def)  # ~2% of test script runtime
    new_op = tf.Operation(  # ~22% of test script runtime
        nnd,
        op.graph,
        list(op.inputs),
        op._output_types[:],  # ~2% of runtime
        op.control_inputs[:],
        op._input_types[:],
        op,
        op_def,
    )
    return new_op


def edit_graph(fxn, op_dict, schedule: Schedule):
    registers = dict()
    output_ops = set(t.op for t in fxn.outputs)

    duplicate_ops = []
    sched_ordered = list(enumerate([s for s in schedule if isinstance(s, OperatorEvaluation)]))

    # duplicate rematerialized operation
    for i, inst in sched_ordered:
        op = op_dict[inst.id]
        new_op = op if op in output_ops else copy_op(op, "{}_copy_at_sched_idx{}".format(op.name, i))
        duplicate_ops.append(new_op)

    # match up args, and assign outputs
    for i, inst in sched_ordered:
        args = [arg for i in inst.arg_regs for arg in registers[i]]
        new_op = duplicate_ops[i]
        for arg in args:
            for j, inp in enumerate(new_op.inputs):
                if can_replace(inp, arg):
                    new_op._update_input(j, arg)
        registers[inst.out_register] = new_op.outputs

    return fxn
