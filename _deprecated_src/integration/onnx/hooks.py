import numpy as np

MEMORY_MULTIPLIER = 4  # 4 bytes per variable
LAST_DIMS = None


def get_attr(node, name, typ="ints"):
    out = []
    for attr in node.attribute:
        if attr.name == name:
            for val in eval("attr.{}".format(typ)):
                out.append(val)
    return tuple(out)


def conv_hook(node, inputs, feats, outputs):
    if len(inputs) != 1:
        print('2many')
        print(inputs)
    inputs = inputs[0]
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    weight = ''
    for k in feats:
        if "weight" in k:
            weight = feats[k]

    cout = weight[0]
    cin = weight[1]
    kernel = weight[2:]
    kernel2 = get_attr(node, 'kernel_shape')
    batch = inputs[0]
    if kernel != kernel2:
        print('KERNEL MISMATCH:\t1:{}\t2:{}'.format(kernel, kernel2))
    ops_per_output = np.prod(kernel) * cin
    ops = ops_per_output * mem_cost
    global LAST_DIMS
    LAST_DIMS = outputs
    return ops, mem_cost

    # NCHW


def bn_hook(node, inputs, feats, outputs):
    if len(inputs) != 1:
        print('2many')
        print(inputs)
    inputs = inputs[0]
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = 4 * np.prod(inputs)
    return ops, mem_cost


def relu_hook(node, inputs, feats, outputs):
    if len(inputs) != 1:
        print('2many')
        print(inputs)
    inputs = inputs[0]
    ops = np.prod(inputs)
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    return ops, mem_cost


def pool_hook(node, inputs, feats, outputs):
    if len(inputs) != 1:
        print('2many')
        print(inputs)
    inputs = inputs[0]
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    kernel = get_attr(node, 'kernel_shape')
    ops_per_output = np.prod(kernel)
    ops = ops_per_output * np.prod(outputs)
    global LAST_DIMS
    LAST_DIMS = outputs
    return ops, mem_cost


def add_hook(node, inputs, feats, outputs):
    ops = len(inputs) * np.prod(inputs[0])
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    global LAST_DIMS
    LAST_DIMS = outputs

    return ops, mem_cost


def const_hook(node, inputs, feats, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def gather_hook(node, inputs, feats, outputs):
    # stop gap as f
    mem_cost = np.prod(LAST_DIMS) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def concat_hook(node, inputs, feats, outputs):
    ops = 0
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    if mem_cost < np.prod(LAST_DIMS) * MEMORY_MULTIPLIER:
        print('concat debug')
        print(node.output)
        return ops, np.prod(LAST_DIMS) * MEMORY_MULTIPLIER

    return ops, mem_cost


def reshape_hook(node, inputs, feats, outputs):
    mem_cost = np.prod(LAST_DIMS) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def pad_hook(node, inputs, feats, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    global LAST_DIMS
    LAST_DIMS = outputs
    ops = 0
    return ops, mem_cost


def shape_hook(node, inputs, feats, outputs):
    mem_cost = len(inputs) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def gemm_hook(node, inputs, feats, outputs):
    global LAST_DIMS
    batch_size = LAST_DIMS[0]
    weight = 0
    for k in feats:
        if "weight" in k:
            weight = feats[k]
    if weight != 0 and len(weight) == 2:
        cout = weight[0]
        cin = weight[1]
        # FFN
        ops = cout * cin * batch_size
        mem_cost = batch_size * cout * MEMORY_MULTIPLIER
        return ops, mem_cost
        LAST_DIMS = (batch_size, cout)
    else:
        raise NotImplementedError()


# def softmax_hook(node, inputs, feats, outputs):


hooks = {
    'Conv': conv_hook,
    'BatchNormalization': bn_hook,
    'Relu': relu_hook,
    'MaxPool': pool_hook,
    'Add': add_hook,
    'GlobalAveragePool': pool_hook,
    'AveragePool': pool_hook,
    'Constant': const_hook,
    'Shape': shape_hook,
    'Gather': gather_hook,
    'Unsqueeze': reshape_hook,
    'Concat': concat_hook,
    'Reshape': reshape_hook,
    'Gemm': gemm_hook,
    'Squeeze': reshape_hook,
    'Pad': pad_hook
}


def op_hook(node, id_to_dim, feat_dim, inputs=()):
    if len(node.input) == 0:
        inputs = ()
    elif node.input[0] != '0':
        inputs = [id_to_dim[inp] for inp in node.input if inp in id_to_dim.keys()]
    feats = {name: feat_dim[name] for name in node.input if name in feat_dim.keys()}
    # elif inputs == None:
    #    raise NotImplementedError()
    if node.output[0] in id_to_dim:
        outputs = id_to_dim[node.output[0]]
    else:
        outputs = ()
    if len(inputs) == 0 or len(outputs) == 0:
        print(node.op_type)
    return hooks[node.op_type](node, inputs, feats, outputs)
