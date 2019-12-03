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


def conv_transpose_hook(node, inputs, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    weight = node.weights[0].shape
    cin = weight[3]
    #cout = weight[2]
    ops_per_output = cin
    ops = ops_per_output * np.prod(outputs)
    return ops, mem_cost


def conv_hook(node, inputs, outputs):
    # NOTE: This method assumes shapes are ordered as NHWC
    if None in outputs and None not in inputs and node.padding == "valid":
        # Fill in unknown height and width.
        # Note that padding = 0 for a "valid" Conv2D.
        H = int((inputs[1] - node.dilation_rate[0] * (node.kernel_size[0] - 1) - 1) / node.strides[0] + 1)
        W = int((inputs[2] - node.dilation_rate[1] * (node.kernel_size[1] - 1) - 1) / node.strides[1] + 1)
        newshape = (outputs[0], H, W, outputs[3])
        print("Inferred Conv2D shape: {} => {}".format(outputs, newshape))
        outputs = newshape

    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    weight = node.weights[0].shape
    # NHWC
    #cout = weight[3]
    cin = weight[2]
    kernel = weight[:2]
    #batch = inputs[0]
    ops_per_output = np.prod(kernel) * cin
    ops = ops_per_output * np.prod(outputs)
    return ops, mem_cost


def depthwise_conv_hook(node, inputs, outputs):
    weight = node.weights[0].shape
    #cout = weight[3]
    #cin = weight[2]
    kernel = weight[:2]
    #batch = inputs[0]
    ops_per_output = np.prod(kernel)  # don't look at rest of input
    ops = ops_per_output * np.prod(outputs)

    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    return ops, mem_cost


def bn_hook(node, inputs, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = 4 * np.prod(inputs)
    return ops, mem_cost


def relu_hook(node, inputs, outputs):
    ops = np.prod(inputs)
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    return ops, mem_cost


def pool_hook(node, inputs, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER

    # ops_per_output = np.prod(kernel)
    ops_per_output = 0  # TODO fix
    ops = ops_per_output * np.prod(outputs)
    return ops, mem_cost


def add_hook(node, inputs, outputs):
    assert len(inputs) > 1, "add needs more than one input"
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = sum([np.prod(inp) for inp in inputs])
    return ops, mem_cost


def pad_hook(node, inputs, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def nlp_matmul_hook(node, inputs, outputs):
    return fc_hook(node, inputs[0], outputs)


def fc_hook(node, inputs, outputs):
    # batch_size = inputs[0]
    cin = np.prod(inputs)
    cout = outputs[-1]

    ops = cin * cout
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    return ops, mem_cost


def concat_hook(node, inputs, outputs):
    assert len(inputs) > 1, "cooncat needs more than one input"
    ops = 0
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    return ops, mem_cost


def reshape_hook(node, inputs, outputs):
    if outputs.count(None) == 1:
        # Compute missing dimension from inputs
        input_count = np.prod(inputs)
        output_count = np.prod([d for d in outputs if d is not None])
        missing_dim = input_count // output_count
        outputs = tuple(d if d is not None else missing_dim for d in outputs)
        assert np.prod(outputs) == input_count, "Could not infer missing dimension in reshape output"

    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def upsample_hook(node, inputs, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    if node.interpolation == "nearest":
        # Assuming 1 op per interpolated point.
        ops = np.prod(outputs)
    elif node.interpolation == "bilinear":
        # 5x cost heuristic from
        # http://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Craver_Resampling.pdf
        ops = np.prod(outputs) * 5
    else:
        raise NotImplementedError("Unsupported interpolation method")
    return ops, mem_cost


def dropout_hook(node, inputs, outputs):
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = 0
    return ops, mem_cost


def pspnet_interp_hook(node, inputs, outputs):
    # Custom layer Interp used in keras_segmentation.models.pspnet_50
    # for bilinear interpolation
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    ops = np.prod(outputs) * 5
    return ops, mem_cost


def pspnet_lambda_hook(node, inputs, outputs):
    # Assume channels are last. Check that this layer removes
    # one row and one column from the input, which is how
    # Lambda is used in PSPNet.
    print("Using lambda from pspnet flops extraction.  If you are running bert, this is an ERROR")
    assert inputs[0] == outputs[0] and inputs[3] == outputs[3]
    assert inputs[1] == outputs[1] + 1
    assert inputs[2] == outputs[2] + 1
    mem_cost = np.prod(outputs) * MEMORY_MULTIPLIER
    return 0, mem_cost


# todo concatenate
# todo flatten
hooks = {
    # General hooks
    'Conv2D': conv_hook,
    'Conv2DTranspose': conv_transpose_hook,
    'Cropping2D': pool_hook,  # TODO fix
    'DepthwiseConv2D': depthwise_conv_hook,
    'BatchNormalization': bn_hook,
    'Activation': relu_hook,
    'ReLU': relu_hook,
    'MaxPooling2D': pool_hook,
    'Dropout': dropout_hook,
    'Concatenate': concat_hook,
    'TensorFlowOpLayer': reshape_hook,
    'LayerNormalization': bn_hook,
    'Add': add_hook,
    'GlobalAveragePooling2D': pool_hook,
    'AveragePooling2D': pool_hook,
    # 'Shape': shape_hook,
    "Flatten": reshape_hook,
    "Concat": concat_hook,
    "Reshape": reshape_hook,
    "UpSampling2D": upsample_hook,
    "Dense": fc_hook,
    # 'Gemm': gemm_hook,
    # 'Squeeze' : reshape_hook,
    "ZeroPadding2D": pad_hook,
    # Model specific hooks
    'Interp': pspnet_interp_hook,
    # Lambda can vary depending on the implementation
    # 'Lambda': pspnet_lambda_hook,
    'Lambda': fc_hook,
    "Interp": pspnet_interp_hook,
}


def add_batch(tup, b):
    if tup[0] is None:
        lst = list(tup)
        lst[0] = b
        return tuple(lst)
    return tup


def op_hook(layer, batch_size=1):
    input_shapes = layer.input_shape
    output_shapes = layer.output_shape

    if type(input_shapes) == tuple:
        inputs = add_batch(input_shapes, batch_size)
    elif type(input_shapes) == list:
        inputs = []
        for input_shape in input_shapes:
            inputs.append(add_batch(input_shape, batch_size))
    else:
        raise ValueError("layer.input_shapes must be tuple or list")

    if type(output_shapes) == tuple:
        outputs = add_batch(output_shapes, batch_size)
    elif type(output_shapes) == list:
        outputs = []
        for output_shape in output_shapes:
            outputs.append(add_batch(output_shape, batch_size))
    else:
        raise ValueError("layer.output_shape must be tuple or list")

    # Shape checks
    if len(inputs) == 0 or len(outputs) == 0:
        print("WARN: No inputs or no outputs?", type(layer), "input shape:", inputs, "output shape:", outputs)

    if None in inputs or None in outputs:
        print(
            "WARN: Layer of type {} has None in shape".format(type(layer)),
            "input shape:",
            inputs,
            "output shape:",
            outputs,
        )

    ops, mem_cost = hooks[layer.__class__.__name__](layer, inputs, outputs)

    return ops, mem_cost
