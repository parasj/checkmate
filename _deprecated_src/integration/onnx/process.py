import numpy as np
import onnx
import torch
from onnx import shape_inference

from integration.tf2.hooks import op_hook

double = ["Gemm", "Conv"]
need_input = ["Gemm", "Conv", "Relu", "BatchNormalization",
              "AveragePool", "GlobalAveragePool", "MaxPool"]


def get_dim(act):
    dims = act.type.tensor_type.shape.dim
    shape = []
    for dim in dims:
        shape.append(dim.dim_value)
    return tuple(shape)


def process(model, inputs):
    # reload model
    torch.onnx.export(model, inputs, 'tmp.onnx', verbose=True)
    model = onnx.load('tmp.onnx')
    onnx.checker.check_model(model)

    # create maps
    features = {inp.name: inp for inp in model.graph.input}
    nodes = model.graph.node

    # helpful dicts
    idx_to_id = [n.output[0] for n in nodes]  # activations
    id_to_idx = {id: idx for idx, id in enumerate(idx_to_id)}

    infer = shape_inference.infer_shapes(model)
    id_to_dim = {act.name: get_dim(act) for act in infer.graph.value_info}
    feat_dim = {name: get_dim(feat) for name, feat in features.items()}

    size = 2 * len(idx_to_id)
    # dependence_graph = np.zeros((size, size))
    dependence_graph = []
    ops = np.zeros(size)
    mem = np.zeros(size)

    for i, node in enumerate(nodes):
        out = int(id_to_idx[node.output[0]])
        for dep in node.input:
            if dep in id_to_idx.keys():
                depi = int(id_to_idx[dep])
                dependence_graph.append((depi, out))
                dependence_graph.append((size - out - 1, size - depi - 1))
                if node.op_type in need_input:
                    dependence_graph.append((out, size - depi - 1))
        n_op, n_mem = op_hook(node, id_to_dim, feat_dim, inputs=inputs)
        if node.op_type in double:
            bw_op = 2 * n_op
        else:
            bw_op = n_op
        ops[out] = n_op
        ops[size - out - 1] = bw_op
        mem[out] = n_mem
        mem[size - out - 1] = n_mem

    return len(nodes), dependence_graph, ops, mem
