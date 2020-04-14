from typing import List

import numpy as np

# List of Device characteristics
MKR1000 = {
    # Theoretical FLOPS_PER_WATT = 48 X 10^(6) / (100 X 3 X 20 X 10^(-3) X 3.3)  = 2.42 M FLOP / J
    "FLOPS_PER_WATT": 2000 / (0.012 * 0.020 * 3.3),  # also flop per joule
    "PAGEIN_LATENCY": 0.109 * 10 ** (-3),  # in seconds
    "PAGEIN_THROUGHPUT": 4616.51 * 10 ** (3),  # in bytes per second
    "PAGEOUT_LATENCY": 0.113 * 10 ** (-3),  # in seconds
    "PAGEOUT_THROUGHPUT": 4440 * 10 ** (3),  # in byte per second
    "MEMORY_POWER": 100 * 10 ** (-3) * 3.3 * 10,  # in ampere*volt
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
}

RPi = {
    "FLOPS_PER_WATT": 0,  # in flop per second
    "PAGEIN_LATENCY": 0,  # in seconds
    "PAGEIN_THROUGHPUT": 0,  # in bytes per second
    "PAGEOUT_LATENCY": 0,  # in seconds
    "PAGEOUT_THROUGHPUT": 0,  # in byte per second
    "MEMORY_POWER": 0,  # in ampere*volt
    "TYPE": 4,  # in bytes; Float - 4, Int - 4, long - 8
}


class DNNLayer:
    def __init__(self, out_shape, depends_on: List["DNNLayer"] = tuple(), param_count=0):
        assert out_shape is not None  # get around varargs restriction
        self.out_shape = out_shape
        self.depends_on = depends_on
        self.param_count = param_count

    def power(self, device) -> float:
        """
        Energy consumed for computing the activations of the given layer
        @param device Choose a device - associated parameters
        @returns energy in joules
        """

    def power_ram2sd(self, device) -> float:
        """
        Energy consumed for paging out activations of the given layer
        from SD card. Assuming we read 512 byte blocks.
         @param device Choose a device - associated parameters
         @returns energy in joules
        """
        _time = device["PAGEOUT_LATENCY"] + (self.output_ram_usage(device) / device["PAGEOUT_THROUGHPUT"])
        return _time * device["MEMORY_POWER"]

    def power_sd2ram(self, device) -> float:
        """
        Energy consumed for paging out activations of the given layer
        from SD card. Assuming we read 512 byte blocks.
        @param device Choose a device - associated parameters
        @returns energy in joules
        """
        _time = device["PAGEIN_LATENCY"] + (self.output_ram_usage(device) / device["PAGEIN_THROUGHPUT"])
        return _time * device["MEMORY_POWER"]

    def output_ram_usage(self, device) -> int:
        """
        RAM consumption in bytes
        @returns memory required to store output of this layer
        """
        return np.prod(self.out_shape) * device["TYPE"]

    def param_ram_usage(self, device) -> int:
        """
        RAM necessary for parameters + workspace memory
        @returns memory required to store
        """
        return self.param_count * device["TYPE"]


class LinearLayer(DNNLayer):
    def __init__(self, in_features, out_features, input: DNNLayer):
        super().__init__((out_features,), [input] if input is not None else [], param_count=(in_features * out_features))
        self.in_features = in_features
        self.out_features = out_features

    def power(self, device):
        return (self.param_count + self.out_features) / device["FLOPS_PER_WATT"]


class ReLULayer(DNNLayer):
    def __init__(self, input: DNNLayer):
        super().__init__(out_shape=input.out_shape, depends_on=[input])

    def power(self, device) -> float:
        return np.prod(self.out_shape) / device["FLOPS_PER_WATT"]


class LossLayer(DNNLayer):
    def __init__(self, input: DNNLayer, n_classes=10):
        super().__init__(out_shape=(n_classes,), depends_on=[input])
        self.n_classes = n_classes

    def power(self, device) -> float:
        return (np.prod(self.depends_on[0].out_shape) * self.n_classes) / device["FLOPS_PER_WATT"]


class GradientLayer(DNNLayer):
    def __init__(self, output: DNNLayer, input: DNNLayer, grad_outputs: "GradientLayer"):
        super().__init__(out_shape=input.out_shape, depends_on=[output, input, grad_outputs])
        self.output = output

    def power(self, device):
        return 2 * self.output.power(device)


def make_linear_network():
    linear_layers = []
    for i in range(3):
        linear_layers.append([256, 64])
        linear_layers.append([64, 64])
        linear_layers.append([64, 256])
    linear_layers.append([256, 10])
    # linear_layers = [[784, 10], [10, 120], [120, 100], [100, 200], [200, 10], [10, 10]]
    layers = []
    for in_dim, out_dim in linear_layers:
        last_layer = layers[-1] if len(layers) > 0 else None
        lin_layer = LinearLayer(in_dim, out_dim, input=last_layer)
        act_layer = ReLULayer(lin_layer)
        layers.extend([lin_layer, act_layer])
    layers.append(LossLayer(layers[-1], n_classes=10))
    reverse_layers = list(reversed(layers))
    for input, output in zip(reverse_layers[:-1], reverse_layers[1:]):
        layers.append(GradientLayer(output, input, layers[-1]))
    return layers


def get_net_costs(device=None, net=None):
    if device is None:
        device = MKR1000
    if net is None:
        net = make_linear_network()

    compute_list, ram_list, param_ram_list, pagein_cost, pageout_cost = [[] for _ in range(5)]
    for layer in net:
        compute_list.append(layer.power(device))
        ram_list.append(layer.output_ram_usage(device))
        param_ram_list.append(layer.param_ram_usage(device))
        pagein_cost.append(layer.power_sd2ram(device))
        pageout_cost.append(layer.power_ram2sd(device))

    return dict(compute=compute_list, memory=ram_list, param_memory=param_ram_list,
                pagein_cost=pagein_cost, pageout_cost=pageout_cost)
