# List of Device characteristics

MKR1000 = {
    # Theoretical FLOPS_PER_WATT = 48 X 10^(6) / (100 X 3 X 20 X 10^(-3) X 3.3)  = 2.42 M FLOP / J
    "FLOPS_PER_WATT": 2000 / (0.012 * 0.020 * 3.3),  # also flop per joule
    "PAGEIN_LATENCY": 0.109 * 10 ** (-3),  # in seconds
    "PAGEIN_THROUGHPUT": 4616.51 * 10 ** (3),  # in bytes per second
    "PAGEOUT_LATENCY": 0.113 * 10 ** (-3),  # in seconds
    "PAGEOUT_THROUGHPUT": 4440 * 10 ** (3),  # in byte per second
    "MEMORY_POWER": 100 * 10 ** (-3) * 3.3,  # in ampere*volt
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

# Network
LIN_NET = [[784, 120], [120, 120], [120, 100], [100, 60], [60, 10], [10, 10]]  # linear network 124 X 100 X 60 X 10 X 10

"""
Energy consumed for computing the activations of the given layer

 @param device Choose a device - associated parameters
 @param network network whose layers are to be evaluated
 @returns energy in joules
"""


def compute_power(device, network):
    power_list = []
    for layer in network:
        _no_parameters = (layer[0]) * layer[1]  # +1 for the bias term
        _power = _no_parameters / device["FLOPS_PER_WATT"]
        power_list.append(_power)
    # 2 X for backward pass
    for item in power_list[::-1]:
        power_list.append(2 * item)
    # Loss function
    _loss_ops = network[-1][-1] * 10  # Assuming 10 ops for each cross-entropy
    power_list.insert(int(len(power_list) / 2), _loss_ops / device["FLOPS_PER_WATT"])
    return power_list


"""
Energy consumed for paging in activations of the given layer
 from SD card. Assuming we read 512 byte blocks.

 @param device Choose a device - associated parameters
 @param network network whose layers are to be evaluated
 @returns energy in joules
"""


def pagein_power(device, network):
    power_list = []
    for layer in network:
        _bytes = layer[1] * device["TYPE"]
        _time = device["PAGEIN_LATENCY"] + (_bytes / device["PAGEIN_THROUGHPUT"])
        _power = _time * device["MEMORY_POWER"]
        power_list.append(_power)
    # Loss is a scalar
    _bytes = device["TYPE"]
    _time = device["PAGEIN_LATENCY"] + (_bytes / device["PAGEIN_THROUGHPUT"])
    _power = _time * device["MEMORY_POWER"]
    power_list.append(_power)
    # Backward pass
    for layer in network[::-1]:
        _bytes = layer[1] * device["TYPE"]
        _time = device["PAGEIN_LATENCY"] + (_bytes / device["PAGEIN_THROUGHPUT"])
        _power = _time * device["MEMORY_POWER"]
        power_list.append(_power)
    return power_list


"""
Energy consumed for paging out activations of the given layer
 from SD card. Assuming we read 512 byte blocks.

 @param device Choose a device - associated parameters
 @param network network whose layers are to be evaluated
 @returns energy in joules
"""


def pageout_power(device, network):
    power_list = []
    for layer in network:
        _bytes = layer[1] * device["TYPE"]
        _time = device["PAGEOUT_LATENCY"] + (_bytes / device["PAGEOUT_THROUGHPUT"])
        _power = _time * device["MEMORY_POWER"]
        power_list.append(_power)
    # Loss is a scalar
    _bytes = device["TYPE"]
    _time = device["PAGEOUT_LATENCY"] + (_bytes / device["PAGEOUT_THROUGHPUT"])
    _power = _time * device["MEMORY_POWER"]
    power_list.append(_power)
    # Backward pass
    for layer in network[::-1]:
        _bytes = layer[1] * device["TYPE"]
        _time = device["PAGEOUT_LATENCY"] + (_bytes / device["PAGEOUT_THROUGHPUT"])
        _power = _time * device["MEMORY_POWER"]
        power_list.append(_power)
    return power_list


"""
Memory in bytes consumed for activations of each layer

 @param device Choose a device - associated parameters
 @param network network whose layers are to be evaluated
 @returns memory for each layers activation in bytes
 	The len(array)/2 memory is that of loss 
"""


def activation_memory(device, network):
    memory_list = []
    for layer in network:
        _bytes = layer[1] * device["TYPE"]
        memory_list.append(_bytes)
    # Loss is a scalar
    _bytes = device["TYPE"]
    memory_list.append(_bytes)
    # Backward pass
    for layer in network[::-1]:
        _bytes = layer[1] * device["TYPE"]
        memory_list.append(_bytes)
    return memory_list


def main():
    compute_list = compute_power(MKR1000, LIN_NET)
    pagein_list = pagein_power(MKR1000, LIN_NET)
    pageout_list = pageout_power(MKR1000, LIN_NET)
    memory_list = activation_memory(MKR1000, LIN_NET)


if __name__ == "__main__":
    main()
