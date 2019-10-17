# Checkmate: Training huge DNNs on a single GPU
[![Actions Status](https://github.com/parasj/tensor-remat/workflows/Python%20package%20testsuite%20(remat)/badge.svg)](https://github.com/parasj/tensor-remat/actions)

`checkmate` is a package to compute schedules for rematerializing tensors in DFGraphs (tensor dataflow graphs).

# Installation
```bash
$  git clone https://github.com/parasj/tensor-remat.git
$  cd tensor-remat
$  pip install -e .
$  py.test
```

If you are evaluating on a GPU instance, you can install `tensorflow-gpu` as a dependency in order to enable GPU support.

<!--
# Project structure
```
.
├── experiments                stores one-off plotting scripts for paper
├── remat                      main python package
│   ├── core
│   │   ├── graph.py           DFGraph data structure
│   │   ├── schedule.py        Schedule defition of concrete evaluation order
│   │   ├── solvers            Package containing solver implementations
│   │   └── utils 
│   └── tensorflow2            Tensorflow 2.0 integration (extraction and execution)
└── tests
```
-->
