# Checkmate: Training huge DNNs on a single GPU
[![Actions Status](https://github.com/parasj/checkmate/workflows/Python%20package%20testsuite%20(checkmate)/badge.svg)](https://github.com/parasj/checkmate/actions)

`checkmate` is a package to compute schedules for rematerializing tensors in DFGraphs (tensor dataflow graphs).

* For more info see the [arXiv paper](https://arxiv.org/abs/1910.02653).
*** Abstract *** Modern neural networks are increasingly bottlenecked by the limited capacity of on-device GPU memory. Prior work explores dropping activations as a strategy to scale to larger neural networks under memory constraints. However, these heuristics assume uniform per-layer costs and are limited to simple architectures with linear graphs, limiting their usability. In this paper, we formalize the problem of trading-off DNN training time and memory requirements as the tensor rematerialization optimization problem, a generalization of prior checkpointing strategies. We introduce Checkmate, a system that solves for optimal schedules in reasonable times (under an hour) using off-the-shelf MILP solvers, then uses these schedules to accelerate millions of training iterations. Our method scales to complex, realistic architectures and is hardware-aware through the use of accelerator-specific, profile-based cost models. In addition to reducing training cost, Checkmate enables real-world networks to be trained with up to 5.1× larger input sizes.


# Installation
```bash
pip install https://github.com/parasj/checkmate/archive/master.zip#egg=checkmate
```

* Also see tutorials subdirectory for a notebook example.



# Citation
If you use Checkmate in your work, please cite us with:
```
@article{jain2019checkmate,
  title={Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization},
  author={Jain, Paras and Jain, Ajay and Nrusimha, Aniruddha and Gholami, Amir and Abbeel, Pieter and Keutzer, Kurt and Stoica, Ion and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:1910.02653},
  year={2019}
}
```
