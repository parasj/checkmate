---
description: 5 minute introduction to Checkmate
---

# Getting started

**5 minute tutorial on Checkmate**: [Colab notebook](https://colab.research.google.com/github/parasj/checkmate/blob/master/tutorials/tutorial_basic_tf2_example.ipynb)

Checkmate is a system for training large-scale deep neural networks when they don't fit within GPU memory. It utilizes the following methods to optimize your graphs:

* **Recomputation:** Checkmate will discard activations for layers that are memory-hungry but computationally cheap, and will recompute them during the backwards pass. This reduces peak RAM consumption at the cost of a small amount of duplicated computation.
* **Memory paging:** Checkmate evicts stale tensors to host memory in order to conserve GPU RAM for the most recent \(and thus "hotter"\) tensors.
* **Automatic reversibility:** Certain layers like 1x1 convolutions as well as the Leaky ReLU are bijective. Therefore, we can invert their cached outputs during the backwards pass to reconstruct their inputs.

_At the moment, Checkmate supports TensorFlow 2.0, but PyTorch integration is coming soon._

