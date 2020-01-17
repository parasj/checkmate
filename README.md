![Checkmate logo](https://checkmateai.github.io/img/dark_logo.png)

[![Actions Status](https://github.com/parasj/checkmate/workflows/Python%20package%20testsuite%20(checkmate)/badge.svg)](https://github.com/parasj/checkmate/actions)

*See the paper!* [https://arxiv.org/abs/1910.02653](https://arxiv.org/abs/1910.02653)

`checkmate` breaks the GPU memory wall by enabling researchers to train large state-of-the-art models that do not fit in GPU memory. Checkmate applies optimal tensor rematerialization (as detailed in our paper at MLSys 2020) to trade off space and time.

At the moment, Checkmate only supports TensorFlow 2.0. PyTorch support is coming soon! To follow updates on PyTorch support, please suscribe to our [Google Group](https://groups.google.com/forum/#!forum/checkmate-dev).

## Installation
Get started with `pip install "https://github.com/parasj/checkmate/archive/master.zip#egg=checkmate"`

Ensure you have installed either `tensorflow-gpu>=2.0.0` or `tensorflow`.

## Quick start
**Get started in 5m with our [TF2.0 quickstart tutorial](https://colab.research.google.com/github/parasj/checkmate/blob/master/tutorials/tutorial_basic_tf2_example.ipynb)**

Adapt your Keras model to fit within the memory constraints of a single GPU:
```python
import checkmate
model = tf.keras.applications.vgg19.VGG19(...)
...

train_iteration_fn = checkmate.tf2.compile(model, loss, optimizer,
    input_spec=sample_input[0], label_spec=sample_input[1])

for image, label in train_ds:
    prediction, loss = train_iteration_fn(image, label)
```

## Key ideas
From our [paper at MLSys 2020](https://arxiv.org/abs/1910.02653):
```
Modern neural networks are increasingly bottlenecked by the limited capacity of on-device
GPU memory. Prior work explores dropping activations as a strategy to scale to larger
neural networks under memory constraints. However, these heuristics assume uniform
per-layer costs and are limited to simple architectures with linear graphs, limiting their
usability. In this paper, we formalize the problem of trading-off DNN training time and
memory requirements as the tensor rematerialization optimization problem, a generalization
of prior checkpointing strategies. We introduce Checkmate, a system that solves for
optimal schedules in reasonable times (under an hour) using off-the-shelf MILP solvers,
then uses these schedules to accelerate millions of training iterations. Our method scales
to complex, realistic architectures and is hardware-aware through the use of
accelerator-specific, profile-based cost models. In addition to reducing training cost,
Checkmate enables real-world networks to be trained with up to 5.1× larger input sizes.
```


## Citation
If you use Checkmate in your work, please cite us with:
```
@article{jain2019checkmate,
  title={Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization},
  author={Jain, Paras and Jain, Ajay and Nrusimha, Aniruddha and Gholami, Amir and
          Abbeel, Pieter and Keutzer, Kurt and Stoica, Ion and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:1910.02653},
  year={2020}
}
```
