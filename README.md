![](https://checkmateai.github.io/img/dark_logo.png)

_See the paper!_ [https://arxiv.org/abs/1910.02653](https://arxiv.org/abs/1910.02653)

`checkmate` breaks the GPU memory wall by enabling researchers to train large state-of-the-art models that do not fit in GPU memory. Checkmate applies optimal tensor rematerialization \(as detailed in our paper at MLSys 2020\) to trade off space and time.

At the moment, Checkmate only supports TensorFlow 2.0. PyTorch support is coming soon!<!-- To follow updates on PyTorch support, please suscribe to our [Google Group](https://groups.google.com/forum/#!forum/checkmate-dev). -->

## Installation

Checkmate depends on:
* [TensorFlow 2.0](https://www.tensorflow.org/install), i.e. `pip install tensorflow` or `pip install tensorflow-gpu`.
* [CyLP solver](https://github.com/coin-or/CyLP)
    <details><summary>Installing CyLP on Debian Linux / Ubuntu</summary>
    <p>

    ```bash
    $ sudo apt install coinor-cbc coinor-libcbc-dev
    $ pip install cylp
    ```
    </p>
    </details>
    <details><summary>Installing CyLP on MacOS</summary>
    <p>
    
    The easiest way to set up CyLP is using [homebrew](https://brew.sh/).
    ```bash
    $ brew tap coin-or-tools/coinor
    $ brew install coin-or-tools/coinor/cbc pkg-config
    $ pip install cylp
    ```
    </p>
    </details>


Once TensorFlow 2.0 and CyLP are installed, Checkmate can be installed using pip via `pip install "https://github.com/parasj/checkmate/archive/master.zip#egg=checkmate"`.

## Quick start

**Get started in 5m with our** [**TF2.0 quickstart tutorial**](https://colab.research.google.com/github/parasj/checkmate/blob/master/tutorials/tutorial_basic_tf2_example.ipynb)

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
```text
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

```text
@incollection{mlsys2020_196,
 author = {Jain, Paras and Jain, Ajay and Nrusimha, Aniruddha and Gholami, Amir and Abbeel, Pieter and Gonzalez, Joseph and Keutzer, Kurt and Stoica, Ion},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {497--511},
 title = {Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization},
 year = {2020}
}


```
