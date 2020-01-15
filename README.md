![Checkmate logo](https://checkmateai.github.io/img/dark_logo.png)

# Training huge DNNs on a single GPU
[![Actions Status](https://github.com/parasj/checkmate/workflows/Python%20package%20testsuite%20(checkmate)/badge.svg)](https://github.com/parasj/checkmate/actions)

**See the paper**: https://arxiv.org/abs/1910.02653

`checkmate` breaks the GPU memory wall by enabling researchers to train large state-of-the-art models that do not fit in GPU memory. Checkmate applies optimal tensor rematerialization (as detailed in our paper at MLSys 2020) to trade off space and time.

At the moment, Checkmate only supports TensorFlow 2.0. PyTorch support is coming soon! Please  


# Quick start
Adapt Keras model

# Installation
```bash
pip install https://github.com/parasj/checkmate/archive/master.zip#egg=checkmate
```

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
