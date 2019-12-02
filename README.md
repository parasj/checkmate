# Checkmate: Training huge DNNs on a single GPU
[![Actions Status](https://github.com/parasj/checkmate/workflows/Python%20package%20testsuite%20(checkmate)/badge.svg)](https://github.com/parasj/checkmate/actions)

`checkmate` is a package to compute schedules for rematerializing tensors in DFGraphs (tensor dataflow graphs).

# Installation
```bash
$  git clone https://github.com/parasj/checkmate.git
$  cd checkmate
$  pip install -e .[eval,test]
$  py.test
```
If you are evaluating on a GPU instance, run `pip install -e .[eval,gpu,test]` to install `tensorflow-gpu`.

ZSH complains with the extras syntax on local directories so you may need to escape square brackets e.g. `pip install -e .\[eval,test\]`.

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
