#!/bin/bash
conda config --add channels http://conda.anaconda.org/gurobi
conda install --yes gurobi graphviz
git clone https://github.com/ajayjain/image-segmentation-keras.git keras-segmentation
(cd keras-segmentation; python setup.py install)
yes | pip install -r requirements.txt
sudo apt-get update
sudo apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
