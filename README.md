# Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization

This document contains instructions for reproducing plots in the MLSys 2020 paper "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization".

## Background
`remat` is a package to compute memory-efficient schedules for evaluating neural network dataflow graphs created by the backpropagation algorithm. To save memory, the package deletes and rematerializes intermediate values via recomputation. The schedule with minimum recomputation for a given memory budget is chosen by solving an integer linear program. For details about our approach, please see the following paper,
```
@inproceedings{jain2019checkmate,
  title={Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization},
  author={Jain, Paras and Jain, Ajay and Nrusimha, Aniruddha and Gholami, Amir and Abbeel, Pieter and Keutzer, Kurt and Stoica, Ion and Gonzalez, Joseph E},
  booktitle = {Proceedings of the 3rd Conference on Machine Learning and Systems},
  series = {MLSys 2020},
  year={2019}
}
```

## Installation
### Step 1: Install Anaconda
Instructions are provided with the Anaconda Python environment manager. First, install Anaconda for Python 3.7 using https://www.anaconda.com/distribution/.
Then, create a new conda environment with Python 3.7.5:
```
$ conda create -n checkmate-mlsys-artifact python=3.7
$ conda activate checkmate-mlsys-artifact
```

### Step 2: Install the `remat` package and dependencies
From this directory,
```
$ conda install -c conda-forge python-graphviz
$ pip install -e .
```

### Step 3: Install Gurobi
Checkmate uses the Gurobi optimziation library to solve an integer linear program that chooses a recomputation schedule for a given neural network architecture. This requires a license to Gurobi, which is free for academic use. The `grbgetkey` command used below must be run on a computer connected to a university network directly or via a VPN.

1. Please follow these instructions to install Gurobi on your system: https://www.gurobi.com/documentation/quickstart.html
2. Make an academic account with Gurobi at: https://pages.gurobi.com/registration
3. Request an acadmic license at: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
4. Install the license by running the `grbgetkey` command at the end of the page. Ensure you are on an academic network like Airbears2.
5. Set up the gurobipy Anaconda channel by running `conda config --add channels http://conda.anaconda.org/gurobi`
6. Install gurobipy by running: `conda install gurobi`


## Reproducing Figure 4: Computational overhead versus memory budget
This experiment evaluates rematerialization strategies at a range of memory budgets. In the MLSys 2020 submission, Figure 4 includes results for the VGG16, MobileNet, and U-Net computer vision neural network architectures.

### Figure 4, VGG16
Results for VGG16 can be reproduced quickly, as the network is simple and small. Run the following command (expected runtime TODO):
```
python experiments/experiment_budget_sweep.py --model-name "VGG16" -b 256 --platform p32xlarge
```
The error `ERROR:root:Infeasible model, check constraints carefully. Insufficient memory?` is expected. For some of the attempted memory budgets, Checkmate or a baseline with not be able to find a feasible (in-memory) schedule. The results plot will be written to `data/budget_sweep/p32xlarge_VGG16_256_None/plot_budget_sweep_VGG16_p32xlarge_b256.pdf`. The experiment uses a profile-based cost model based on the AWS p32xlarge server, which includes a NVIDIA V100 GPU.
### Figure 4, MobileNet
Run the following command:
```
python experiments/experiment_budget_sweep.py --model-name "MobileNet" -b 512 --platform p32xlarge
```
### Figure 4, U-Net
Run the following command:
```
python experiments/experiment_budget_sweep.py --model-name "vgg_unet" -b 32 --platform p32xlarge
```


## Reproducing Figure 5: Maximum model batch size
We provide instructions for the VGG19 architecture. Batch size can be maximized for other architectures by changing the `--model-name` argument to, e.g. `vgg_unet`, `MobileNet`, `segnet`, and `fcn_8`.

### Maximum batch size using baseline strategies
We find the maximum batch size that baselines can support by reevaluating our implementations of their strategies at each budget in a range. To run the baselines,
```
python experiments/experiment_max_batchsize_baseline.py --model-name VGG19 --platform flops
```
This produces the following results,
```
                          strategy  batch_size
0    SolveStrategy.CHEN_SQRTN_NOAP         196
1         SolveStrategy.CHEN_SQRTN         196
2   SolveStrategy.CHEN_GREEDY_NOAP         260
3  SolveStrategy.CHECKPOINT_ALL_AP         164
4     SolveStrategy.CHECKPOINT_ALL         164
5        SolveStrategy.CHEN_GREEDY         260
```
As VGG19 is a linear-chain architecture, the articulation point and linearized (NOAP) generalizations of baselines are the same. For the default resolution for the VGG19 implementation used in this repository, we find that checkpointing all nodes supports batch sizes up to 164 on a V100 (result 4), Chen's sqrt(n) strategy can achieve a batch size of 196, and Chen's greedy strategy can achieve a batch size of 260. The batch size reported for the checkpoint all baseline is lower than that in the paper as we computed the paper's number optimistically via a calculation that assumes activation memory scales linearly with batch size, whereas this code actually finds the schedule that retains all activations. Since submitting the paper, we also increased the number of points evaluated in the hyperparameter search for Chen's greedy baseline, improving its results slightly at the expense of longer solve time. Results will be updated.

### Maximum batch size using Checkmate
The optimization problem used to generate Figure 5 is computationally intensive to solve, but reasonable for VGG19. To run the experiment (5-10 minutes on 12 cores),
```
python experiments/experiment_max_batchsize_ilp.py --model-name VGG19 --platform flops --batch-size-min 160
```
For other networks that take longer to solve, you can monitor the highest feasible batch size found during the solving process. During solving, Checkmate prints the best incumbent solution and lowest upper bound for the batch size. For example, in the following log message, `289.0...` denotes the highest batch size for which a schedule has been found so far (Incumbent), and `371.5...` denotes the lowest certifiable upper bound on the maximum batch size so far (BestBd). Solving will terminate when the incumbent and best bound have a sufficiently small gap.
```
INFO:gurobipy: Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time
...
INFO:gurobipy:   117   133  350.00957    9  729  289.03459  371.53971  28.5%   576   77s
```
Incumbents may be fractional as the optimization problem maximizes a real multiplier for the memory consumption. The final, max batch size found will be printed, e.g.:
```
INFO:root:Max batch size = 289
```
You can terminate the solving process early by pressing Ctrl-C if desired. After completion, the model, dataflow graph, and final schedule will be visualized as PNG, PDF and PNG files, respectively, in `data/max_batch_size_ilp/flops_VGG19_None/`.


## Troubleshooting
If Gurobi is unable to locate your license file, set its path via an environment variable:
```
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```
For example, the licence is stored by default at `$HOME/gurobi.lic`.


## All supported model architectures
The following architectures are implemented via the `--model-name` argument: DenseNet121,DenseNet169,DenseNet201,InceptionV3,MobileNet,MobileNetV2,NASNetLarge,NASNetMobile,ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2,VGG16,VGG19,Xception,fcn_32,fcn_32_mobilenet,fcn_32_resnet50,fcn_32_vgg,fcn_8,fcn_8_mobilenet,fcn_8_resnet50,fcn_8_vgg,linear0,linear1,linear10,linear11,linear12,linear13,linear14,linear15,linear16,linear17,linear18,linear19,linear2,linear20,linear21,linear22,linear23,linear24,linear25,linear26,linear27,linear28,linear29,linear3,linear30,linear31,linear4,linear5,linear6,linear7,linear8,linear9,mobilenet_segnet,mobilenet_unet,pspnet,pspnet_101,pspnet_50,resnet50_pspnet,resnet50_segnet,resnet50_unet,segnet,test,unet,unet_mini,vgg_pspnet,vgg_segnet,vgg_unet
