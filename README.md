# Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization

This document contains instructions for reproducing plots in the MLSys 2020 paper "Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization".

## Background
`remat` is a package to compute memory-efficient schedules for evaluating neural network dataflow graphs created by the backpropagation algorithm. To save memory, the package deletes and rematerializes intermediate values via recomputation. The schedule with minimum recomputation for a given memory budget is chosen by solving an integer linear program. For details about our approach, please see the following paper,
```
@inproceedings{jain2020checkmate,
  title={Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization},
  author={Jain, Paras and Jain, Ajay and Nrusimha, Aniruddha and Gholami, Amir and Abbeel, Pieter and Keutzer, Kurt and Stoica, Ion and Gonzalez, Joseph E},
  booktitle = {Proceedings of the 3rd Conference on Machine Learning and Systems},
  series = {MLSys 2020},
  year={2020}
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
Clone this repository and check out the `mlsys20_artifact` branch:
```
$ git clone git@github.com:parasj/checkmate.git
$ cd checkmate
$ git checkout mlsys20_artifact
```
From this directory,
```
$ conda install -c conda-forge python-graphviz
$ pip install -e .
```
If you are running setup on a machine without a GPU, run,
```
$ pip install tensorflow-cpu
```
CPU-only is enough to use this artifact. If you have a GPU-enabled machine with CUDA installed, you can instead run
```
$ pip install tensorflow>=2.0.0
```

### Step 3: Install Gurobi
Checkmate uses the Gurobi optimziation library to solve an integer linear program that chooses a recomputation schedule for a given neural network architecture. This requires a license to Gurobi, which is free for academic use. The `grbgetkey` command used below must be run on a computer connected to a university network directly or via a VPN.

1. Please follow these instructions to install Gurobi on your system: https://www.gurobi.com/documentation/quickstart.html. For example, on Linux, follow `https://www.gurobi.com/documentation/9.0/quickstart_linux/software_installation_guid.html`.
2. Make an academic account with Gurobi at: https://pages.gurobi.com/registration
3. Request an acadmic license at: https://www.gurobi.com/downloads/end-user-license-agreement-academic/
4. Install the license by running the `grbgetkey` command at the end of the page. Ensure you are on an academic network like Airbears2 for UC Berkeley. If you save the license to a non-default location (outside your home directory), you will need to export the `GRB_LICENSE_FILE` variable with the path to the licence.
5. Set up the gurobipy Anaconda channel by running `conda config --add channels http://conda.anaconda.org/gurobi`
6. Install gurobipy by running: `conda install gurobi`


## Reproducing Figure 4: Computational overhead versus memory budget
This experiment evaluates rematerialization strategies at a range of memory budgets. In the MLSys 2020 submission, Figure 4 includes results for the VGG16, MobileNet, and U-Net computer vision neural network architectures.

### Figure 4, VGG16
Results for VGG16 can be reproduced relatively quickly, as the network is simple and small. Run the following command:
```
$ python experiments/experiment_budget_sweep.py --model-name "VGG16" -b 256 --platform p32xlarge
```
The error `ERROR:root:Infeasible model, check constraints carefully. Insufficient memory?` is expected. For some of the attempted memory budgets, Checkmate or a baseline with not be able to find a feasible (in-memory) schedule. The results plot will be written to `data/budget_sweep/p32xlarge_VGG16_256_None/plot_budget_sweep_VGG16_p32xlarge_b256.pdf`. The experiment uses a profile-based cost model based on the AWS p32xlarge server, which includes a NVIDIA V100 GPU.
### Figure 4, MobileNet
Run the following command:
```
$ python experiments/experiment_budget_sweep.py --model-name "MobileNet" -b 512 --platform p32xlarge
```
The results plot will be written to `data/budget_sweep/p32xlarge_MobileNet_512_None/plot_budget_sweep_MobileNet_p32xlarge_b512.pdf`.
### Figure 4, U-Net
Run the following command:
```
$ python experiments/experiment_budget_sweep.py --model-name "vgg_unet" -b 32 --platform p32xlarge
```
The results plot will be written to `data/budget_sweep/p32xlarge_vgg_unet_32_None/plot_budget_sweep_vgg_unet_p32xlarge_b32.pdf`.

The expected results plots form the subplots of Figure 4 in the paper.


## Reproducing Figure 6: Maximum model batch size

### Maximum batch size using baseline strategies
We find the maximum batch size that baselines can support by reevaluating our implementations of their strategies at each budget in a range. To run the baselines (7 minutes),
```
$ python experiments/experiment_max_batchsize_baseline.py --model-name VGG19 --batch-size-min 160 --batch-size-max 300 --batch-size-increment 1
```
For a coarser search, use `--batch-size-increment 8`. Arguments `--batch-size-min` and `--batch-size-max` control the search interval, allowing you to narrow the search range. The above command produces the following results,
```
                          strategy  batch_size
0    SolveStrategy.CHEN_SQRTN_NOAP         197
1   SolveStrategy.CHEN_GREEDY_NOAP         266
2     SolveStrategy.CHECKPOINT_ALL         167
3  SolveStrategy.CHECKPOINT_ALL_AP         167
4         SolveStrategy.CHEN_SQRTN         197
5        SolveStrategy.CHEN_GREEDY         266
```
Note that as VGG19 is a linear-chain architecture, the articulation point and linearized (NOAP) generalizations of baselines are the same. For the default resolution for the VGG19 implementation used in this repository, we find that checkpointing all nodes supports batch sizes up to 167 on a V100 (result 4), Chen's sqrt(n) strategy can achieve a batch size of 197, and Chen's greedy strategy can achieve a batch size of 266. The batch size reported for the checkpoint all baseline is lower than that in the paper as we computed the paper's number via a calculation that assumes activation memory scales linearly with batch size, whereas this code actually finds the schedule that retains all activations; this is more realistic. Since submitting the paper, we also modified the greedy and sqrtn baselines to more closely match the 2016 paper that proposed these heuristics, and increased number of hyperparameters searched for greedy, so results will slightly differ. Camera ready results will be updated. However, Checkmate results should be the same if sufficient time is allowed for optimization.

Baseline commands (again, feel free to use `--batch-size-increment 8` to get approximate results):
```
$ python experiments/experiment_max_batchsize_baseline.py --model-name vgg_unet --batch-size-min 10 --batch-size-max 40 --batch-size-increment 1
$ python experiments/experiment_max_batchsize_baseline.py --model-name fcn_8_vgg --batch-size-min 10 --batch-size-max 80 --batch-size-increment 1
$ python experiments/experiment_max_batchsize_baseline.py --model-name segnet --batch-size-min 20 --batch-size-max 50 --batch-size-increment 1
$ python experiments/experiment_max_batchsize_baseline.py --model-name ResNet50 --batch-size-min 90 --batch-size-max 200 --batch-size-increment 1
$ python experiments/experiment_max_batchsize_baseline.py --model-name VGG19 --batch-size-min 160 --batch-size-max 300 --batch-size-increment 1
$ python experiments/experiment_max_batchsize_baseline.py --model-name MobileNet --batch-size-min 200 --batch-size-max 650 --batch-size-increment 1
```

### Maximum batch size using Checkmate
The optimization problem used to generate Figure 6 is computationally intensive to solve, but reasonable for VGG19. To run the experiment (about 10 minutes),
```
$ python experiments/experiment_max_batchsize_ilp.py --model-name VGG19 --batch-size-min 160
```
The argument `--num-threads <number>` can be used to enable multicore optimization, which significantly accelerates scheduling. For other networks that take longer to solve, you can monitor the highest feasible batch size found during the solving process. During solving, Checkmate prints the best incumbent solution and lowest upper bound for the batch size. For example, in the following log message, `289.0...` denotes the highest batch size for which a schedule has been found so far (Incumbent), and `371.5...` denotes the lowest certifiable upper bound on the maximum batch size so far (BestBd). Solving will terminate when the incumbent and best bound have a sufficiently small gap.
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

ILP solve commands (provide `--num-threads <number>` to speed up solving):
```
$ python experiments/experiment_max_batchsize_ilp.py --model-name vgg_unet --batch-size-min 20  # Reaches 57 batch size after approx. 30 min (suboptimal)
$ python experiments/experiment_max_batchsize_ilp.py --model-name fcn_8_vgg --batch-size-min 20  # Can reach at least 60. Reaches 57 batch size after approx. 10 min (suboptimal)
$ python experiments/experiment_max_batchsize_ilp.py --model-name segnet --batch-size-min 20  # Can reach at least 62 batch size.
$ python experiments/experiment_max_batchsize_ilp.py --model-name ResNet50 --batch-size-min 100  # Can reach at least 193. Very computationally intensive.
                                                                                                 # In the paper, earlier, slower form of ILP was run for approx. 2 days.
$ python experiments/experiment_max_batchsize_ilp.py --model-name VGG19 --batch-size-min 160  # Reaches 289 batch size after approx. 10 min (certifiably optimal)
$ python experiments/experiment_max_batchsize_ilp.py --model-name MobileNet --batch-size-min 450  # Can reach at least 1105. Very computationally intensive.
                                                                                                  # In the paper, earlier, slower form of ILP was run for approx. 2 days.
```


## Troubleshooting
### Gurobi license errors
If Gurobi is unable to locate your license file, set its path via an environment variable:
```
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```
For example, the licence is stored by default at `$HOME/gurobi.lic`.


### Evaluation machine resources
* 2x Intel E5-2670 CPUs - (Haswell 12 Cores / 24 Threads)
* 256GB DDR4 RAM
* 4TB HDD
* Kernel: `Ubuntu 18.04.3 LTS (GNU/Linux 5.3.0-24-generic x86_64)`

## All supported model architectures
The following architectures are implemented via the `--model-name` argument:
```DenseNet121,DenseNet169,DenseNet201,InceptionV3,MobileNet,MobileNetV2,NASNetLarge,NASNetMobile,ResNet101,ResNet101V2,ResNet152,ResNet152V2,ResNet50,ResNet50V2,VGG16,VGG19,Xception,fcn_32,fcn_32_mobilenet,fcn_32_resnet50,fcn_32_vgg,fcn_8,fcn_8_mobilenet,fcn_8_resnet50,fcn_8_vgg,linear0,linear1,linear10,linear11,linear12,linear13,linear14,linear15,linear16,linear17,linear18,linear19,linear2,linear20,linear21,linear22,linear23,linear24,linear25,linear26,linear27,linear28,linear29,linear3,linear30,linear31,linear4,linear5,linear6,linear7,linear8,linear9,mobilenet_segnet,mobilenet_unet,pspnet,pspnet_101,pspnet_50,resnet50_pspnet,resnet50_segnet,resnet50_unet,segnet,test,unet,unet_mini,vgg_pspnet,vgg_segnet,vgg_unet```
