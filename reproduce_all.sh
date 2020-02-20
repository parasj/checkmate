#!/bin/bash
# Thank you to the reproducibility reviewers for MLSys 2020 who provided
# the following scripts to replicate our paper's results.

commands=(
    'python experiments/experiment_budget_sweep.py --model-name "VGG16" -b 256 --platform p32xlarge'
    'python experiments/experiment_budget_sweep.py --model-name "MobileNet" -b 512 --platform p32xlarge'
    'python experiments/experiment_budget_sweep.py --model-name "vgg_unet" -b 32 --platform p32xlarge'
    'python experiments/experiment_max_batchsize_baseline.py --model-name vgg_unet --batch-size-min 10 --batch-size-max 40 --batch-size-increment 1'
    'python experiments/experiment_max_batchsize_baseline.py --model-name fcn_8_vgg --batch-size-min 10 --batch-size-max 80 --batch-size-increment 1'
    'python experiments/experiment_max_batchsize_baseline.py --model-name segnet --batch-size-min 20 --batch-size-max 50 --batch-size-increment 1'
    'python experiments/experiment_max_batchsize_baseline.py --model-name ResNet50 --batch-size-min 90 --batch-size-max 200 --batch-size-increment 1'
    'python experiments/experiment_max_batchsize_baseline.py --model-name VGG19 --batch-size-min 160 --batch-size-max 300 --batch-size-increment 1'
    'python experiments/experiment_max_batchsize_baseline.py --model-name MobileNet --batch-size-min 200 --batch-size-max 650 --batch-size-increment 1'
)
rm -rf stdout_err
mkdir stdout_err
rm results.txt
index=1
for i in "${commands[@]}"; do
    echo $i
    start=$(date +%s%N | cut -b1-13)
    eval "$i" &>stdout_err/$index.txt
    end=$(date +%s%N | cut -b1-13)
    runtime=$((end - start))
    echo $i >>results.txt
    echo "$runtime ms" >>results.txt
    ((index = index + 1))
done

commands=(
    'python experiments/experiment_max_batchsize_ilp.py --model-name vgg_unet --batch-size-min 20 --num-threads 40'
    'python experiments/experiment_max_batchsize_ilp.py --model-name fcn_8_vgg --batch-size-min 20 --num-threads 40'
    'python experiments/experiment_max_batchsize_ilp.py --model-name segnet --batch-size-min 20 --num-threads 40'
    'python experiments/experiment_max_batchsize_ilp.py --model-name ResNet50 --batch-size-min 100 --num-threads 40'
    'python experiments/experiment_max_batchsize_ilp.py --model-name VGG19 --batch-size-min 160 --num-threads 40'
    'python experiments/experiment_max_batchsize_ilp.py --model-name MobileNet --batch-size-min 450 --num-threads 40'
)
rm -rf max_batchsize_ilp
mkdir max_batchsize_ilp
index=1
for i in "${commands[@]}"; do
    echo $i
    eval "$i &> max_batchsize_ilp/$index.txt &"
    pid=$!
    sleep 1800 # change 1800 to any number to have a larger or smaller timeout
    kill -SIGINT $pid
    ((index = index + 1))
done
