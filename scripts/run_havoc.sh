#!/bin/sh
export RAY_NUM_CPU=63
python src/eval_runner.py -b 32 --model-name fcn_8 --platform flops --mode budget_sweep
python src/eval_runner.py -b 64 --model-name fcn_8 --platform flops --mode budget_sweep
python src/eval_runner.py -b 16 --model-name fcn_8 --platform flops --mode budget_sweep
python src/eval_runner.py -b 128 --model-name ResNet152 --platform flops --mode budget_sweep
python src/eval_runner.py -b 256 --model-name ResNet152 --platform flops --mode budget_sweep
python src/eval_runner.py -b 16 --model-name resnet50_pspnet --platform flops --mode budget_sweep

