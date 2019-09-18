#!/bin/sh
export RAY_NUM_CPU=60
python src/eval_runner.py -b 256 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 128 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 512 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
