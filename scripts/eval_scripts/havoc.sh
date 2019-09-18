#!/usr/bin/env bash
python src/eval_runner.py --mode=budget_sweep --model-name=ResNet50 --platform=flops
python src/eval_runner.py --mode=budget_sweep --model-name=VGG16 --platform=flops
python src/eval_runner.py --mode=budget_sweep --model-name=VGG16 --platform=p32xlarge
python src/eval_runner.py --mode=budget_sweep --model-name=VGG16 --platform=p2xlarge


