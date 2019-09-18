#!/usr/bin/env bash
python src/eval_runner.py --mode=budget_sweep --model-name=vgg_unet --platform=flops
python src/eval_runner.py --mode=budget_sweep --model-name=vgg_unet --platform=p32xlarge
python src/eval_runner.py --mode=budget_sweep --model-name=vgg_unet --platform=p2xlarge
