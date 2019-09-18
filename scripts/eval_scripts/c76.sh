#!/usr/bin/env bash
export RAY_NUM_CPU=36
python src/eval_runner.py --mode=budget_sweep --model-name=vgg_unet --platform=p32xlarge --batch-size=8
python src/eval_runner.py --mode=budget_sweep --model-name=vgg_unet --platform=p32xlarge --batch-size=16
python src/eval_runner.py --mode=budget_sweep --model-name=VGG16 --platform=p32xlarge --batch-size=128
python src/eval_runner.py --mode=budget_sweep --model-name=MobileNet --platform=p32xlarge --batch-size=128
