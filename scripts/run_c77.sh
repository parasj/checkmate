#!/bin/sh
export RAY_NUM_CPU=60
python src/eval_runner.py -b 32 --model-name vgg_unet --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 16 --model-name vgg_unet --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 512 --model-name VGG16 --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 128 --model-name VGG16 --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 512 --model-name MobileNet --platform p32xlarge --mode budget_sweep
