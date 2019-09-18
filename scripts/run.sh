#!/bin/sh
export RAY_NUM_CPU=96
python src/eval_runner.py -b 256 --model-name VGG16 --platform flops --mode budget_sweep
python src/eval_runner.py -b 256 --model-name VGG16 --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 256 --model-name MobileNet --platform flops --mode budget_sweep
python src/eval_runner.py -b 256 --model-name MobileNet --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 32 --model-name vgg_unet --platform flops --mode budget_sweep
python src/eval_runner.py -b 32 --model-name vgg_unet --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 256 --model-name ResNet50 --platform flops --mode budget_sweep
python src/eval_runner.py -b 256 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
python src/eval_runner.py -b 64 --model-name vgg_unet --platform flops --mode budget_sweep
python src/eval_runner.py -b 64 --model-name vgg_unet --platform p32xlarge --mode budget_sweep
