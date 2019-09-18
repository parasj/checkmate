#!/bin/sh
export RAY_NUM_CPU=96
#python src/eval_runner.py -b 256 --model-name MobileNetV2 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name MobileNetV2 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 64 --model-name DenseNet128 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name DenseNet128 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name NASNetMobile --platform flops --mode budget_sweep
#python src/eval_runner.py -b 64 --model-name NASNetMobile --platform flops --mode budget_sweep
#python src/eval_runner.py -b 256 --model-name NASNetMobile --platform flops --mode budget_sweep

#python src/eval_runner.py -b 256 --model-name VGG19 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 32 --model-name vgg_pspnet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 16 --model-name vgg_pspnet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name VGG19 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name ResNet101 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 256 --model-name ResNet101 --platform flops --mode budget_sweep

python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=vgg_unet --platform=p32xlarge -b 8 --overwrite
python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=vgg_unet --platform=p32xlarge -b 16 --overwrite
python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=vgg_unet --platform=p32xlarge -b 32 --overwrite
python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=vgg_unet --platform=p32xlarge -b 64 --overwrite
