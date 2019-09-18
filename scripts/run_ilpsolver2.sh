#!/bin/sh
export RAY_NUM_CPU=96
#python src/eval_runner.py -b 512 --model-name MobileNet --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 512 --model-name MobileNet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 1024 --model-name MobileNet --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 1024 --model-name MobileNet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 2048 --model-name MobileNet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 5096 --model-name MobileNet --platform flops --mode budget_sweep

#python src/eval_runner.py -b 256 --model-name MobileNet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 32 --model-name vgg_unet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 64 --model-name vgg_unet --platform flops --mode budget_sweep
#python src/eval_runner.py -b 256 --model-name VGG16 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 512 --model-name VGG16 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name VGG16 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 16 --model-name vgg_unet --platform flops --mode budget_sweep

python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=ResNet50 --platform=p32xlarge -b=128 --overwrite
python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=ResNet50 --platform=p32xlarge -b=512 --overwrite

