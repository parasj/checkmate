#!/bin/sh
export RAY_NUM_CPU=96
#python src/eval_runner.py -b 256 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 128 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 512 --model-name ResNet50 --platform p32xlarge --mode budget_sweep

#python src/eval_runner.py -b 16 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 64 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 512 --model-name ResNet50 --platform flops --mode budget_sweep
#python src/eval_runner.py -b 32 --model-name ResNet50 --platform p32xlarge --mode budget_sweep
#python src/eval_runner.py -b 16 --model-name ResNet50 --platform flops --mode budget_sweep

python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=MobileNet --platform=p32xlarge -b=128 --overwrite
python src/eval_runner.py --ilp-eval-points 14000 14300 14600 14900 15000 15200 15500 15800 15900 16000 --model-name=VGG16 --platform=p32xlarge -b 256 --overwrite


