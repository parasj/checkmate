#!/usr/bin/env bash
python src/eval_runner.py --mode=budget_sweep --model-name=MobileNet --platform=flops
python src/eval_runner.py --mode=budget_sweep --model-name=MobileNet --platform=p32xlarge
python src/eval_runner.py --mode=budget_sweep --model-name=MobileNet --platform=p2xlarge
