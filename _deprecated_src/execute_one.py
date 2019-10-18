#!/bin/env/python
# Executes a model under a single batch size, input size, solver, and platform configuration

import argparse
import os

import pickle

import dotenv

from remat.core.enum_strategy import SolveStrategy
from experiments.common.load_keras_model import MODEL_NAMES
from evaluation.eval_execution import execute_one
from utils.setup_logger import setup_logger


def extract_params():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', choices=MODEL_NAMES, required=True)
    parser.add_argument('--model-version', default="v1",
                        help="Version number for model. Increment to ignore existing"
                             "cached solutions for a model.")
    parser.add_argument("-b", "--batch-size", type=int, required=True)
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])
    parser.add_argument('--platform', default="flops", choices=['p32xlarge', 'p2xlarge', 'c524xlarge', 'flops'])
    parser.add_argument('--num-runs', type=int, required=True,
                        help="number of times to run the model for exec testing.  not used otherwise")
    parser.add_argument('--strategy', type=str, required=True,
                        choices=[ss.value for ss in SolveStrategy])
    parser.add_argument('--buffer-mem-mb', type=int, default=0, help='Don\'t allow solutions within this buffer'
            'of the platform budget')
    parser.add_argument('--mode', type=str, choices=["run_single_model", "get_allstrat_ram"], default="run_single_model")

    args = parser.parse_args()
    args.input_shape = None
    return args


def run_single_model(args):
    log_base = os.path.join("data", "run_single_model",
                            f"{args.platform}_{args.model_name}_{args.model_version}_{args.batch_size}_{args.input_shape}_{args.strategy}_{args.buffer_mem_mb}_gradless_eagerfalse")
    os.makedirs(log_base, exist_ok=True)
    logger = setup_logger("run_single_model")

    # load redis config
    dotenv_location = dotenv.find_dotenv()
    if len(dotenv_location):
        logger.info(f'Loading dotenv config from {dotenv_location}')
        dotenv.load_dotenv(dotenv_location)
    else:
        logger.warn("Failed to load dotenv config!")

    strategy = SolveStrategy(args.strategy)
    result, result_key, throughput = execute_one(log_base=log_base,
                                                 solve_strategy=strategy,
                                                 model_name=args.model_name,
                                                 batch_size=args.batch_size,
                                                 platform=args.platform,
                                                 input_shape=args.input_shape,
                                                 model_version=args.model_version,
                                                 num_runs=args.num_runs,
                                                 buffer_mem=args.buffer_mem_mb * 1000 * 1000)

    if result is None:
        logger.error("No result returned from execute_one")
        return

    metrics_single = dict(
        solve_strategy=strategy,
        model_name=args.model_name,
        batch_size=args.batch_size,
        platform=args.platform,
        input_shape=args.input_shape,
        model_version=args.model_version,
        num_runs=args.num_runs,
        result_key=result_key,
        buffer_mem=args.buffer_mem_mb * 1000 * 1000,
        throughput_it_per_s=throughput,
    )

    if strategy == SolveStrategy.OPTIMAL_ILP_GC:
        metrics_single["vars"] = result.ilp_num_variables
        metrics_single["constraints"] = result.ilp_num_constraints
        metrics_single["solve_time"] = result.solve_time_s

    output_file = os.path.join(log_base, "metrics.pickle")
    with open(output_file, "wb") as f:
        pickle.dump(metrics_single, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved throughput metrics to {output_file}")


def get_allstrat_ram(args):
    from evaluation.eval_execution import get_solution_to_evaluate

    log_base = os.path.join("data", "get_allstrat_ram",
                            f"{args.platform}_{args.model_name}_{args.model_version}_{args.batch_size}_{args.input_shape}_{args.strategy}_{args.buffer_mem_mb}_gradless_eagerfalse")
    os.makedirs(log_base, exist_ok=True)
    logger = setup_logger("get_allstrat_ram")

    # load redis config
    dotenv_location = dotenv.find_dotenv()
    if len(dotenv_location):
        logger.info(f'Loading dotenv config from {dotenv_location}')
        dotenv.load_dotenv(dotenv_location)
    else:
        logger.warn("Failed to load dotenv config!")

    for strategy in SolveStrategy:
        if strategy == SolveStrategy.NOT_SPECIFIED:
            continue
        result, result_key = get_solution_to_evaluate(strategy, args.model_name, args.batch_size, args.platform, args.input_shape, args.model_version,
                                                      args.buffer_mem_mb)

        if result:
            logger.info(f"For strategy {strategy.name}, peak ram is {result.peak_ram:.3E}, compute is {result.cpu:.3E}")
        else:
            logger.warn(f"no solution for strategy {strategy.name}")


if __name__ == "__main__":
    args = extract_params()
    if args.mode == "run_single_model":
        run_single_model(args)
    elif args.mode == "get_allstrat_ram":
        get_allstrat_ram(args)
