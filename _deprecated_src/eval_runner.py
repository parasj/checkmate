import argparse
import os
import pathlib
import shutil
import uuid
from typing import List, Dict, Any

import dotenv
import ray

from evaluation.budget_sweep import eval_budget_sweep
from evaluation.maximize_batch_size import eval_maximize_batch_size
from evaluation.solve_time_plot import eval_solve_time
from remat.core.solvers.enum_strategy import SolveStrategy
from experiments.common.keras_extractor import MODEL_NAMES
from utils.redis import RedisCache
from utils.setup_logger import setup_logger

PLATFORM_CHOICES = ['p32xlarge', 'p32xlarge_fp16', 'p2xlarge', 'c524xlarge', 'flops']


def extract_params():
    # todo split into hierarchical args per eval script like:
    # https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["budget_sweep", "solve_timer", "max_batch_size"],
                        default="budget_sweep")
    parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument('--model-name', default="ResNet50", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument('--model-version', default="v1",
                        help="Version number for model. Increment to ignore existing cached solutions for a model.")
    parser.add_argument('--ilp-eval-points', nargs='+', type=int, default=[],
                        help="If set, will only search a specific set of ILP points in MB, else will perform global search.")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite results in redis cache. If not set, results will be loaded from Redis.")

    # Budget sweep specific arguments
    parser.add_argument('--skip-ilp', action='store_true', help="If set, skip running the ILP during evaluation.")
    parser.add_argument('--ilp-time-limit', type=int, default=3600, help="Time limit for individual ILP solves, in sec")
    parser.add_argument('--hide-points', action="store_true")
    parser.add_argument('--develop', action="store_true", help="If true, will not reload cached values during ILP solves")

    args = parser.parse_args()
    args.input_shape = args.input_shape if args.input_shape else None
    print("Args:", args)
    return args


def setup_environment(log_base: str = None, logger=setup_logger("setup_environment"), init_ray=True, overwrite=False):
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"

    if os.cpu_count() > 16 and "ILP_CORES" not in os.environ.keys():
        os.environ["ILP_CORES"] = "12"

    if overwrite:
        os.environ["REDIS_OVERWRITE"] = "1"

    # load redis config
    dotenv_location = dotenv.find_dotenv()
    if len(dotenv_location):
        logger.info(f'Loading dotenv config from {dotenv_location}')
        dotenv.load_dotenv(dotenv_location)
    else:
        logger.warn("Failed to load dotenv config!")

    # initialize ray
    if init_ray:
        RAY_REDIS_ADDRESS = os.environ.get("RAY_REDIS_ADDRESS")
        RAY_REDIS_PASSWORD = os.environ.get("RAY_REDIS_PASSWORD", str(uuid.uuid1()))
        if RAY_REDIS_ADDRESS is None:  # local mode
            num_cpus = int(os.environ.get("RAY_NUM_CPU", os.cpu_count() - 1))
            logger.info(f'Initializing local mode ray with {num_cpus} cpus')
            ray.init(temp_dir="/tmp/ray_checkpoint", redis_password=RAY_REDIS_PASSWORD,
                     num_cpus=num_cpus,
                     object_store_memory=1024 * 1024 * 1024 if os.cpu_count() < 48 else None)
        else:
            ray.init(redis_address=RAY_REDIS_ADDRESS, redis_password=RAY_REDIS_PASSWORD)

    if log_base is not None:
        # confirm("This will overwrite plots under {}, are you sure?".format(log_base))
        shutil.rmtree(log_base, ignore_errors=True)
        pathlib.Path(log_base).mkdir(parents=True)


def get_ray_nodes():
    @ray.remote
    def f():
        import time
        import socket
        time.sleep(0.001)
        return socket.gethostname()

    return set(ray.get([f.remote() for _ in range(1000)]))


def get_metrics(results, tputs):
    metrics = {}
    for result, tput in zip(results, tputs):
        if result.solve_strategy == SolveStrategy.OPTIMAL_ILP_GC:
            metrics['vars'] = result.ilp_num_variables
            metrics['constraints'] = result.ilp_num_constraints
            metrics['solve_time'] = result.solve_time_s
            metrics['ilp throughput'] = tput
        if result.solve_strategy == SolveStrategy.CHEN_GREEDY:
            metrics['greedy throughput'] = tput
        if result.solve_strategy == SolveStrategy.CHECKPOINT_ALL:
            metrics['all throughput'] = tput
    metrics['speedup'] = metrics['ilp throughput'] / metrics['greedy throughput']
    return metrics


def dict_list_to_list_dict(dl: List[Dict[Any, Any]]) -> Dict[Any, List[Any]]:
    keys = dl[0].keys()
    ld = {key: [] for key in keys}
    for d in dl:
        for k, v in d.items():
            ld[k].append(v)
    return ld


def main():
    args = extract_params()
    key_prefix = RedisCache.make_key(args.platform, args.model_name, args.model_version, args.batch_size,
                                     args.input_shape)
    log_base = os.path.join("data", args.mode, key_prefix.replace("/", "_"))
    cost_file = os.path.join("profiles", args.model_name, f"b{args.batch_size}_{args.platform}.npy")
    cost_file = cost_file if os.path.exists(cost_file) else None
    if args.mode == "budget_sweep":
        # if args.overwrite:
        #     confirm("This will overwrite existing Redis cache keys {}/..., are you sure?".format(key_prefix))
        setup_environment(log_base=log_base, overwrite=args.overwrite)
        logger = setup_logger("eval_runner", os.path.join(log_base, "eval_runner.log"))
        logger.info(f"Ray nodes: {get_ray_nodes()}")
        logger.info(f"Using profile file {cost_file}")

        eval_budget_sweep(args, log_base=log_base, key_prefix=key_prefix, cost_file=cost_file)
        ray.timeline(filename=os.path.join(log_base, "timeline.json"))
    elif args.mode == "solve_timer":
        log_base = os.path.join("data", "solve_timer")
        setup_environment(log_base=log_base, init_ray=False)
        eval_solve_time(args, log_base)
    elif args.mode == "max_batch_size":
        setup_environment(log_base=log_base, overwrite=args.overwrite, init_ray=False)
        logger = setup_logger("max_batch_size", os.path.join(log_base, "max_batch_size.log"))
        logger.info(f"Using profile file {cost_file}")
        eval_maximize_batch_size(args, log_base=log_base, key_prefix=key_prefix, cost_file=cost_file)


if __name__ == "__main__":
    main()
