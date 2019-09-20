import argparse
import logging
import os
import uuid
from typing import Dict, List

import numpy as np
import ray

from experiments.common.cost_model import CostModel
from experiments.common.keras_extractor import MODEL_NAMES, get_keras_model, CHAIN_GRAPH_MODELS
from experiments.common.platforms import PLATFORM_CHOICES
from experiments.common.utils import get_futures
from remat.core.schedule import ScheduledResult
from remat.core.solvers.enum_strategy import SolveStrategy
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from remat.core.solvers.strategy_griewank import solve_griewank
from remat.tensorflow2.extraction import dfgraph_from_keras


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument('--model-name', default="VGG16", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument('--ilp-eval-points', nargs='+', type=int, default=[],
                        help="If set, will only search a specific set of ILP points in MB, else perform global search.")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])

    parser.add_argument('--skip-ilp', action='store_true', help="If set, skip running the ILP during evaluation.")
    parser.add_argument('--ilp-time-limit', type=int, default=3600, help="Time limit for individual ILP solves, in sec")
    parser.add_argument('--hide-points', action="store_true")

    _args = parser.parse_args()
    if _args.skip_ilp and len(_args.ilp_eval_points) > 0:
        parser.error("--skip-ilp and --ilp-eval-points cannot both be set")
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


if __name__ == "__main__":
    logger = logging.getLogger("budget_sweep")
    logger.setLevel(logging.DEBUG)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    if os.cpu_count() > 12 and "ILP_CORES" not in os.environ.keys():
        os.environ["ILP_CORES"] = "12"
    args = extract_params()

    ray.init(temp_dir="/tmp/ray_checkpoint", redis_password=str(uuid.uuid1()), num_cpus=os.cpu_count(),
             object_store_memory=1024 * 1024 * 1024 if os.cpu_count() < 48 else None)

    key = "_".join(map(str, [args.platform, args.model_name, args.batch_size, args.input_shape]))
    log_base = os.path.join("data", "budget_sweep", key)

    # todo make web url fetcher w/ local cache to get profiles from from S3
    cost_file = os.path.join("profiles", args.model_name, f"b{args.batch_size}_{args.platform}.npy")
    cost_file = cost_file if os.path.exists(cost_file) else None

    ####
    # Begin budget_sweep data collection
    ####
    result_dict: Dict[SolveStrategy, List[ScheduledResult]] = {}
    model_name = args.model_name

    # load costs, and plot optionally, if platform is not flops
    logger.info(f"Loading costs")
    if args.platform != "flops":
        cost_model = CostModel(model_name, args.platform, log_base, quantization=5)
        cost_model.fit()
        cost_model.plot_costs()
        costs_np = cost_model.get_costs(args.batch_size)
    else:
        costs_np = None

    # load model from Keras
    logger.info(f"Loading model {model_name}")
    model = get_keras_model(model_name, input_shape=args.input_shape)
    g = dfgraph_from_keras(model, batch_size=args.batch_size, costs_np=costs_np,
                           loss_cpu_cost=0, loss_ram_cost=(4 * args.batch_size))

    # sweep constant baselines
    logger.info(f"Running constant baselines (ALL, ALL_AP, LAST_NODE, SQRTN_NOAP, SQRTN)")
    result_dict[SolveStrategy.CHECKPOINT_ALL] = [solve_checkpoint_all(g)]
    result_dict[SolveStrategy.CHECKPOINT_ALL_AP] = [solve_checkpoint_all_ap(g)]
    result_dict[SolveStrategy.CHECKPOINT_LAST_NODE] = [solve_checkpoint_last_node(g)]
    result_dict[SolveStrategy.CHEN_SQRTN_NOAP] = [solve_chen_sqrtn(g, False)]
    result_dict[SolveStrategy.CHEN_SQRTN] = [solve_chen_sqrtn(g, True)]

    # sweep chen's greedy baseline
    logger.info(f"Running Chen's greedy baseline (APs only)")
    greedy_eval_points = result_dict[SolveStrategy.CHEN_SQRTN_NOAP][0].schedule_aux_data.activation_ram * (1. + np.arange(-1, 2, 0.01))
    remote_solve_chen_greedy = ray.remote(num_cpus=1)(solve_chen_greedy).remote
    futures = [remote_solve_chen_greedy(g, float(b), False) for b in greedy_eval_points]
    result_dict[SolveStrategy.CHEN_GREEDY] = get_futures(list(futures), desc="Greedy (APs only)")
    if model_name not in CHAIN_GRAPH_MODELS:
        logger.info(f"Running Chen's greedy baseline (no AP) as model is non-linear")
        futures = [remote_solve_chen_greedy(g, float(b), True) for b in greedy_eval_points]
        result_dict[SolveStrategy.CHEN_SQRTN_NOAP] = get_futures(list(futures), desc="Greedy (No AP)")

    # sweep griewank baselines
    logger.info(f"Running Griewank baseline (APs only)")
    griewank_eval_points = range(1, g.size + 1)
    remote_solve_griewank = ray.remote(num_cpus=1)(solve_griewank).remote
    futures = [remote_solve_griewank(g, float(b)) for b in griewank_eval_points]
    result_dict[SolveStrategy.GRIEWANK_LOGN] = get_futures(list(futures), desc="Griewank (APs only)")

    # sweep optimal ilp baseline
    if not args.skip_ilp:
        # todo load any ILP results from cache
        if len(args.ilp_eval_points) > 0:
            local_ilp_eval_points = [p * 1000 * 1000 - g.cost_ram_fixed for p in args.ilp_eval_points]
        else:  # run global search routine
            pass

        # run local search routine

    ####
    # Plot result_dict
    ####

