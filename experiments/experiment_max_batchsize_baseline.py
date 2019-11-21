import argparse
import logging
import os
import pathlib
import shutil
import uuid
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas
import tensorflow as tf
import ray
from tqdm import tqdm

from experiments.common.definitions import remat_data_dir
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model
from experiments.common.graph_plotting import render_dfgraph
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory
from experiments.common.ray_utils import get_futures
from remat.core.schedule import ScheduledResult
from remat.core.enum_strategy import SolveStrategy
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from remat.tf2_keras.extraction import dfgraph_from_keras


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument('--model-name', default="VGG16", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])
    parser.add_argument("--batch-size-min", type=int, default=4)
    parser.add_argument("--batch-size-max", type=int, default=512)
    parser.add_argument("--batch-size-increment", type=int, default=8)

    _args = parser.parse_args()
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    args = extract_params()

    key = "_".join(map(str, [args.platform, args.model_name, args.input_shape]))
    log_base = remat_data_dir() / "max_batch_size" / key
    shutil.rmtree(log_base, ignore_errors=True)
    pathlib.Path(log_base).mkdir(parents=True, exist_ok=True)
    result_dict: Dict[int, Dict[SolveStrategy, List[ScheduledResult]]] = defaultdict(lambda: defaultdict(list))
    model_name = args.model_name

    # load costs, and plot optionally, if platform is not flops
    logging.info(f"Loading costs")
    if args.platform == "flops":
        cost_model = None
    else:
        cost_model = CostModel(model_name, args.platform, log_base, quantization=5)
        cost_model.fit()
        cost_model.plot_costs()

    model = get_keras_model(model_name, input_shape=args.input_shape)
    tf.keras.utils.plot_model(model, to_file=log_base / f"plot_{model_name}.png",
                              show_shapes=True, show_layer_names=True)

    platform_ram = platform_memory("p32xlarge")
    bs_futures: Dict[int, List] = defaultdict(list)
    bs_param_ram_cost: Dict[int, int] = {}
    bs_fwd2xcost: Dict[int, int] = {}
    rg = list(range(args.batch_size_min, args.batch_size_max, args.batch_size_increment))
    for bs in tqdm(rg, desc="Event dispatch"):
        while not ray.is_initialized():
            ray.init(temp_dir="/tmp/ray_checkpoint_" + str(str(uuid.uuid4())[:8]), redis_password=str(uuid.uuid1()),
                     num_cpus=os.cpu_count() - 2)
        futures = []

        # load model at batch size
        g = dfgraph_from_keras(model, batch_size=bs, cost_model=cost_model, loss_cpu_cost=0, loss_ram_cost=(4 * bs))
        bs_fwd2xcost[bs] = sum(g.cost_cpu_fwd.values()) + sum(g.cost_cpu.values())
        bs_param_ram_cost[bs] = g.cost_ram_fixed
        render_dfgraph(g, log_base, name=model_name)

        # run constant baselines
        result_dict[bs][SolveStrategy.CHEN_SQRTN_NOAP] = [solve_chen_sqrtn(g, False)]
        futures.extend([
            ray.remote(num_cpus=1)(solve_checkpoint_all).remote(g),
            ray.remote(num_cpus=1)(solve_checkpoint_all_ap).remote(g),
            ray.remote(num_cpus=1)(solve_checkpoint_last_node).remote(g),
            ray.remote(num_cpus=1)(solve_chen_sqrtn).remote(g, True),
            ray.remote(num_cpus=1)(solve_chen_sqrtn).remote(g, False)
        ])

        # sweep chen's greedy baseline
        chen_sqrtn_noap = result_dict[bs][SolveStrategy.CHEN_SQRTN_NOAP][0]
        greedy_eval_points = chen_sqrtn_noap.schedule_aux_data.activation_ram * (1. + np.arange(-1, 2, 0.05))
        remote_solve_chen_greedy = ray.remote(num_cpus=1)(solve_chen_greedy).remote
        futures.extend([remote_solve_chen_greedy(g, float(b), False) for b in greedy_eval_points])
        futures.extend([remote_solve_chen_greedy(g, float(b), True) for b in greedy_eval_points])

        # # sweep griewank baselines
        # if model_name in CHAIN_GRAPH_MODELS:
        #     solve_griewank(g, 1)  # prefetch griewank solution from s3, otherwise ray will cause race condition
        #     griewank_eval_points = range(1, g.size + 1)
        #     remote_solve_griewank = ray.remote(num_cpus=1)(solve_griewank).remote
        #     futures.extend([remote_solve_griewank(g, float(b)) for b in griewank_eval_points])

        for result in get_futures(futures, desc=f"Batch size: {bs}"):
            result_dict[bs][result.solve_strategy].append(result)

        ray.shutdown()

    max_batch_sizes = defaultdict(int)
    for bs, strategy_results in result_dict.items():
        for strategy, results in strategy_results.items():
            is_valid = lambda r: r.schedule_aux_data is not None \
                                 and r.schedule_aux_data.peak_ram <= platform_ram - bs_param_ram_cost[bs] \
                                 and r.schedule_aux_data.cpu <= bs_fwd2xcost[bs]
            if any(map(is_valid, results)):
                max_batch_sizes[strategy] = max(bs, max_batch_sizes[strategy])
                logging.info(f"SolveStrategy {strategy} succeeded at batch size {bs}")

    df = pandas.DataFrame([{'strategy': k, 'batch_size': v} for k, v in max_batch_sizes.items()])
    df.to_csv(log_base / 'max_batch_size.csv')
    print(df)
