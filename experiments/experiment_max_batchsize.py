import argparse
import logging
import os
import pathlib
import shutil
import uuid
from typing import Dict, List

import numpy as np
import pandas
import tensorflow as tf
import ray

from experiments.common.keras_extractor import MODEL_NAMES, get_keras_model, CHAIN_GRAPH_MODELS
from experiments.common.plotting.graph_plotting import render_dfgraph
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory
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
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])

    _args = parser.parse_args()
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    args = extract_params()

    ray.init(temp_dir="/tmp/ray_checkpoint", redis_password=str(uuid.uuid1()), num_cpus=os.cpu_count(),
             object_store_memory=1024 * 1024 * 1024 if os.cpu_count() < 48 else None)

    key = "_".join(map(str, [args.platform, args.model_name, args.input_shape]))
    log_base = os.path.join("data", "max_batch_size", key)
    shutil.rmtree(log_base, ignore_errors=True)
    pathlib.Path(log_base).mkdir(parents=True)
    result_dict: Dict[SolveStrategy, List[ScheduledResult]] = {}
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
    tf.keras.utils.plot_model(model, to_file=os.path.join(log_base, f"plot_{model_name}.png"),
                              show_shapes=True, show_layer_names=True)

    max_batch_sizes = {}
    platform_ram = platform_memory(args.platform)
    for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        logging.info(f"Sweeping batch size = {bs}")
        # load model at batch size
        logging.info(f"Loading model {model_name}")
        g = dfgraph_from_keras(model, batch_size=bs, cost_model=cost_model,
                               loss_cpu_cost=0, loss_ram_cost=(4 * bs))
        render_dfgraph(g, log_base, name=model_name)

        # run constant baselines
        logging.info(f"Running constant baselines (ALL, ALL_AP, LAST_NODE, SQRTN_NOAP, SQRTN)")
        result_dict[SolveStrategy.CHECKPOINT_ALL] = [solve_checkpoint_all(g)]
        result_dict[SolveStrategy.CHECKPOINT_ALL_AP] = [solve_checkpoint_all_ap(g)]
        result_dict[SolveStrategy.CHECKPOINT_LAST_NODE] = [solve_checkpoint_last_node(g)]
        result_dict[SolveStrategy.CHEN_SQRTN_NOAP] = [solve_chen_sqrtn(g, False)]
        result_dict[SolveStrategy.CHEN_SQRTN] = [solve_chen_sqrtn(g, True)]

        # sweep chen's greedy baseline
        logging.info(f"Running Chen's greedy baseline (APs only)")
        chen_sqrtn_noap = result_dict[SolveStrategy.CHEN_SQRTN_NOAP][0]
        greedy_eval_points = chen_sqrtn_noap.schedule_aux_data.activation_ram * (1. + np.arange(-1, 2, 0.01))
        remote_solve_chen_greedy = ray.remote(num_cpus=1)(solve_chen_greedy).remote
        chen_futures = [remote_solve_chen_greedy(g, float(b), False) for b in greedy_eval_points]
        if model_name not in CHAIN_GRAPH_MODELS:
            logging.info(f"Running Chen's greedy baseline (no AP) as model is non-linear")
            chen_chain_futures = [remote_solve_chen_greedy(g, float(b), True) for b in greedy_eval_points]

        # sweep griewank baselines
        if model_name in CHAIN_GRAPH_MODELS:
            logging.info(f"Running Griewank baseline (APs only)")
            griewank_eval_points = range(1, g.size + 1)
            remote_solve_griewank = ray.remote(num_cpus=1)(solve_griewank).remote
            griewank_futures = [remote_solve_griewank(g, float(b)) for b in griewank_eval_points]

        # load futures results
        result_dict[SolveStrategy.CHEN_GREEDY] = get_futures(list(chen_futures), desc="Greedy (APs only)")
        if model_name in CHAIN_GRAPH_MODELS:
            result_dict[SolveStrategy.GRIEWANK_LOGN] = get_futures(list(griewank_futures), desc="Griewank (APs only)")
        else:
            result_dict[SolveStrategy.CHEN_SQRTN_NOAP] = get_futures(list(chen_chain_futures), desc="Greedy (No AP)")

        fwd2xcpu = sum(g.cost_cpu_fwd.values()) + sum(g.cost_cpu.values())
        for strategy, results in result_dict.items():
            for result in results:
                if result.schedule_aux_data is not None and result.schedule_aux_data.peak_ram <= platform_ram and result.schedule_aux_data.cpu <= fwd2xcpu:
                    max_batch_sizes[strategy] = bs

    df = pandas.DataFrame([{'strategy': k, 'batch_size': v} for k, v in max_batch_sizes.items()])
    print(df)
    df.to_csv(os.path.join(log_base, 'max_batch_size.csv'))
