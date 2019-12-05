import argparse
import datetime
import logging
import math
import os
import pathlib
import shutil
from collections import defaultdict
from typing import Dict, List

import tensorflow as tf

from experiments.common.definitions import checkmate_data_dir
from experiments.common.graph_plotting import plot_dfgraph, plot_schedule
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory
from experiments.solver_ilp_max_batchsize import MaxBatchILPSolver
from checkmate.core.enum_strategy import SolveStrategy
from checkmate.core.schedule import ScheduledResult
from checkmate.tf2_keras.extraction import dfgraph_from_keras

GB = 1000 * 1000 * 1000


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument("--model-name", default="VGG16", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])
    parser.add_argument("--batch-size-min", type=int, default=1)

    _args = parser.parse_args()
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    args = extract_params()

    key = "_".join(map(str, [args.platform, args.model_name, args.input_shape]))
    log_base = checkmate_data_dir() / "max_batch_size_ilp" / key / str(datetime.datetime.now().isoformat())
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
    tf.keras.utils.plot_model(model, to_file=log_base / f"plot_{model_name}.png", show_shapes=True, show_layer_names=True)

    platform_ram = platform_memory("p32xlarge")
    bs_futures: Dict[int, List] = defaultdict(list)
    bs_fwd2xcost: Dict[int, int] = {}
    # load model at batch size
    g = dfgraph_from_keras(model, batch_size=1, cost_model=cost_model, loss_cpu_cost=0, loss_ram_cost=(4))
    plot_dfgraph(g, log_base, name=model_name)

    model_file = str(log_base / f"max_bs_{model_name}.mps")
    param_dict = {
        "LogToConsole": 1,
        "LogFile": str(log_base / f"max_bs_{model_name}.solve.log"),
        "Threads": os.cpu_count(),
        "TimeLimit": math.inf,
    }
    ilp_solver = MaxBatchILPSolver(
        g,
        budget=platform_memory("p32xlarge") - g.cost_ram_fixed,
        model_file=model_file,
        gurobi_params=param_dict,
        cpu_fwd_factor=2,
    )
    ilp_solver.build_model()
    result, batch_size = ilp_solver.solve()
    logging.info(f"Max batch size = {batch_size}")

    save_file = log_base / f"{model}_plot.png"
    plot_schedule(result, plot_mem_usage=True, save_file=save_file)
