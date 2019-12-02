import argparse
import logging
import os
import pathlib
import pickle
import shutil
import uuid
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import seaborn as sns
import tensorflow as tf
from matplotlib.lines import Line2D
from scipy.stats.mstats import gmean

from experiments.common.definitions import remat_data_dir
from experiments.common.graph_plotting import plot_dfgraph
from experiments.common.load_keras_model import MODEL_NAMES, get_keras_model, CHAIN_GRAPH_MODELS
from experiments.common.profile.cost_model import CostModel
from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory, pretty_platform_name
from experiments.common.ray_utils import get_futures
from checkmate.core.dfgraph import DFGraph
from checkmate.core.enum_strategy import SolveStrategy
from checkmate.core.schedule import ScheduledResult
from checkmate.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_sweep, \
    solve_approx_lp_deterministic_rand_threshold, solve_approx_lp_deterministic_05_threshold, solve_approx_lp_randomized
from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from checkmate.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn, solve_chen_greedy
from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.tf2_keras.extraction import dfgraph_from_keras

# ILP solve params
NUM_ILP_CORES = os.environ.get("ILP_CORES", 12 if os.cpu_count() > 12 else 4)

# Budget selection parameters
NUM_ILP_GLOBAL = 32
NUM_ILP_LOCAL = 32
ILP_SEARCH_RANGE = [0.5, 1.5]
ILP_ROUND_FACTOR = 1000  # 1KB
PLOT_UNIT_RAM = 1e9  # 9 = GB
DENSE_SOLVE_MODELS = ["VGG16", "VGG19"]
ADDITIONAL_ILP_LOCAL_POINTS = {  # additional budgets to add to ILP local search.
    # measured in GB, including fixed parameters.
    ("ResNet50", 256): [9.25, 9.5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    ("MobileNet", 512): [15.9, 15, 14, 17, 18, 19, 38, 37, 36, 39],
    ("vgg_unet", 32): [20, 21, 22, 23, 24, 25, 26, 6, 7, 8, 9, 10, 11, 12, 16, 15.9]
}

# Plotting parameters
XLIM = {
    ("ResNet50", 256): [8, 42],
    ("MobileNet", 512): [4, 48],
    ("vgg_unet", 32): [4, 32],
    ("VGG16", 256): [13, 22]
}

YLIM = {
    ("ResNet50", 256): [0.95, 1.5],
    ("MobileNet", 512): [0.95, 1.5],
    ("vgg_unet", 32): [0.95, 1.5],
    ("VGG16", 256): [0.95, 1.5]
}


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument('--model-name', default="VGG16", choices=list(sorted(MODEL_NAMES)))
    parser.add_argument('--ilp-eval-points', nargs='+', type=int, default=[],
                        help="If set, will only search a specific set of ILP points in MB, else perform global search.")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-s", "--input-shape", type=int, nargs="+", default=[])

    parser.add_argument('--debug', action='store_true', help="If set, write debug files like model files.")
    parser.add_argument('--exact-ilp-solve', action='store_true', help="If set, disable approx in ILP solve.")
    parser.add_argument('--skip-ilp', action='store_true', help="If set, skip running the ILP during evaluation.")
    parser.add_argument('--ilp-time-limit', type=int, default=3600, help="Time limit for individual ILP solves, in sec")
    parser.add_argument('--hide-points', action="store_true")

    _args = parser.parse_args()
    if _args.skip_ilp and len(_args.ilp_eval_points) > 0:
        parser.error("--skip-ilp and --ilp-eval-points cannot both be set")
    _args.input_shape = _args.input_shape if _args.input_shape else None
    return _args


def get_closest_budget_result(results: Dict[SolveStrategy, List[ScheduledResult]], b: int) -> Optional[ScheduledResult]:
    clean_results = [r for l in results.values() for r in l if r is not None and r.schedule_aux_data is not None]
    under_budget = [r for r in clean_results if r.schedule_aux_data.activation_ram <= b]
    return min(under_budget, key=lambda r: r.schedule_aux_data.cpu, default=None)


def prefix_min_np(values: np.ndarray):
    assert values.ndim == 1
    values_min = np.copy(values)
    for i in range(1, values.shape[0]):
        values_min[i] = min(values_min[i - 1], values[i])
    return values_min


def roundup(round_factor: int, number: float) -> int:
    """helper function to round up a number by a factor"""
    return int(np.ceil(float(number) / round_factor) * round_factor)


def dist_points(start, stop, n, min_val=0.0):
    assert start < stop, "Start range must be below end of range"
    assert start > 0 and stop > 0, "Start and stop ranges must be positive"
    pts = sorted(start + np.arange(0, 1, 1. / n) * (stop - start))
    return [p for p in pts if p > min_val]


def get_global_eval_points(g: DFGraph, results: Dict[SolveStrategy, List[ScheduledResult]]) -> List[int]:
    """Global point generation strategy:
     * k = number of samples (depends on DENSE_SOLVE_MODELS
     * min point = min feasible RAM (in + out for any node)
     * max point = max RAM that achieves no overhead (ie CHECKPOINT_ALL)
     * dist_range will select k points [min, max]
     * round points to quantize to even rounding factor
    :param g: DFGraph
    :param results: Dict[SolveStrategy, List[ScheduledResult]]
    :return: list of integral evaluation points
    """
    number_samples = NUM_ILP_GLOBAL * 2 if model_name in DENSE_SOLVE_MODELS else NUM_ILP_GLOBAL
    min_ram = g.max_degree_ram

    # get max point by finding closest matching greedy result
    check_all_result = results[SolveStrategy.CHECKPOINT_ALL][0]
    check_all_sched = check_all_result.schedule_aux_data
    max_greedy_result = check_all_result
    all_greedy = results.get(SolveStrategy.CHEN_GREEDY, []) + results.get(SolveStrategy.CHEN_GREEDY_NOAP, [])
    for greedy_sol in all_greedy:
        greedy_sched = greedy_sol.schedule_aux_data
        temp_max_sched = max_greedy_result.schedule_aux_data
        if greedy_sched.cpu <= check_all_sched.cpu and greedy_sched.activation_ram < temp_max_sched.activation_ram:
            max_greedy_result = greedy_sol
    max_ram = max_greedy_result.schedule_aux_data.activation_ram

    # sample k points and round
    rounded_eval_points = [roundup(ILP_ROUND_FACTOR, b) for b in (dist_points(min_ram, max_ram, number_samples))]

    # add checkpoint all result to list of global points to force plot to reach zero overhead
    rounded_eval_points.append(roundup(ILP_ROUND_FACTOR, check_all_sched.peak_ram))
    rounded_eval_points.append(roundup(ILP_ROUND_FACTOR, check_all_sched.peak_ram - g.cost_ram_fixed))

    return rounded_eval_points


if __name__ == "__main__":
    logger = logging.getLogger("budget_sweep")
    logger.setLevel(logging.DEBUG)
    # due to bug on havoc, limit parallelism on high-core machines
    if os.cpu_count() > 48:
        os.environ["OMP_NUM_THREADS"] = "1"
    args = extract_params()

    ray.init(temp_dir="/tmp/ray_checkpoint", redis_password=str(uuid.uuid1()), num_cpus=os.cpu_count(),
             object_store_memory=1024 * 1024 * 1024 if os.cpu_count() < 48 else None)  # include_webui=args.debug

    key = "_".join(map(str, [args.platform, args.model_name, args.batch_size, args.input_shape]))
    log_base = remat_data_dir() / "budget_sweep" / key
    shutil.rmtree(log_base, ignore_errors=True)
    pathlib.Path(log_base).mkdir(parents=True, exist_ok=True)

    ####
    # Begin budget_sweep data collection
    ####
    result_dict: Dict[SolveStrategy, List[ScheduledResult]] = {}
    model_name = args.model_name

    # load costs, and plot optionally, if platform is not flops
    logger.info(f"Loading costs")
    if args.platform == "flops":
        cost_model = None
    else:
        cost_model = CostModel(model_name, args.platform, log_base, quantization=5)
        cost_model.fit()
        if args.debug:
            cost_model.plot_costs()

    # gen redis key
    if cost_model is None:
        key_list = ["flops", args.batch_size]
    else:
        key_list = [cost_model.platform, cost_model.quantization, args.batch_size]
    redis_cost_key = "_".join(map(str, key_list))

    # load model from Keras
    logger.info(f"Loading model {model_name}")
    model = get_keras_model(model_name, input_shape=args.input_shape)
    g = dfgraph_from_keras(model, batch_size=args.batch_size, cost_model=cost_model,
                           loss_cpu_cost=0, loss_ram_cost=(4 * args.batch_size))
    if args.debug:
        tf.keras.utils.plot_model(model,
                                  to_file=log_base / f"plot_{model_name}_keras.png",
                                  show_shapes=True,
                                  show_layer_names=True)
        plot_dfgraph(g, log_base, name=model_name)

    # sweep constant baselines
    logger.info(f"Running constant baselines (ALL, ALL_AP, LAST_NODE, SQRTN_NOAP, SQRTN)")
    result_dict[SolveStrategy.CHECKPOINT_ALL] = [solve_checkpoint_all(g)]
    result_dict[SolveStrategy.CHECKPOINT_ALL_AP] = [solve_checkpoint_all_ap(g)]
    result_dict[SolveStrategy.CHECKPOINT_LAST_NODE] = [solve_checkpoint_last_node(g)]
    result_dict[SolveStrategy.CHEN_SQRTN_NOAP] = [solve_chen_sqrtn(g, False)]
    result_dict[SolveStrategy.CHEN_SQRTN] = [solve_chen_sqrtn(g, True)]

    # sweep chen's greedy baseline
    logger.info(f"Running Chen's greedy baseline (APs only)")
    chen_sqrtn_noap = result_dict[SolveStrategy.CHEN_SQRTN_NOAP][0]
    greedy_eval_points = chen_sqrtn_noap.schedule_aux_data.activation_ram * (1. + np.arange(-1, 2, 0.01))
    remote_solve_chen_greedy = ray.remote(num_cpus=1)(solve_chen_greedy).remote
    futures = [remote_solve_chen_greedy(g, float(b), False) for b in greedy_eval_points]
    result_dict[SolveStrategy.CHEN_GREEDY] = get_futures(list(futures), desc="Greedy (APs only)")
    if model_name not in CHAIN_GRAPH_MODELS:
        logger.info(f"Running Chen's greedy baseline (no AP) as model is non-linear")
        futures = [remote_solve_chen_greedy(g, float(b), True) for b in greedy_eval_points]
        result_dict[SolveStrategy.CHEN_SQRTN_NOAP] = get_futures(list(futures), desc="Greedy (No AP)")

    # sweep griewank baselines
    logger.error("Skipping Griewank baselines as it was broken in parasj/checkmate#65")
    # if model_name in CHAIN_GRAPH_MODELS:
    #     logger.info(f"Running Griewank baseline (APs only)")
    #     clean_griewank_cache()
    #     solve_griewank(g, 1)  # prefetch griewank solution from s3, otherwise ray will cause race condition
    #     griewank_eval_points = range(1, g.size + 1)
    #     remote_solve_griewank = ray.remote(num_cpus=1)(solve_griewank).remote
    #     futures = [remote_solve_griewank(g, float(b)) for b in griewank_eval_points]
    #     result_dict[SolveStrategy.GRIEWANK_LOGN] = get_futures(list(futures), desc="Griewank (APs only)")

    # sweep optimal ilp baseline
    if not args.skip_ilp:
        ilp_log_base = log_base / "ilp_log"
        ilp_log_base.mkdir(parents=True, exist_ok=True)
        # todo load any ILP results from cache
        remote_ilp = ray.remote(num_cpus=NUM_ILP_CORES)(solve_ilp_gurobi).remote
        if len(args.ilp_eval_points) > 0:
            local_ilp_eval_points = [p * 1000 * 1000 - g.cost_ram_fixed for p in args.ilp_eval_points]
        else:
            # run global search routine
            global_eval_points = get_global_eval_points(g, result_dict)
            logger.info(f"Evaluating ILP at global evaluation points: {global_eval_points}")
            futures = []
            for b in global_eval_points:
                seed_result = get_closest_budget_result(result_dict, b)
                seed_s = seed_result.schedule_aux_data.S if seed_result is not None else None
                future = remote_ilp(g, b, time_limit=args.ilp_time_limit, solver_cores=NUM_ILP_CORES, seed_s=seed_s,
                                    write_log_file=ilp_log_base / f"ilp_{b}.log", print_to_console=False,
                                    write_model_file=ilp_log_base / f"ilp_{b}.lp" if args.debug else None,
                                    eps_noise=0 if args.exact_ilp_solve else 0.01, approx=args.exact_ilp_solve)
                futures.append(future)
            result_dict[SolveStrategy.OPTIMAL_ILP_GC] = get_futures(futures, desc="Global optimal ILP sweep")

            # sample n local points around minimum feasible ram (all methods)
            min_r = min([r.schedule_aux_data.activation_ram or np.inf for l in result_dict.values() for r in l if
                         r is not None and r.schedule_aux_data is not None])
            logger.debug(f"Minimum feasible ILP solution at {min_r}")
            nlocal_samples = NUM_ILP_LOCAL * 2 if model_name in DENSE_SOLVE_MODELS else NUM_ILP_LOCAL
            min_ram = ILP_SEARCH_RANGE[0] * min_r
            max_ram = ILP_SEARCH_RANGE[1] * min_r
            k_pts = dist_points(min_ram, max_ram, nlocal_samples)
            local_ilp_eval_points = [roundup(ILP_ROUND_FACTOR, p) for p in k_pts]

        # run local search routine
        futures = []
        for b in local_ilp_eval_points:
            seed_result = get_closest_budget_result(result_dict, b)
            seed_s = seed_result.schedule_aux_data.S if seed_result is not None else None
            future = remote_ilp(g, b, time_limit=args.ilp_time_limit, solver_cores=NUM_ILP_CORES, seed_s=seed_s,
                                write_log_file=ilp_log_base / f"ilp_{b}.log", print_to_console=False,
                                write_model_file=ilp_log_base / f"ilp_{b}.lp" if args.debug else None,
                                eps_noise=0 if args.exact_ilp_solve else 0.01, approx=args.exact_ilp_solve)
            futures.append(future)
        result_dict[SolveStrategy.OPTIMAL_ILP_GC].extend(get_futures(futures, desc="Local optimal ILP sweep"))

    approx_eval_points = list(get_global_eval_points(g, result_dict))
    if not args.skip_ilp:
        approx_eval_points.extend(list(local_ilp_eval_points))
    logger.info(f"Evaluating LP approximation evaluation points: {approx_eval_points}")

    # sweep LP rounding (deterministic w/ 0.5 threshold)
    lpdet05_log_base = log_base / "lp_det_05"
    lpdet05_log_base.mkdir(parents=True, exist_ok=True)
    remote_lp_det_05 = ray.remote(num_cpus=NUM_ILP_CORES)(solve_approx_lp_deterministic_05_threshold).remote
    futures = []
    for b in approx_eval_points:
        future = remote_lp_det_05(g, b, time_limit=args.ilp_time_limit, solver_cores=NUM_ILP_CORES,
                                  write_log_file=lpdet05_log_base / f"lp_det_rand_{b}.log", print_to_console=False,
                                  write_model_file=lpdet05_log_base / f"lp_det_rand_{b}.lp" if args.debug else None,
                                  eps_noise=0, approx=False)
        futures.append(future)
    result_dict[SolveStrategy.APPROX_DET_ROUND_LP_05_THRESH] = get_futures(futures,
                                                                           desc="LP approx det 0.5")

    # sweep LP rounding (deterministic w/ randomly sampled threshold)
    lpdetrand_log_base = log_base / "lp_det_rand"
    lpdetrand_log_base.mkdir(parents=True, exist_ok=True)
    remote_lp_det_rand = ray.remote(num_cpus=NUM_ILP_CORES)(solve_approx_lp_deterministic_rand_threshold).remote
    futures = []
    for b in approx_eval_points:
        future = remote_lp_det_rand(g, b, time_limit=args.ilp_time_limit, solver_cores=NUM_ILP_CORES,
                                    write_log_file=lpdetrand_log_base / f"lp_det_05_{b}.log", print_to_console=False,
                                    write_model_file=lpdetrand_log_base / f"lp_det_05_{b}.lp" if args.debug else None,
                                    eps_noise=0, approx=False)
        futures.append(future)
    result_dict[SolveStrategy.APPROX_DET_RANDOM_THRESH_ROUND_LP] = get_futures(futures,
                                                                               desc="LP approx det rand")

    # sweep LP rounding (deterministic sweep)
    lpdet_log_base = log_base / "lp_det_sweep"
    lpdet_log_base.mkdir(parents=True, exist_ok=True)
    remote_lp_det_sweep = ray.remote(num_cpus=NUM_ILP_CORES)(solve_approx_lp_deterministic_sweep).remote
    futures = []
    for b in approx_eval_points:
        future = remote_lp_det_sweep(g, b, time_limit=args.ilp_time_limit, solver_cores=NUM_ILP_CORES,
                                     write_log_file=lpdet_log_base / f"lp_det_{b}.log", print_to_console=False,
                                     write_model_file=lpdet_log_base / f"lp_det_{b}.lp" if args.debug else None,
                                     eps_noise=0, approx=False)
        futures.append(future)
    result_dict[SolveStrategy.APPROX_DET_ROUND_LP_SWEEP] = get_futures(futures, desc="LP approx det sweep")

    # sweep LP randomized rounding
    lprand_log_base = log_base / "lp_rand"
    lprand_log_base.mkdir(parents=True, exist_ok=True)
    remote_lp_rand = ray.remote(num_cpus=NUM_ILP_CORES)(solve_approx_lp_randomized).remote
    futures = []
    for b in approx_eval_points:
        future = remote_lp_rand(g, b, time_limit=args.ilp_time_limit, solver_cores=NUM_ILP_CORES,
                                write_log_file=lprand_log_base / f"lp_rand_{b}.log", print_to_console=False,
                                write_model_file=lprand_log_base / f"lp_rand_{b}.lp" if args.debug else None,
                                eps_noise=0, approx=False)
        futures.append(future)
    result_dict[SolveStrategy.APPROX_RANDOMIZED_ROUND] = get_futures(futures, desc="LP approx rand")

    ####
    # Plot result_dict
    ####
    # todo save pandas results dict
    sns.set()
    sns.set_style("white")

    baseline_cpu = np.sum(list(g.cost_cpu.values()))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_xlabel("Memory budget (GB)")
    ax.set_ylabel("Overhead (x)")
    xmax = max(
        [r.schedule_aux_data.peak_ram or 0 for rlist in result_dict.values() for r in rlist if
         r is not None and r.schedule_aux_data is not None])
    logger.info(f"xmax value = {xmax}")
    legend_elements = []

    export_prefix_min = {}

    for solve_strategy, results in result_dict.items():
        # checkpoint last node has too high compute, checkpoint all is plotted later
        if solve_strategy in [SolveStrategy.CHECKPOINT_LAST_NODE, SolveStrategy.CHECKPOINT_ALL]:  continue

        label = SolveStrategy.get_description(solve_strategy, model_name=model_name)
        color, marker, markersize = SolveStrategy.get_plot_params(solve_strategy)

        # Scatter candidate solutions
        valid_data = [r.schedule_aux_data for r in results if r is not None and r.schedule_aux_data is not None]
        sorted_data = sorted(valid_data, key=lambda r: r.peak_ram)
        data_points = [(t.peak_ram / PLOT_UNIT_RAM, t.cpu * 1.0 / baseline_cpu) for t in sorted_data]
        logger.info(f"Strategy {solve_strategy} has {len(data_points)} samples from {len(results)}")
        if not len(data_points):
            continue

        x, y = map(list, zip(*data_points))
        x_step = x + [xmax * 1.0 / PLOT_UNIT_RAM]
        y_step = prefix_min_np(np.array(y + [min(y)]))

        # Plot best solution over budgets <= x
        # Add a point to the right of the plot, so ax.step can draw a horizontal line
        ax.step(x_step, y_step, where='post', zorder=1, color=color)
        scatter_zorder = 3 if solve_strategy == SolveStrategy.CHECKPOINT_ALL_AP else 2
        if args.hide_points:
            # Plot only the first and last points
            ax.scatter([x[0], x[-1]], [y[0], y[-1]], label="", zorder=scatter_zorder, s=markersize ** 2,
                       color=color, marker=marker)
        else:
            ax.scatter(x, np.array(y), label="", zorder=scatter_zorder, s=markersize ** 2, color=color,
                       marker=marker)
        legend_elements.append(Line2D([0], [0], lw=2, label=label, markersize=markersize, color=color, marker=marker))

        export_prefix_min[solve_strategy.name] = list(zip(x_step, y_step))

    # Plot ideal (checkpoint all)
    xlim_min, xlim_max = ax.get_xlim()
    checkpoint_all_result = result_dict[SolveStrategy.CHECKPOINT_ALL][0].schedule_aux_data
    x = checkpoint_all_result.peak_ram / PLOT_UNIT_RAM
    y = checkpoint_all_result.cpu / baseline_cpu
    color, marker, markersize = SolveStrategy.get_plot_params(SolveStrategy.CHECKPOINT_ALL)
    label = SolveStrategy.get_description(SolveStrategy.CHECKPOINT_ALL, model_name=model_name)
    xlim_max = max(x, xlim_max)
    ax.scatter([x], [y], label="", zorder=2, color=color, marker=marker, s=markersize ** 2)
    ax.hlines(y=y, xmin=xlim_min, xmax=x, linestyles="dashed", color=color)
    ax.hlines(y=y, xmin=x, xmax=xlim_max, color=color, zorder=2)
    legend_elements.append(Line2D([0], [0], lw=2, label=label, color=color, marker=marker, markersize=markersize))
    ax.set_xlim([xlim_min, xlim_max])

    # Plot platform memory
    ylim_min, ylim_max = ax.get_ylim()
    mem_gb = platform_memory(args.platform) / 1e9
    if xlim_min <= mem_gb <= xlim_max:
        ax.vlines(x=mem_gb, ymin=ylim_min, ymax=ylim_max, linestyles="dotted", color="b")
        legend_elements.append(
            Line2D([0], [0], lw=2, label=f"{pretty_platform_name(args.platform)} memory", color="b",
                   linestyle="dotted"))
        ax.set_ylim([ylim_min, ylim_max])

    if (model_name, args.batch_size) in XLIM:
        ax.set_xlim(XLIM[(model_name, args.batch_size)])

    if (model_name, args.batch_size) in YLIM:
        ax.set_ylim(YLIM[(model_name, args.batch_size)])

    # Make legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.2),
              fancybox=False, shadow=False, ncol=2)

    fig.savefig(log_base / f"plot_budget_sweep_{model_name}_{args.platform}_b{args.batch_size}.pdf",
                format='pdf', bbox_inches='tight')
    fig.savefig(log_base / f"plot_budget_sweep_{model_name}_{args.platform}_b{args.batch_size}.png",
                bbox_inches='tight', dpi=300)

    # export list of budget, CPU tuples for each strategy
    pickle.dump(export_prefix_min, (log_base / f"export_prefix_min_data.pickle").open('wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    if not args.skip_ilp:
        optimal_ilp_budgets = [x[0] for x in export_prefix_min['OPTIMAL_ILP_GC']]
        optimal_ilp_cpu = [x[1] for x in export_prefix_min['OPTIMAL_ILP_GC']]

        slowdowns = defaultdict(list)
        for key in [x for x in export_prefix_min.keys() if x != 'OPTIMAL_ILP_GC']:
            for budget, optimal_cpu in export_prefix_min['OPTIMAL_ILP_GC']:
                filtered_budgets = [x for x in export_prefix_min[key] if x[0] <= budget]
                if len(filtered_budgets) == 0:
                    continue
                min_budget, min_cpu = min(filtered_budgets, key=lambda x: x[1])
                slowdown = min_cpu / optimal_cpu
                slowdowns[key].append(slowdown)

        df_data = []
        for key, slowdown_list in slowdowns.items():
            max_slowdown = max(slowdown_list)
            gmean_slowdown = gmean(slowdown_list)
            df_data.append({'method': key, 'max': max_slowdown, 'geomean_slowdown': gmean_slowdown})
        df = pd.DataFrame(df_data)
        df.to_csv(log_base / "slowdowns.csv")
