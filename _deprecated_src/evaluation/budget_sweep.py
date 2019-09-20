from __future__ import division

import functools
import os
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import ray
import seaborn as sns
from matplotlib.lines import Line2D
from tqdm import tqdm

from evaluation.util.cost_model import CostModel
from evaluation.util.evaluation_utils import prefix_min_np, result_dict_to_dataframe, RSResultDict, get_futures
from remat.core.solvers.enum_strategy import SolveStrategy
from evaluation.util.solver_utils import remote_evaluation_iteration
from integration.tf2.TF2ExtractorParams import TF2ExtractorParams
from experiments.common.keras_extractor import CHAIN_GRAPH_MODELS, pretty_platform_name, platform_memory, get_keras_model
from solvers.result import RSResult
from utils.redis import RedisCache
from utils.setup_logger import setup_logger

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



def result_dict_to_cache_list(results_dict: RSResultDict) -> List[RSResult]:
    return [r for rlist in results_dict.values() for r in rlist if r is not None]


def num_ilp_global(model_name):
    if model_name in DENSE_SOLVE_MODELS:
        return NUM_ILP_GLOBAL * 2
    return NUM_ILP_GLOBAL


def num_ilp_local(model_name):
    if model_name in DENSE_SOLVE_MODELS:
        return NUM_ILP_LOCAL * 2
    return NUM_ILP_LOCAL


def distribute_k_points(start, stop, n, min_val=0.0):
    assert start < stop, "Start range must be below end of range"
    assert start > 0 and stop > 0, "Start and stop ranges must be positive"
    pts = sorted(start + np.arange(0, 1, 1. / n) * (stop - start))
    return [p for p in pts if p > min_val]


def round_up_by_factor(k: int, number: float):
    return int(np.ceil(float(number) / k) * k)


def eval_budget_sweep(args, log_base, key_prefix, cost_file=None):
    model_name = args.model_name
    logger = setup_logger("eval_budget_sweep", os.path.join(log_base, 'eval_budget_sweep.log'))

    logger.debug(f"Loading model {model_name}")
    model = get_keras_model(model_name, input_shape=args.input_shape)
    loss_cpu_cost = 0  # todo: use real loss cpu runtime cost
    loss_ram_cost = 4 * args.batch_size  # 1 scalar per batch

    # plot costs
    if args.platform != "flops":
        cost_model = CostModel(model_name, args.platform, log_base, quantization=5)
        cost_model.fit()
        cost_model.plot_costs()
        costs_np = cost_model.get_costs(args.batch_size)
    else:
        costs_np = None

    # setup remote solver
    cache = RedisCache(key_prefix=key_prefix)
    tf2_params = TF2ExtractorParams(model, batch_size=args.batch_size, log_base=log_base,
                                    loss_cpu_cost=loss_cpu_cost, loss_ram_cost=loss_ram_cost, costs_np=costs_np)
    eval_result = functools.partial(remote_evaluation_iteration, cache, tf2_params, cost_file=cost_file)
    result_dict: Dict[SolveStrategy, List[RSResult]] = {}

    # evaluate constant baselines
    with ray.profile("budget_sweep constant baselines"):
        baseline_futures = filter(lambda x: x is not None, [
            eval_result(SolveStrategy.CHECKPOINT_ALL, plot=False, skip_cache=True),
            eval_result(SolveStrategy.CHECKPOINT_ALL_AP, plot=False,
                        skip_cache=True) if model_name not in CHAIN_GRAPH_MODELS else None,
            eval_result(SolveStrategy.CHECKPOINT_LAST_NODE, plot=False, skip_cache=True),
            eval_result(SolveStrategy.CHEN_SQRTN, plot=False,
                        skip_cache=True) if model_name not in CHAIN_GRAPH_MODELS else None,
            eval_result(SolveStrategy.CHEN_SQRTN_NOAP, plot=False, skip_cache=True)])
        constant_baselines = ray.get(list(baseline_futures))
        for check in constant_baselines:
            logger.info(f"{check.solve_strategy} = {(check.peak_ram / 1e6):.3f}mb with {check.cpu:.2E} runtime")
            # RSResult.plot(tf2_params.g, check.R, check.S, plot_mem_usage=True, plt=plt,
            #               save_file=os.path.join(log_base, f"plot_RS_f{check.solve_strategy}.pdf"))
            # RSResult.plot(tf2_params.g, check.R, check.S, plot_mem_usage=True, plt=plt,
            #               save_file=os.path.join(log_base, f"plot_RS_f{check.solve_strategy}.png"))
            result_dict[check.solve_strategy] = [check]

    # evaluate greedy sweeps
    with ray.profile("budget_sweep greedy"):
        greedy_strategies = [(SolveStrategy.CHEN_GREEDY_NOAP, SolveStrategy.CHEN_SQRTN_NOAP)]
        if model_name not in CHAIN_GRAPH_MODELS:
            greedy_strategies.append((SolveStrategy.CHEN_GREEDY, SolveStrategy.CHEN_SQRTN))
        for strategy, eval_point_source in greedy_strategies:
            result_dict[strategy] = []
            eval_points = result_dict[eval_point_source][0].activation_ram * (1. + np.arange(-1, 2, 0.01))
            pending_futures = [eval_result(strategy, float(b), plot=False, skip_cache=True) for b in
                               map(int, eval_points)]
            logger.debug(f"Launched {len(pending_futures)} tasks for greedy strategy {strategy.name}")
            results = get_futures(pending_futures, desc=strategy.name)
            result_dict[strategy].extend(results)

    # evaluate griewank sweep
    with ray.profile("budget_sweep griewank"):
        result_dict[SolveStrategy.GRIEWANK_LOGN] = []
        pending_futures = [eval_result(SolveStrategy.GRIEWANK_LOGN, float(b), plot=True, skip_cache=True) for b in
                           range(1, tf2_params.g.size + 1)]
        logger.debug(f"Launched {len(pending_futures)} tasks for griewank solution")
        results = get_futures(pending_futures, desc="GRIEWANK_LOGN")
        result_dict[SolveStrategy.GRIEWANK_LOGN].extend(results)

    # ILP evaluation template
    # We will search locally from minimum viable memory to the max ram budget (e.g. striding by a factor of 16)
    # From the minimum viable point, we will search a collection of points
    result_dict[SolveStrategy.OPTIMAL_ILP_GC] = []

    def eval_ilp(eval_budgets: List[int], cached_results: List[RSResult], log_base=".", pbar_prexix=""):
        """evaluate ilp at a range of points. helper method to speed up search."""
        cores = int(os.environ.get('ILP_CORES', 11))
        ilp_params = {'solver_cores': cores, 'time_limit': args.ilp_time_limit, 'print_to_console': False}
        tbd_futures = []
        for b in map(int, eval_budgets):
            # lookup nearest point to seed S
            # todo -- why are Nones showing up in the cached result list?
            results_under_budget = [r for r in cached_results if
                                    r is not None and (r.activation_ram or np.inf) < b * 0.95]
            best_result = min(results_under_budget, key=lambda r: r.cpu, default=None)
            best_result_s = None if best_result is None else best_result.S
            log_fname = os.path.join(log_base, f'ILP_b{b}.log')
            model_fname = os.path.join(log_base, f'ILP_b{b}.lp')
            computed_params = dict(seed_s=best_result_s, log_file=log_fname, model_file=model_fname, **ilp_params)
            eval_res = eval_result(SolveStrategy.OPTIMAL_ILP_GC, float(b), plot=False, solver_params=computed_params,
                                   skip_cache=args.overwrite)
            tbd_futures.append(eval_res)

        pbar_desc = pbar_prexix + SolveStrategy.get_description(SolveStrategy.OPTIMAL_ILP_GC, model_name=model_name)
        with tqdm(total=len(tbd_futures), desc=pbar_desc) as pbar:
            while len(tbd_futures):
                done_futures, tbd_futures = ray.wait(tbd_futures)
                for v in ray.get(done_futures):
                    logger.debug(f"ILP solved: {v.solver_budget} budget with act. mem {v.activation_ram}, "
                                 f"cpu {v.cpu}, in {v.solve_time_s}")
                    pbar.update(1)
                    yield v

    logger.debug("args.skip_ilp=%s", args.skip_ilp)
    if not args.skip_ilp:
        if not args.develop:
            logger.info("Checking cache for more ILP results (turn off with --develop)")
            cached_ilp_results, _ = cache.read_results(solver=SolveStrategy.OPTIMAL_ILP_GC, cost_file=cost_file)
            for result in cached_ilp_results:
                if result is not None and result.peak_ram is not None:
                    logger.debug("Loaded result with %.2E peak ram, %.2E compute", result.peak_ram, result.cpu)
                    result_dict[SolveStrategy.OPTIMAL_ILP_GC].append(result)

        logger.debug("Going to compute ILPs")
        if len(args.ilp_eval_points) > 0:
            logger.debug(f"Using user defined ilp_eval_points (in KB), but subtracting fixed"
                         f"ram {tf2_params.g.cost_ram_fixed:.2E} to get solver budget")
            eval_points = [p * 1000 * 1000 - tf2_params.g.cost_ram_fixed for p in args.ilp_eval_points]
        else:
            logger.debug("Searching over budgets")
            # global search
            # select top of range by finding min-memory greedy solution that the same cpu time as CHECKPOINT_ALL
            with ray.profile("budget_sweep global ILP"):
                max_result = result_dict[SolveStrategy.CHECKPOINT_ALL][0]
                max_greedy_result = max_result
                all_greedy = result_dict.get(SolveStrategy.CHEN_GREEDY, []) + \
                             result_dict.get(SolveStrategy.CHEN_GREEDY_NOAP, [])
                for greedy_sol in all_greedy:
                    if greedy_sol.cpu <= max_result.cpu and greedy_sol.activation_ram < max_greedy_result.activation_ram:
                        max_greedy_result = greedy_sol
                eval_points = distribute_k_points(tf2_params.g.max_degree_ram(), max_greedy_result.activation_ram, num_ilp_global(model_name))
                eval_points = list(map(functools.partial(round_up_by_factor, ILP_ROUND_FACTOR), eval_points))  # round
                checkpoint_all_result = result_dict.get(SolveStrategy.CHECKPOINT_ALL, [])
                if checkpoint_all_result:
                    checkpoint_all_mem = checkpoint_all_result[0].peak_ram
                    eval_points.append(round_up_by_factor(ILP_ROUND_FACTOR, checkpoint_all_mem))
                    eval_points.append(
                        round_up_by_factor(ILP_ROUND_FACTOR, checkpoint_all_mem - tf2_params.g.cost_ram_fixed))
                logger.debug(f"Evaluation points for global search: {eval_points}")
                global_results = list(eval_ilp(eval_points, result_dict_to_cache_list(result_dict), log_base=log_base,
                                               pbar_prexix="GLOBAL "))
                result_dict[SolveStrategy.OPTIMAL_ILP_GC].extend(global_results)
                logger.debug(f"Global search yielded {len(global_results)} results")

            # calculate local search range
            min_r = min([r.activation_ram or np.inf for r in result_dict_to_cache_list(result_dict)])
            logger.debug(f"Minimum feasible ILP solution at {min_r}")
            k_pts = distribute_k_points(ILP_SEARCH_RANGE[0] * min_r, ILP_SEARCH_RANGE[1] * min_r, num_ilp_local(model))
            eval_points = [round_up_by_factor(ILP_ROUND_FACTOR, p) for p in k_pts]

        # Add additional points specified above
        if (model_name, args.batch_size) in ADDITIONAL_ILP_LOCAL_POINTS:
            addl_pts = ADDITIONAL_ILP_LOCAL_POINTS[(model_name, args.batch_size)]
            addl_pts = [int(adp * 1e9 - tf2_params.g.cost_ram_fixed) for adp in addl_pts]
            eval_points = addl_pts + eval_points

        # local search
        with ray.profile("budget_sweep local ILP"):
            logger.debug(f"Evaluation points for local search: {eval_points}")
            local_results = list(
                eval_ilp(eval_points, result_dict_to_cache_list(result_dict), log_base=log_base, pbar_prexix="LOCAL "))
            result_dict[SolveStrategy.OPTIMAL_ILP_GC].extend(local_results)
            logger.debug(f"Local search yielded {len(local_results)} results")

    logger.info("Plotting results")
    with ray.profile("budget_sweep plotting"):
        df = result_dict_to_dataframe(result_dict)
        df.to_csv(os.path.join(log_base, 'solver_results.csv'))
        plot_budget_sweep(args, model_name, args.platform, result_dict, tf2_params, log_base,
                          batch_size=args.batch_size)


def plot_budget_sweep(args, model_name: str, platform: str, result_dict: Dict[SolveStrategy, List[RSResult]],
                      tf2_params: TF2ExtractorParams, log_base: str, batch_size: int):
    sns.set()
    sns.set_style("white")

    logger = setup_logger("eval_budget_sweep", os.path.join(log_base, 'eval_budget_sweep.log'))

    baseline_cpu = np.sum(list(tf2_params.g.cost_cpu.values()))
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_xlabel("Memory budget (GB)")
    ax.set_ylabel("Overhead (x)")
    xmax = max(
        [r.peak_ram for rlist in result_dict.values() for r in rlist if r is not None and r.peak_ram is not None])
    logger.info(f"xmax value = {xmax}")
    legend_elements = []

    for solve_strategy, results in result_dict.items():
        if solve_strategy in [SolveStrategy.CHECKPOINT_LAST_NODE, SolveStrategy.CHECKPOINT_ALL]:
            continue  # checkpoint last node has too high compute, checkpoint all is plotted later

        label = SolveStrategy.get_description(solve_strategy, model_name=model_name)
        color, marker, markersize = SolveStrategy.get_plot_params(solve_strategy)

        # Scatter candidate solutions
        valid_data = [r for r in results if r is not None and r.peak_ram is not None and r.cpu is not None]
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

    # Plot ideal (checkpoint all)
    xlim_min, xlim_max = ax.get_xlim()
    checkpoint_all_result = result_dict[SolveStrategy.CHECKPOINT_ALL][0]
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
    mem_gb = platform_memory(platform) / 1e9
    if xlim_min <= mem_gb <= xlim_max:
        ax.vlines(x=mem_gb, ymin=ylim_min, ymax=ylim_max, linestyles="dotted", color="b")
        legend_elements.append(
            Line2D([0], [0], lw=2, label=f"{pretty_platform_name(platform)} memory", color="b", linestyle="dotted"))
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

    fig.savefig(os.path.join(log_base, f"!budget_sweep_{model_name}_{platform}_b{batch_size}.pdf"),
                format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(log_base, f"!budget_sweep_{model_name}_{platform}_b{batch_size}.png"),
                bbox_inches='tight', dpi=300)

