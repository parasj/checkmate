import os
from typing import Optional, List, Tuple
from tqdm import tqdm

import pandas
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np

from remat.core.solvers.strategy_enum import SolveStrategy
from integration.tf2.TF2ExtractorParams import TF2ExtractorParams
from integration.tf2.extraction import get_keras_model, platform_memory
from integration.tf2.TF2Runner import TF2Runner
from integration.tf2.misc import categorical_cross_entropy, random_batch
from solvers.result import RSResult
from utils.redis import RedisCache
from utils.setup_logger import setup_logger
from remat.core.utils.timer import Timer


def plot_solver_result(results: pandas.DataFrame, plot_file: str):
    import seaborn as sns
    sns.set()
    data = results.sort_values('cpu')
    data = data[data['solve_strategy'].isin([SolveStrategy.CHECKPOINT_ALL])]
    data['solve_strategy'] = data['solve_strategy'].apply(SolveStrategy.get_description)
    compare_plot = sns.barplot(data=data, x='cpu', y='solve_strategy', hue='peak_ram')
    compare_plot.figure.savefig(plot_file)


def get_param(strategy: SolveStrategy, log_base=""):
    base_params = {}
    if strategy == SolveStrategy.OPTIMAL_ILP_GC:
        return {'solver_cores': os.cpu_count() - 1, 'time_limit': None, 'print_to_console': False,
                'log_file': os.path.join(log_base, "optimal_ilp_gc.log"),
                'model_file': os.path.join(log_base, "optimal_ilp_gc.lp")}
    return base_params


def evaluate_solved_model(result: RSResult, runner: TF2Runner, warmup, trials, batch_size):
    logger = setup_logger("evaluate_solved_model")
    
    recompute_baseline = runner.tf_graph

    logger.debug("Warming up models")
    in_shape = runner.keras_model.input_shape
    out_shape = runner.keras_model.output_shape


    h = in_shape[1]
    w = in_shape[2]
    c = np.prod(out_shape[1:])
    reshape_to = list(out_shape)
    print(reshape_to)
    reshape_to[0] = -1
    for data in tqdm([random_batch(batch_size, img_h=h, img_w=w,num_classes=c ) for _ in range(warmup)], desc="Warmup"):
        dat, lab = data
        lab = tf.reshape(lab, reshape_to)
        recompute_baseline(dat, lab)

    # run actual evaluation
    timer = Timer("timer_recompute")
    for i in tqdm(range(trials), desc="Profiling"):
        # TODO: Should we generate random batches on CPU and copy to GPU, inside the timing loop?
        #       This would model the overhead of loading data, and bring throughputs down to be
        #       more realistic

        images, labels = random_batch(batch_size, img_h=h, img_w=w, num_classes=c)
        labels =  tf.reshape(labels, reshape_to)
        with timer:
            loss , gradients= recompute_baseline(images, labels)

        # todo assert correctness of the model by applying gradients

    tput = trials / timer.elapsed
    logger.info(f"{result.solve_strategy} throughput: {tput :2.4} iters/s")
    return tput


EAGER = False

def execute_one(log_base: str, solve_strategy: SolveStrategy, model_name: str, batch_size: int,
        platform: str, input_shape=None, model_version="v1", num_runs=16, buffer_mem: int=0) -> Tuple[Optional[RSResult], str, int]:
    logger = setup_logger("eval_one")
    results_and_keys = get_solutions_to_evaluate(solve_strategy, model_name, batch_size, platform, input_shape,
                                                 model_version, buffer_mem)
    if not results_and_keys:
        logger.info("No results found")
        return None, "", 0

    if not EAGER:
        tf1.disable_eager_execution()
    for result, result_key in results_and_keys:
        tf.keras.backend.clear_session()
        model = get_keras_model(model_name, input_shape=input_shape)
        tf2 = TF2ExtractorParams(model, batch_size=batch_size, log_base=log_base)
        loss_fn = categorical_cross_entropy  # TODO: vgg_unet may need a different loss
        graph = tf2.g

        # TODO TEST THIS VS TENSORSPEC
        runner = TF2Runner(model, graph, result.schedule,
                           loss_fn=loss_fn,
                           eager=EAGER,
                           log_base=log_base,
                           batch_size=batch_size)

        try:
            throughput = evaluate_solved_model(
                result=result,
                runner=runner,
                warmup=10 if EAGER else 64,
                trials=num_runs,
                batch_size=batch_size)
            logger.info(f"Successfully executed model with predicted memory usage {result.peak_ram}, "
                        f"predicted cpu {result.cpu}, actual throughput {throughput}")
            return result, result_key, throughput
        except Exception as e:
            logger.error("Error running model with predicted mem usage %s: %s", result.peak_ram, e)
            logger.error("Traceback: %s", e.__traceback__)
            logger.error("Skipping result, going to next candidate.")
    return None, "", 0


def get_solutions_to_evaluate(solve_strategy: SolveStrategy, model_name: str, batch_size: int,
        platform: str, input_shape=None, model_version="v1", buffer_mem: int=0) -> List[Tuple[RSResult, str]]:
    """

    :param solve_strategy:
    :param model_name:
    :param batch_size:
    :param platform:
    :param input_shape:
    :param model_version:
    :return: Instance of RSResult, or None. Returns None if the solution is not available in cache
    or no solution is available under the budget
    """
    logger = setup_logger("test_execution_get_solution")

    # Load all results for this configuration, regardless of budget
    key_prefix = RedisCache.make_key(
        platform=platform,
        model_name=model_name,
        model_version=model_version,
        batch_size=batch_size,
        input_shape=input_shape)
    cache = RedisCache(key_prefix=key_prefix)
    cost_file = f"b{batch_size}_{platform}.npy"
    logger.info(f"Querying results for SS={solve_strategy}, model_name=f{model_name}, bs=f{batch_size}, "
                f"platform={platform}, cost_file={cost_file}, key prefix={key_prefix}")
    results, keys = cache.read_results(solver=solve_strategy, cost_file=cost_file, model_name=model_name)
    if not results:
        logger.error(f"No solutions found in cache for SS={solve_strategy}, model_name=f{model_name}, "
                     f"bs=f{batch_size}, platform={platform}, cost_file={cost_file}, key prefix={key_prefix}")
        return []

    # Filter results to those that abide by the budget
    platform_budget = platform_memory(platform)
    within_budget = []
    for result, key in zip(results, keys):
        if not result.peak_ram:
            logger.warn(f"Falsey peak ram? {result.peak_ram}")
            continue
        if result.peak_ram + buffer_mem <= platform_budget:
            within_budget.append((result, key))
    logger.info(f"Out of {len(results)} solver results, {len(within_budget)} had <= {platform_budget} - {buffer_mem} peak ram")
    if not within_budget:
        logger.warn(f"While {len(results)} solutions were found in cache, no solutions are within budget")
        return []

    # Return solution in increasing order of cost
    within_budget.sort(key=lambda r: r[0].cpu)
    return within_budget

    # Return min compute solution
    min_compute = within_budget[0]
    for result in within_budget:
        if result[0].cpu < min_compute[0].cpu:
            min_compute = result
    logger.info(f"Using solution with f{min_compute[0].cpu} compute, f{min_compute[0].peak_ram} ram")
    return min_compute

