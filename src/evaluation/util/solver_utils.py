import os

import graphviz
import ray

import utils.redis
from remat.core.solvers.strategy import SolveStrategy
from integration.tf2.TF2ExtractorParams import TF2ExtractorParams
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from solvers.result import PartialRSResult, RSResult
from solvers.solver import CheckpointSolver
from utils.setup_logger import setup_logger
from utils.timer import Timer

RAY_OVERPROVISON_PCT = 0.75


def tf2_solve(solve_params: TF2ExtractorParams, solve_strategy: SolveStrategy, budget: int = -1, cost_file: str = None,
              remote=False, ilp_solve_params=None, plot=False) -> PartialRSResult:
    logger = setup_logger("TF2Extractor")
    params = {}
    with Timer(f"Solving {solve_strategy}") as solve_timer:
        if solve_strategy == SolveStrategy.CHEN_SQRTN:
            sol = CheckpointSolver.schedule_sqrtn_chen16(solve_params.g, use_actuation_points=True)
        elif solve_strategy == SolveStrategy.CHEN_SQRTN_NOAP:
            sol = CheckpointSolver.schedule_sqrtn_chen16(solve_params.g, use_actuation_points=False)
        elif solve_strategy == SolveStrategy.CHECKPOINT_LAST_NODE:
            sol = solve_checkpoint_last_node(solve_params.g)
        elif solve_strategy == SolveStrategy.CHECKPOINT_ALL:
            sol = solve_checkpoint_all(solve_params.g)
        elif solve_strategy == SolveStrategy.CHECKPOINT_ALL_AP:
            sol = CheckpointSolver.schedule_checkpoint_all_ap(solve_params.g)
        elif solve_strategy == SolveStrategy.CHEN_GREEDY:
            assert budget is not None
            sol = CheckpointSolver.schedule_greedy_chen16(solve_params.g, budget, use_actuation_points=True)
        elif solve_strategy == SolveStrategy.CHEN_GREEDY_NOAP:
            assert budget is not None
            sol = CheckpointSolver.schedule_greedy_chen16(solve_params.g, budget, use_actuation_points=False)
        elif solve_strategy == SolveStrategy.OPTIMAL_ILP_GC:
            assert budget is not None
            sol, params = CheckpointSolver.schedule_ilp_gurobi(solve_params.g, budget, remote=remote,
                                                               **ilp_solve_params)
        elif solve_strategy == SolveStrategy.GRIEWANK_LOGN:
            assert budget is not None
            sol = CheckpointSolver.schedule_griewank(solve_params.g, budget)
            plot = True
        else:
            raise ValueError("No such SolveStrategy")
    params['solve_time_s'] = params['solve_time_s'] if 'solve_time_s' in params else solve_timer.elapsed
    input_shape = solve_params.input_shape
    partial_sol = PartialRSResult(solve_strategy, cost_file, input_shape, *sol, solver_budget=budget, **params)
    if all(s is not None for s in sol) and plot and solve_params.log_base is not None:
        try:
            input_shape_str = "-".join(map(str, input_shape))
            out_file = os.path.join(solve_params.log_base, f"viz_{solve_strategy}_bud{budget}_is{input_shape_str}.png")
            RSResult.plot(solve_params.g, partial_sol.R, partial_sol.S, partial_sol.U, save_file=out_file)
        except ValueError as e:
            logger.exception(e)
    return partial_sol


def tf2_schedule(solve_params: TF2ExtractorParams, solver_sol: PartialRSResult, plot=True) -> RSResult:
    with Timer("Solution scheduling") as verify_timer:
        sched_verify = RSResult.verify_solution(solve_params.g, solver_sol.R, solver_sol.S)
    result = RSResult(*solver_sol, *sched_verify, schedule_time_s=verify_timer.elapsed)
    if plot and solve_params.log_base is not None and solve_params.g.size < 100:  # due to large runtime w/ large graphs
        try:
            tensor_plot_tag = f"{solver_sol.solve_strategy}_{result.peak_ram}"
            solve_params.g.tensor_plot(result.schedule, solve_params.log_base, tag=tensor_plot_tag)
        except (FileNotFoundError, graphviz.ExecutableNotFound, graphviz.backend.CalledProcessError) as e:
            logger = setup_logger("TF2_Schedule")
            logger.exception(e)
            logger.warn("GraphViz error.")
    return result


def remote_evaluation_iteration(redis_cache: utils.redis.RedisCache, tf2_params: TF2ExtractorParams,
                                strategy: SolveStrategy, budget: int = -1, cost_file=None, plot=True,
                                skip_cache=False, solver_params={}):
    def redis_reader(_redis, args):
        return _redis.read_result(*args)

    def redis_writer(_redis, _result: RSResult):
        _redis.write_result(_result)
        return _result

    if not (skip_cache or os.environ.get("REDIS_OVERWRITE") is "1"):
        key = (strategy, budget, cost_file)
        cache_res = redis_reader(redis_cache, key)
        if cache_res is not None and cache_res.solve_strategy == strategy and cache_res.solver_budget == budget:
            # annoying, we have to re-retrieve the key so ray can recover from eviction
            # return ray.remote(num_cpus=1)(redis_reader).remote(redis_cache, key)
            return ray.put(cache_res)

    assert solver_params is not None, "solver params is None!"
    sched_cores = max(1, int(solver_params.get('solver_cores', 1) * RAY_OVERPROVISON_PCT))
    sched_fn = ray.remote(num_cpus=sched_cores)(tf2_solve).remote
    remote_sched = sched_fn(tf2_params, strategy, budget, cost_file, True, solver_params)
    remote_result = ray.remote(num_cpus=1)(tf2_schedule).remote(tf2_params, remote_sched, plot)
    remote_cache_write = ray.remote(num_cpus=1)(redis_writer).remote(redis_cache, remote_result)
    return remote_cache_write
