import logging
import math
import os
from typing import Optional

import numpy as np

from checkmate.core.dfgraph import DFGraph
from checkmate.core.enum_strategy import SolveStrategy, ImposedSchedule
from checkmate.core.schedule import ILPAuxData, ScheduledResult
from checkmate.core.solvers.strategy_optimal_ilp import ILPSolver
from checkmate.core.utils.definitions import PathLike
from checkmate.core.utils.scheduler import schedule_from_rs
from checkmate.core.utils.solver_common import solve_r_opt


def solve_approx_lp_deterministic_sweep(
    g: DFGraph,
    budget: int,
    seed_s: Optional[np.ndarray] = None,
    approx=True,
    time_limit: Optional[int] = None,
    write_log_file: Optional[PathLike] = None,
    print_to_console=True,
    write_model_file: Optional[PathLike] = None,
    eps_noise=0.01,
    solver_cores=os.cpu_count(),
    thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    imposed_schedule: ImposedSchedule = ImposedSchedule.FULL_SCHEDULE,
):
    param_dict = {
        "LogToConsole": 1 if print_to_console else 0,
        "LogFile": str(write_log_file) if write_log_file is not None else "",
        "Threads": solver_cores,
        "TimeLimit": math.inf if time_limit is None else time_limit,
        "OptimalityTol": 1e-2 if approx else 1e-4,
        "IntFeasTol": 1e-3 if approx else 1e-5,
        "Presolve": 2,
        "StartNodeLimit": 10000000,
    }
    lpsolver = ILPSolver(
        g,
        int(0.9 * budget),  # hack to get values under the budget
        gurobi_params=param_dict,
        seed_s=seed_s,
        integral=False,
        solve_r=False,
        eps_noise=eps_noise,
        imposed_schedule=imposed_schedule,
        write_model_file=write_model_file,
    )
    lpsolver.build_model()
    try:
        r, s, u, free_e = lpsolver.solve()
        lp_feasible = True
    except ValueError as e:
        logging.exception(e)
        r, s, u, free_e = (None, None, None, None)
        lp_feasible = False
    schedule, aux_data, min_threshold = None, None, None
    if lp_feasible:  # round the solution
        for threshold in thresholds:
            s_ = (s >= threshold).astype(np.int)
            r_ = solve_r_opt(g, s_)
            schedule_, aux_data_ = schedule_from_rs(g, r_, s_)
            if aux_data_.activation_ram <= budget and (aux_data is None or aux_data_.cpu <= aux_data.cpu):
                aux_data = aux_data_
                schedule = schedule_
                min_threshold = threshold
    return ScheduledResult(
        solve_strategy=SolveStrategy.APPROX_DET_ROUND_LP_SWEEP,
        solver_budget=budget,
        feasible=lp_feasible and aux_data is not None,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=lpsolver.solve_time,
        ilp_aux_data=ILPAuxData(
            U=u,
            Free_E=free_e,
            ilp_approx=approx,
            ilp_time_limit=time_limit,
            ilp_eps_noise=eps_noise,
            ilp_num_constraints=lpsolver.m.numConstrs,
            ilp_num_variables=lpsolver.m.numVars,
            approx_deterministic_round_threshold=min_threshold,
        ),
    )


def solve_approx_lp_deterministic_rand_threshold(
    g: DFGraph,
    budget: int,
    seed_s: Optional[np.ndarray] = None,
    approx=True,
    time_limit: Optional[int] = None,
    write_log_file: Optional[PathLike] = None,
    print_to_console=True,
    write_model_file: Optional[PathLike] = None,
    eps_noise=0.01,
    solver_cores=os.cpu_count(),
    n_samples=1,
):
    thresholds = [min(1.0, max(0.0, np.random.normal(0.5, 0.5))) for i in range(n_samples)]
    return solve_approx_lp_deterministic_sweep(
        g,
        budget,
        seed_s,
        approx,
        time_limit,
        write_log_file,
        print_to_console,
        write_model_file,
        eps_noise,
        solver_cores,
        thresholds=thresholds,
    )


def solve_approx_lp_deterministic_05_threshold(
    g: DFGraph,
    budget: int,
    seed_s: Optional[np.ndarray] = None,
    approx=True,
    time_limit: Optional[int] = None,
    write_log_file: Optional[PathLike] = None,
    print_to_console=True,
    write_model_file: Optional[PathLike] = None,
    eps_noise=0.01,
    solver_cores=os.cpu_count(),
    n_samples=1,
):
    return solve_approx_lp_deterministic_sweep(
        g,
        budget,
        seed_s,
        approx,
        time_limit,
        write_log_file,
        print_to_console,
        write_model_file,
        eps_noise,
        solver_cores,
        thresholds=[0.5],
    )


def solve_approx_lp_randomized(
    g: DFGraph,
    budget: int,
    seed_s: Optional[np.ndarray] = None,
    approx=True,
    time_limit: Optional[int] = None,
    write_log_file: Optional[PathLike] = None,
    print_to_console=True,
    write_model_file: Optional[PathLike] = None,
    eps_noise=0.01,
    solver_cores=os.cpu_count(),
    num_rounds=100
):
    param_dict = {
        "LogToConsole": 1 if print_to_console else 0,
        "LogFile": str(write_log_file) if write_log_file is not None else "",
        "Threads": solver_cores,
        "TimeLimit": math.inf if time_limit is None else time_limit,
        "OptimalityTol": 1e-2 if approx else 1e-4,
        "IntFeasTol": 1e-3 if approx else 1e-5,
        "Presolve": 2,
        "StartNodeLimit": 10000000,
    }
    lpsolver = ILPSolver(
        g,
        int(0.9 * budget),  # hack to get values under the budget
        gurobi_params=param_dict,
        seed_s=seed_s,
        solve_r=False,
        integral=False,
        eps_noise=eps_noise,
        write_model_file=write_model_file,
    )
    lpsolver.build_model()
    try:
        r, s, u, free_e = lpsolver.solve()
        lp_feasible = True
    except ValueError as e:
        logging.exception(e)
        r, s, u, free_e = (None, None, None, None)
        lp_feasible = False
    best_solution = (float("inf"), None, None)
    if lp_feasible:  # round the solution
        for i in range(num_rounds):
            s_ = (np.random.rand(*s.shape) <= s).astype(np.int32)
            r_ = solve_r_opt(g, s_)
            schedule, aux_data = schedule_from_rs(g, r_, s_)
            if aux_data.cpu < best_solution[0]:
                best_solution = (aux_data.cpu, schedule, aux_data)
            if (i+1) % 1 == 0:
                print(f"Rounded relaxation argmin {i+1} / num_rounds times, best cost {best_solution[0]}")
    schedule, aux_data = best_solution[1], best_solution[2]
    return ScheduledResult(
        solve_strategy=SolveStrategy.APPROX_DET_ROUND_LP_SWEEP,
        solver_budget=budget,
        feasible=lp_feasible,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=lpsolver.solve_time,
        ilp_aux_data=ILPAuxData(
            U=u,
            Free_E=free_e,
            ilp_approx=approx,
            ilp_time_limit=time_limit,
            ilp_eps_noise=eps_noise,
            ilp_num_constraints=lpsolver.m.numConstrs,
            ilp_num_variables=lpsolver.m.numVars,
            approx_deterministic_round_threshold=None,
        ),
    )
