import logging
import math
import os
from typing import Optional

import numpy as np

from checkmate.core.dfgraph import DFGraph
from checkmate.core.enum_strategy import SolveStrategy, ImposedSchedule
from checkmate.core.schedule import ILPAuxData, ScheduledResult, SchedulerAuxData
from checkmate.core.solvers.strategy_optimal_ilp import ILPSolver
from checkmate.core.utils.definitions import PathLike


def lower_bound_lp_relaxation(
    g: DFGraph,
    budget: int,
    seed_s: Optional[np.ndarray] = None,
    approx=True,
    imposed_schedule=ImposedSchedule.FULL_SCHEDULE,
    time_limit: Optional[int] = None,
    write_log_file: Optional[PathLike] = None,
    print_to_console=True,
    write_model_file: Optional[PathLike] = None,
    eps_noise=0.01,
    solver_cores=os.cpu_count(),
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
        budget,
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

    total_ram = u.max()
    total_cpu = lpsolver.m.getObjective().getValue()
    aux_data = SchedulerAuxData(
        R=r,
        S=s,
        cpu=total_cpu,
        peak_ram=total_ram,
        activation_ram=total_ram,
        mem_timeline=None,
        mem_grid=None,
    )

    return ScheduledResult(
        solve_strategy=SolveStrategy.LB_LP,
        solver_budget=budget,
        feasible=lp_feasible,
        schedule=None,
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
