import numpy as np

from remat.core.dfgraph import DFGraph
from remat.core.schedule import ScheduledResult
from remat.core.solvers.common import SOLVER_DTYPE, solve_r_opt
from remat.core.solvers.scheduler import schedule_rs_matrix
from remat.core.solvers.enum_strategy import SolveStrategy
from remat.core.utils.timer import Timer


def solve_checkpoint_last_node(g: DFGraph):
    """Checkpoint only one node between stages"""
    with Timer('solve_checkpoint_last_node') as timer_solve:
        s = np.zeros((g.size, g.size), dtype=SOLVER_DTYPE)
        np.fill_diagonal(s[1:], 1)
        r = solve_r_opt(g, s)
    schedule, aux_data = schedule_rs_matrix(g, r, s)
    return ScheduledResult(
        solve_strategy=SolveStrategy.CHECKPOINT_LAST_NODE,
        solver_budget=0,
        feasible=True,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=timer_solve.elapsed
    )
