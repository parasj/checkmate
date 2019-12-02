import numpy as np

from checkmate.core.dfgraph import DFGraph
from checkmate.core.schedule import ScheduledResult
from checkmate.core.utils.solver_common import SOLVER_DTYPE, solve_r_opt
from checkmate.core.enum_strategy import SolveStrategy
from checkmate.core.utils.scheduler import schedule_from_rs
from checkmate.core.utils.timer import Timer


def solve_checkpoint_last_node(g: DFGraph):
    """Checkpoint only one node between stages"""
    with Timer('solve_checkpoint_last_node') as timer_solve:
        s = np.zeros((g.size, g.size), dtype=SOLVER_DTYPE)
        np.fill_diagonal(s[1:], 1)
        r = solve_r_opt(g, s)
    schedule, aux_data = schedule_from_rs(g, r, s)
    return ScheduledResult(
        solve_strategy=SolveStrategy.CHECKPOINT_LAST_NODE,
        solver_budget=0,
        feasible=True,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=timer_solve.elapsed
    )
