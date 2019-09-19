from remat.core.dfgraph import DFGraph
from remat.core.solvers.common import SOLVER_DTYPE, solve_r_opt
import numpy as np


def schedule_checkpoint_last_node(g: DFGraph):
    """Checkpoint only one node between stages"""
    S = np.zeros((g.size, g.size), dtype=SOLVER_DTYPE)
    np.fill_diagonal(S[1:], 1)
    R = solve_r_opt(g, S)
    return R, S
