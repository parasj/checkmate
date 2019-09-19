from remat.core.dfgraph import DFGraph
from remat.core.solvers.common import gen_s_matrix_fixed_checkpoints, solve_r_opt


def solve_checkpoint_all(g: DFGraph):
    """Checkpoint only one node between stages"""
    s = gen_s_matrix_fixed_checkpoints(g, g.vfwd)
    r = solve_r_opt(g, s)
    return r, s
