import numpy as np

from remat.core.dfgraph import gen_linear_graph
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all


def test_checkpointall():
    for graph_length in [2, 4, 8, 16]:
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        r, s = solve_checkpoint_all(g)
        print(f"{graph_length} -> {np.mean(r)} {np.mean(s)}")
