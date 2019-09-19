from remat.core.dfgraph import gen_linear_graph
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all


def test_checkpointall():
    for graph_length in range(2, 32):
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        scheduler_result = solve_checkpoint_all(g)
        assert scheduler_result.schedule_aux_data.cpu == g.size
        # todo check memory cost, need closed form for this for linear graphs
