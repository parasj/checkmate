from remat.core.dfgraph import gen_linear_graph
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node


def test_checkpoint_all():
    for graph_length in range(2, 32):
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        scheduler_result = solve_checkpoint_all(g)
        assert scheduler_result.feasible
        assert scheduler_result.schedule_aux_data.cpu == g.size
        # todo check memory cost, need closed form for this for linear graphs


def test_checkpoint_last():
    for graph_length in range(2, 32):
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        scheduler_result = solve_checkpoint_last_node(g)
        assert scheduler_result.feasible


def test_checkpoint_all_ap():
    for graph_length in range(2, 32):
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        scheduler_result = solve_checkpoint_all_ap(g)
        assert scheduler_result.feasible
