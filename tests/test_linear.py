import logging

from remat.core.graph_builder import gen_linear_graph
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_greedy, solve_chen_sqrtn
from remat.core.solvers.strategy_griewank import solve_griewank


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


def test_chen_sqrtn():
    for graph_length in [2, 4, 5, 7, 8]:
        for budget in range(1, min(graph_length, 4)):
            g = gen_linear_graph(graph_length)
            assert g.size_fwd == graph_length
            total_cost = sum(g.cost_ram.values())
            scheduler_result = solve_chen_sqrtn(g, total_cost)
            assert scheduler_result.feasible


def test_chen_greedy():
    for graph_length in [2, 4, 5, 7, 8]:
        for budget in range(1, min(graph_length, 4)):
            g = gen_linear_graph(graph_length)
            assert g.size_fwd == graph_length
            total_cost = sum(g.cost_ram.values())
            scheduler_result = solve_chen_greedy(g, total_cost, False)
            assert scheduler_result.feasible


def test_chen_greedy_ap():
    for graph_length in [2, 4, 5, 7, 8]:
        for budget in range(1, min(graph_length, 4)):
            g = gen_linear_graph(graph_length)
            assert g.size_fwd == graph_length
            total_cost = sum(g.cost_ram.values())
            scheduler_result = solve_chen_greedy(g, total_cost, True)
            assert scheduler_result.feasible


def test_ilp():
    try:
        import gurobipy as _
    except ImportError as e:
        logging.exception(e)
        logging.warning("Continuing with tests, gurobi not installed")
        return
    from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
    for graph_length in [2, 4, 8]:
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_ilp_gurobi(g, total_cost, print_to_console=False, write_log_file=None)
        assert scheduler_result.feasible


def test_griewank():
    for graph_length in [2 ** i for i in range(1, 6)]:
        g = gen_linear_graph(graph_length)
        assert g.size_fwd == graph_length
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_griewank(g, total_cost)
        assert scheduler_result.feasible
