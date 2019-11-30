import logging

from experiments.common.graph_plotting import plot_schedule, plot_dfgraph
from remat.core.graph_builder import gen_linear_graph
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_greedy, solve_chen_sqrtn
from remat.core.solvers.strategy_griewank import solve_griewank

test_points = [2, 4, 6, 8, 9, 11, 13, 16, 32]


def test_checkpoint_all():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        scheduler_result = solve_checkpoint_all(g)
        assert scheduler_result.feasible
        assert scheduler_result.schedule_aux_data.cpu == g.size


def test_checkpoint_last():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        scheduler_result = solve_checkpoint_last_node(g)
        plot_schedule(scheduler_result, False, save_file=f"/tmp/test_remat/plot{graph_length}.png")
        assert scheduler_result.feasible


def test_checkpoint_all_ap():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        scheduler_result = solve_checkpoint_all_ap(g)
        assert scheduler_result.feasible


def test_chen_sqrtn():
    for graph_length in [2, 4, 5, 7, 8]:
        for budget in range(1, min(graph_length, 4)):
            g = gen_linear_graph(graph_length)
            assert g.size == 2 * graph_length + 1
            total_cost = sum(g.cost_ram.values())
            scheduler_result = solve_chen_sqrtn(g, total_cost)
            assert scheduler_result.feasible


def test_chen_greedy():
    for graph_length in [2, 4, 5, 7, 8]:
        for budget in range(1, min(graph_length, 4)):
            g = gen_linear_graph(graph_length)
            assert g.size == 2 * graph_length + 1
            total_cost = sum(g.cost_ram.values())
            scheduler_result = solve_chen_greedy(g, total_cost, False)
            assert scheduler_result.feasible


def test_chen_greedy_ap():
    for graph_length in [2, 4, 5, 7, 8]:
        for budget in range(1, min(graph_length, 4)):
            g = gen_linear_graph(graph_length)
            assert g.size == 2 * graph_length + 1
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
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_ilp_gurobi(g, total_cost, print_to_console=False, write_log_file=None)
        assert scheduler_result.feasible


def test_griewank():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_griewank(g, total_cost)
        assert scheduler_result.feasible
