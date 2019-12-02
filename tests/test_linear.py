import logging

import numpy as np

from experiments.common.graph_plotting import plot_schedule
from checkmate.core.graph_builder import gen_linear_graph
from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from checkmate.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from checkmate.core.solvers.strategy_chen import solve_chen_greedy, solve_chen_sqrtn

test_points = [2, 4, 6, 8, 9, 11, 13, 16, 32]
SAVE_DEBUG_PLOTS = False


def test_checkpoint_all():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        scheduler_result = solve_checkpoint_all(g)
        assert scheduler_result.feasible
        assert scheduler_result.schedule_aux_data.cpu == g.size
        if SAVE_DEBUG_PLOTS:
            plot_schedule(scheduler_result, save_file=f"/tmp/test_checkmate/plot_checkpoint_all/{graph_length}.png")


def test_checkpoint_last():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        scheduler_result = solve_checkpoint_last_node(g)
        assert scheduler_result.feasible
        if SAVE_DEBUG_PLOTS:
            plot_schedule(scheduler_result, False, save_file=f"/tmp/test_checkmate/plot_checkmate_last/{graph_length}.png")


def test_checkpoint_all_ap():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        scheduler_result = solve_checkpoint_all_ap(g)
        assert scheduler_result.feasible
        if SAVE_DEBUG_PLOTS:
            plot_schedule(scheduler_result, save_file=f"/tmp/test_checkmate/plot_checkpoint_all_ap/{graph_length}.png")


def test_chen_sqrtn():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_chen_sqrtn(g, total_cost)
        assert scheduler_result.feasible
        if SAVE_DEBUG_PLOTS:
            plot_schedule(scheduler_result, save_file=f"/tmp/test_checkmate/plot_chen_sqrtn/{graph_length}.png")


def test_chen_greedy():
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_chen_greedy(g, total_cost, False)
        assert scheduler_result.feasible
        if SAVE_DEBUG_PLOTS:
            for budget in np.arange(0, 1, 0.1):
                scheduler_result = solve_chen_greedy(g, total_cost * budget, False)
                if scheduler_result.feasible:
                    plot_schedule(scheduler_result, save_file=f"/tmp/test_checkmate/plot_chen_greedy/{graph_length}_{budget}.png")


def test_chen_greedy_ap():
    for graph_length in [2, 4, 5, 7, 8]:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_chen_greedy(g, total_cost, True)
        assert scheduler_result.feasible
        if SAVE_DEBUG_PLOTS:
            for budget in np.arange(0, 1, 0.1):
                scheduler_result = solve_chen_greedy(g, total_cost * budget, False)
                if scheduler_result.feasible:
                    plot_schedule(scheduler_result, save_file=f"/tmp/test_checkmate/plot_chen_greedy_ap/{graph_length}_{budget}.png")


def test_ilp():
    try:
        import gurobipy as _
    except ImportError as e:
        logging.exception(e)
        logging.warning("Continuing with tests, gurobi not installed")
        return
    from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
    for graph_length in test_points:
        g = gen_linear_graph(graph_length)
        assert g.size == 2 * graph_length + 1
        total_cost = sum(g.cost_ram.values())
        scheduler_result = solve_ilp_gurobi(g, total_cost, print_to_console=False, write_log_file=None)
        assert scheduler_result.feasible
        if SAVE_DEBUG_PLOTS:
            for budget in np.arange(0, 1, 0.25):
                scheduler_result = solve_ilp_gurobi(g, total_cost * budget, print_to_console=False, write_log_file=None, time_limit=15)
                if scheduler_result.feasible:
                    plot_schedule(scheduler_result, save_file=f"/tmp/test_checkmate/plot_ilp/{graph_length}_{budget}.png")


# NOTE: Griewank test is disabled as the solver does not support nonlinear graphs.
# def test_griewank():
#     for graph_length in test_points:
#         g = gen_linear_graph(graph_length)
#         assert g.size == 2 * graph_length + 1
#         total_cost = sum(g.cost_ram.values())
#         scheduler_result = solve_griewank(g, total_cost)
#         assert scheduler_result.feasible
