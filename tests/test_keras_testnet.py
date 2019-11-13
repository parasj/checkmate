import logging

from experiments.common.load_keras_model import get_keras_model
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.tensorflow2.extraction import dfgraph_from_keras


def test_testnet_checkpointall():
    model = get_keras_model("test")
    g = dfgraph_from_keras(mod=model)
    assert g.size_fwd == 6
    scheduler_result = solve_checkpoint_all(g)
    assert scheduler_result.feasible
    assert scheduler_result.schedule_aux_data.cpu == sum(g.cost_cpu.values())


def test_testnet_optimalilp():
    try:
        import gurobipy as _
    except ImportError as e:
        logging.exception(e)
        logging.warning("Continuing with tests, gurobi not installed")
        return
    from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi

    model = get_keras_model("test")
    g = dfgraph_from_keras(mod=model)
    assert g.size_fwd == 6
    budget = sum(g.cost_ram.values()) + g.cost_ram_parameters
    scheduler_result = solve_ilp_gurobi(g, budget)
    assert scheduler_result.feasible
    assert scheduler_result.schedule_aux_data.cpu <= sum(g.cost_cpu.values())
    assert scheduler_result.schedule_aux_data.activation_ram <= sum(g.cost_cpu.values())
    assert scheduler_result.schedule_aux_data.peak_ram <= budget
