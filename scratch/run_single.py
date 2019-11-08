from experiments.common.graph_plotting import plot
from remat.core.dfgraph import gen_linear_graph
from remat.core.solvers.strategy_griewank import solve_griewank
from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi

# model = get_keras_model("MobileNet")
# g = dfgraph_from_keras(mod=model)
g = gen_linear_graph(64)
budget = 8
seed_result = solve_griewank(g, budget)
scheduler_result = solve_ilp_gurobi(g, budget, solver_cores=12, integral=False, approx=False, seed_s=seed_result.schedule_aux_data.S)
plot(scheduler_result, False, None, True)
