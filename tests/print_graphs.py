from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all, solve_checkpoint_all_ap
from remat.core.solvers.strategy_checkpoint_last import solve_checkpoint_last_node
from remat.core.solvers.strategy_chen import solve_chen_greedy, solve_chen_sqrtn
from remat.core.solvers.strategy_griewank import solve_griewank
from experiments.common.load_keras_model import get_keras_model
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.tensorflow2.extraction import dfgraph_from_keras

if __name__ == "__main__":
    model = get_keras_model("test")
    g = dfgraph_from_keras(mod=model)
    scheduler_result = solve_checkpoint_all(g)
    print(scheduler_result.schedule)