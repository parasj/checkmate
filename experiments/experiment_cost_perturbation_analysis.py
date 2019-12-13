import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.tf2_keras.extraction import dfgraph_from_keras
from experiments.common.load_keras_model import get_keras_model

# get sample network and generate a graph on it
model = get_keras_model("VGG16")
g = dfgraph_from_keras(mod=model)
assert g.size_fwd == 6
budget = sum(g.cost_ram.values()) + g.cost_ram_parameters

# solve for a schedule
scheduler_result = solve_ilp_gurobi(g, budget)
R = scheduler_result.schedule_aux_data.R

# compute costs for 1000 runs
r = R.sum(axis=0)
C @ r
results = [np.random.normal(C, 1e7) @ r for i in range(1000)]
plt.figure()
x = pd.Series(results, name="Cost in flops")
sns.distplot(x)
plt.savefig('distribution_of_perturbed_c.png')
