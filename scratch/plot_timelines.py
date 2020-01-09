import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from experiments.common.graph_plotting import plot
from experiments.common.load_keras_model import get_keras_model
from remat.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from remat.core.solvers.strategy_chen import solve_chen_sqrtn
from remat.tensorflow2.extraction import dfgraph_from_keras
from remat.core.schedule import ScheduledResult, OperatorEvaluation
from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi

model = get_keras_model("VGG16")
g = dfgraph_from_keras(model, next_outputs_deps=False)
# render_dfgraph(g, directory="data/plot_timelines", format="png")

CHEN_LABEL = "Chen 2016"

scheds = {}
scheds["TF2.0"] = solve_checkpoint_all(g)
scheds[CHEN_LABEL] = solve_chen_sqrtn(g, False)
chen_budget = scheds[CHEN_LABEL].schedule_aux_data.activation_ram * 0.69
scheds["Checkmate"] = solve_ilp_gurobi(g, budget=int(chen_budget), approx=True, eps_noise=0.0)

with plt.style.context("bmh"):
    pd_data = []
    for sched_name, sched in scheds.items():
        sched: ScheduledResult = sched
        exec_data_runtime = []
        exec_data = []
        for sched_item, ram_usage in zip(sched.schedule, sched.schedule_aux_data.mem_timeline):
            if isinstance(sched_item, OperatorEvaluation):
                last_sched_cost = exec_data_runtime[-1] if len(exec_data_runtime) > 0 else 0
                cost = sched_item.operator_cost + last_sched_cost
                pd_data.append({"Schedule": sched_name, "Time": cost, "RAM": ram_usage})
                exec_data_runtime.append(cost)

    df = pd.DataFrame(pd_data)
    fig = sns.lineplot("Time", "RAM", hue="Schedule", data=df, markers="o", ci=None)
    fig.legend_.remove()
    plt.savefig(f"data/plot_timelines/ram_timeline.pdf", bbox_inches="tight")
    plt.figure()
    for sched_name, sched in scheds.items():
        plot(sched, plot_mem_usage=False, show=False, plt=plt)
        plt.savefig(f"data/plot_timelines/{sched_name}_sched.pdf", bbox_inches="tight")
