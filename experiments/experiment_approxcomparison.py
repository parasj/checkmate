import logging

from checkmate.core.graph_builder import gen_linear_graph
from checkmate.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_05_threshold, solve_approx_lp_randomized
from experiments.common.definitions import checkmate_data_dir
from experiments.common.graph_plotting import plot_schedule
from checkmate.core.enum_strategy import SolveStrategy
from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.core.utils.timer import Timer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    N = 16
    B = 8

    # model = get_keras_model("MobileNet")
    # g = dfgraph_from_keras(mod=model)
    g = gen_linear_graph(N)
    scratch_dir = checkmate_data_dir() / "approxcomparison" / f"linear{N}" / f"budget{B}"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    data = []

    common_kwargs = dict(
        g=g,
        budget=B,
        print_to_console=False,
        eps_noise=0,
        approx=False,
    )

    scheduler_result_all = solve_checkpoint_all(g)
    data.append(
        {
            "Strategy": str(scheduler_result_all.solve_strategy.value),
            "Name": "CHECKPOINT_ALL",
            "CPU": scheduler_result_all.schedule_aux_data.cpu,
            "Activation RAM": scheduler_result_all.schedule_aux_data.activation_ram,
        }
    )

    scheduler_result_sqrtn = solve_chen_sqrtn(g, True)
    data.append(
        {
            "Strategy": str(scheduler_result_sqrtn.solve_strategy.value),
            "Name": "CHEN_SQRTN",
            "CPU": scheduler_result_sqrtn.schedule_aux_data.cpu,
            "Activation RAM": scheduler_result_sqrtn.schedule_aux_data.activation_ram,
        }
    )

    with Timer("ilp") as timer_ilp:
        scheduler_result_ilp = solve_ilp_gurobi(
            g, B,
            seed_s=scheduler_result_sqrtn.schedule_aux_data.S,
            write_log_file=scratch_dir / "ilp.log",
            print_to_console=False,
        )
        data.append(
            {
                "Strategy": str(scheduler_result_ilp.solve_strategy.value),
                "Name": "CHECKMATE_ILP",
                "CPU": scheduler_result_ilp.schedule_aux_data.cpu,
                "Activation RAM": scheduler_result_ilp.schedule_aux_data.activation_ram,
            }
        )

    scheduler_lp_det = solve_approx_lp_deterministic_05_threshold(
        write_log_file=scratch_dir / "lp_det_05.log",
        **common_kwargs
    )
    if scheduler_lp_det.schedule_aux_data is not None:
        data.append({
                "Strategy": str(scheduler_lp_det.solve_strategy.value),
                "Name": "CHECKM8_DET_APPROX_05",
                "CPU": scheduler_lp_det.schedule_aux_data.cpu,
                "Activation RAM": scheduler_lp_det.schedule_aux_data.activation_ram,
        })

    scheduler_lp_rand, rounding_stats = solve_approx_lp_randomized(
        write_log_file=scratch_dir / "lp_rand.log",
        num_rounds=500,
        return_rounds=True,
        **common_kwargs
    )
    if scheduler_lp_rand.schedule_aux_data is not None:
        data.append({
                "Strategy": str(scheduler_lp_rand.solve_strategy.value),
                "Name": "CHECKM8_RAND_APPROX",
                "CPU": scheduler_lp_rand.schedule_aux_data.cpu,
                "Activation RAM": scheduler_lp_rand.schedule_aux_data.activation_ram,
        })


    # Plot solution memory usage vs cpu scatter plot
    sns.set()
    sns.set_style("white")

    plt.figure()
    plt.xlabel("Activation memory usage")
    plt.ylabel("Cost")

    _, marker, markersize = SolveStrategy.get_plot_params(scheduler_lp_rand.solve_strategy)
    plt.scatter(rounding_stats["activation_ram"], rounding_stats["cpu"],
                s=markersize ** 2, color="lightcoral", marker=marker, label="Randomized rounding")

    color, marker, markersize = SolveStrategy.get_plot_params(scheduler_lp_det.solve_strategy)
    plt.scatter([scheduler_lp_det.schedule_aux_data.activation_ram], [scheduler_lp_det.schedule_aux_data.cpu],
                s=markersize ** 2, color="royalblue", marker=marker, label="Deterministic rounding")

    color, marker, markersize = SolveStrategy.get_plot_params(scheduler_result_sqrtn.solve_strategy)
    plt.scatter([scheduler_result_sqrtn.schedule_aux_data.activation_ram], [scheduler_result_sqrtn.schedule_aux_data.cpu],
                s=markersize ** 2, color=color, marker=marker, label="Chen $\sqrt{n}$")

    color, marker, markersize = SolveStrategy.get_plot_params(scheduler_result_all.solve_strategy)
    plt.scatter([scheduler_result_all.schedule_aux_data.activation_ram], [scheduler_result_all.schedule_aux_data.cpu],
                s=markersize ** 2, color=color, marker=marker, label="Checkpoint all (ideal)")

    color, marker, markersize = SolveStrategy.get_plot_params(scheduler_result_ilp.solve_strategy)
    plt.scatter([scheduler_result_ilp.schedule_aux_data.activation_ram], [scheduler_result_ilp.schedule_aux_data.cpu],
                s=markersize ** 2, color=color, marker=marker, label="ILP")

    plt.legend()
    plt.tight_layout()
    plt.savefig(scratch_dir / "scatter.png")


    df = pd.DataFrame(data)
    df.to_csv(scratch_dir / "data.csv")
    df.plot.barh(y="CPU", x="Name")
    plt.savefig(scratch_dir / "barplot.png")
