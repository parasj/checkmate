import logging

from checkmate.core.graph_builder import gen_linear_graph
from checkmate.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_sweep
from experiments.common.definitions import checkmate_data_dir
from experiments.common.graph_plotting import plot_schedule
from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.core.utils.timer import Timer
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    N = 16
    for B in range(4, 12):
        # model = get_keras_model("MobileNet")
        # g = dfgraph_from_keras(mod=model)
        g = gen_linear_graph(N)
        scratch_dir = checkmate_data_dir() / f"scratch_linear" / str(N) / str(B)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        data = []

        scheduler_result_all = solve_checkpoint_all(g)
        scheduler_result_sqrtn = solve_chen_sqrtn(g, True)
        plot_schedule(scheduler_result_all, False, save_file=scratch_dir / "CHECKPOINT_ALL.png")
        plot_schedule(scheduler_result_sqrtn, False, save_file=scratch_dir / "CHEN_SQRTN.png")
        data.append(
            {
                "Strategy": str(scheduler_result_all.solve_strategy.value),
                "Name": "CHECKPOINT_ALL",
                "CPU": scheduler_result_all.schedule_aux_data.cpu,
                "Activation RAM": scheduler_result_all.schedule_aux_data.activation_ram,
            }
        )
        data.append(
            {
                "Strategy": str(scheduler_result_sqrtn.solve_strategy.value),
                "Name": "CHEN_SQRTN",
                "CPU": scheduler_result_sqrtn.schedule_aux_data.cpu,
                "Activation RAM": scheduler_result_sqrtn.schedule_aux_data.activation_ram,
            }
        )

        logging.error("Skipping Griewank baselines as it was broken in parasj/checkmate#65")
        # scheduler_result_griewank = solve_griewank(g, B)
        # plot_schedule(scheduler_result_griewank, False, save_file=scratch_dir / "GRIEWANK.png")
        # data.append(
        #     {
        #         "Strategy": str(scheduler_result_griewank.solve_strategy.value),
        #         "Name": "GRIEWANK",
        #         "CPU": scheduler_result_griewank.schedule_aux_data.cpu,
        #         "Activation RAM": scheduler_result_griewank.schedule_aux_data.activation_ram,
        #     }
        # )

        with Timer("ilp") as timer_ilp:
            scheduler_result_ilp = solve_ilp_gurobi(g, B, seed_s=scheduler_result_sqrtn.schedule_aux_data.S)
            plot_schedule(scheduler_result_ilp, False, save_file=scratch_dir / "CHECKMATE_ILP.png")
            data.append(
                {
                    "Strategy": str(scheduler_result_ilp.solve_strategy.value),
                    "Name": "CHECKMATE_ILP",
                    "CPU": scheduler_result_ilp.schedule_aux_data.cpu,
                    "Activation RAM": scheduler_result_ilp.schedule_aux_data.activation_ram,
                }
            )

        with Timer("det_lp") as timer_lp_det:
            scheduler_lp_deterministicround = solve_approx_lp_deterministic_sweep(g, B)
            if scheduler_lp_deterministicround.schedule_aux_data is not None:
                plot_schedule(
                    scheduler_lp_deterministicround,
                    False,
                    save_file=scratch_dir
                    / f"CHECKMATE_LP_DETERMINISTICROUND_{scheduler_lp_deterministicround.ilp_aux_data.approx_deterministic_round_threshold}.png",
                )
                data.append(
                    {
                        "Strategy": str(scheduler_lp_deterministicround.solve_strategy.value),
                        "Name": f"CHECKM8_DET_APPROX_{scheduler_lp_deterministicround.ilp_aux_data.approx_deterministic_round_threshold:.2f}",
                        "CPU": scheduler_lp_deterministicround.schedule_aux_data.cpu,
                        "Activation RAM": scheduler_lp_deterministicround.schedule_aux_data.activation_ram,
                    }
                )

        with (scratch_dir / "times.log").open("w") as f:
            f.write(timer_ilp._format_results())
            f.write(timer_lp_det._format_results())

        df = pd.DataFrame(data)
        df.to_csv(scratch_dir / "data.csv")
        df.plot.barh(y="CPU", x="Name")
        plt.savefig(scratch_dir / "barplot.png")
