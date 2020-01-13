import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from experiments.common.profile.platforms import PLATFORM_CHOICES, platform_memory
from experiments.common.profile.cost_model import CostModel
from checkmate.tf2.load_keras_model import MODEL_NAMES, get_keras_model
from experiments.common.graph_plotting import plot_schedule
from experiments.common.definitions import checkmate_data_dir
from checkmate.tf2_keras.extraction import dfgraph_from_keras
from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_05_threshold, solve_approx_lp_randomized
from checkmate.core.graph_builder import gen_linear_graph
from checkmate.core.enum_strategy import SolveStrategy


LINEAR_MODELS = ["linear16", "VGG16", "VGG19", "MobileNet", "MobileNetV2"]


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", default="flops", choices=PLATFORM_CHOICES)
    parser.add_argument("--model-name", default="VGG16", choices=list(sorted(MODEL_NAMES)) + ["linear16"])
    parser.add_argument("-b", "--batch-size", type=int, default=256)
    parser.add_argument("-nr", "--num-rounds", type=int, default=256)
    parser.add_argument("--skip-ilp", action="store_true", help="If set, skip running the ILP during evaluation.")
    return parser.parse_args()


def b2gb(data):
    if hasattr(data, "__iter__"):
        return [d * 1e-9 for d in data]
    return data * 1e-9


if __name__ == "__main__":
    args = extract_params()

    if args.model_name == "linear16":
        N = 16
        B = 8
        scratch_dir = checkmate_data_dir() / "approxcomparison" / args.model_name / f"budget{B}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        g = gen_linear_graph(N)
    else:
        B = platform_memory(args.platform)
        scratch_dir = checkmate_data_dir() / "approxcomparison" / args.model_name / f"budget{B}"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # load costs, and plot optionally, if platform is not flops
        print("Loading costs")
        if args.platform == "flops":
            cost_model = None
        else:
            cost_model = CostModel(args.model_name, args.platform, scratch_dir, quantization=5)
            cost_model.fit()

        # load model from Keras
        print("Loading model {}".format(args.model_name))
        model = get_keras_model(args.model_name)
        g = dfgraph_from_keras(
            model, batch_size=args.batch_size, cost_model=cost_model, loss_cpu_cost=0, loss_ram_cost=(4 * args.batch_size)
        )

    common_kwargs = dict(g=g, budget=B, print_to_console=False, eps_noise=0, approx=False)

    print("Common args:", common_kwargs)

    data = []

    # Checkpoint all
    scheduler_result_all = solve_checkpoint_all(g)
    plot_schedule(scheduler_result_all, False, save_file=scratch_dir / "ALL.png")
    data.append(
        {
            "Strategy": str(scheduler_result_all.solve_strategy.value),
            "Name": "CHECKPOINT_ALL",
            "CPU": scheduler_result_all.schedule_aux_data.cpu,
            "Activation RAM": scheduler_result_all.schedule_aux_data.activation_ram,
        }
    )

    if args.model_name in LINEAR_MODELS:
        # Sqrt(n)
        scheduler_result_sqrtn = solve_chen_sqrtn(g, True)
        plot_schedule(scheduler_result_sqrtn, False, save_file=scratch_dir / "SQRTN.png")
        data.append(
            {
                "Strategy": str(scheduler_result_sqrtn.solve_strategy.value),
                "Name": "CHEN_SQRTN",
                "CPU": scheduler_result_sqrtn.schedule_aux_data.cpu,
                "Activation RAM": scheduler_result_sqrtn.schedule_aux_data.activation_ram,
            }
        )

    if not args.skip_ilp:
        # ILP
        scheduler_result_ilp = solve_ilp_gurobi(
            g=g,
            budget=B,
            seed_s=scheduler_result_sqrtn.schedule_aux_data.S if args.model_name in LINEAR_MODELS else None,
            write_log_file=scratch_dir / "ilp.log",
            print_to_console=False,
            eps_noise=0,
            approx=False,
        )
        if scheduler_result_ilp.schedule_aux_data is not None:
            plot_schedule(scheduler_result_ilp, False, save_file=scratch_dir / "CHECKM8_ILP.png")
            data.append(
                {
                    "Strategy": str(scheduler_result_ilp.solve_strategy.value),
                    "Name": "CHECKMATE_ILP",
                    "CPU": scheduler_result_ilp.schedule_aux_data.cpu,
                    "Activation RAM": scheduler_result_ilp.schedule_aux_data.activation_ram,
                }
            )

    # Deterministic rounding
    scheduler_lp_det = solve_approx_lp_deterministic_05_threshold(
        write_log_file=scratch_dir / "lp_det_05.log", allow_return_infeasible_schedule=True, **common_kwargs
    )
    if scheduler_lp_det.schedule_aux_data is not None:
        plot_schedule(scheduler_lp_det, False, save_file=scratch_dir / "CHECKM8_DET_APPROX_05.png")
        data.append(
            {
                "Strategy": str(scheduler_lp_det.solve_strategy.value),
                "Name": "CHECKM8_DET_APPROX_05",
                "CPU": scheduler_lp_det.schedule_aux_data.cpu,
                "Activation RAM": scheduler_lp_det.schedule_aux_data.activation_ram,
            }
        )

    # Randomized rounding
    scheduler_lp_rand, rounding_stats = solve_approx_lp_randomized(
        write_log_file=scratch_dir / "lp_rand.log", num_rounds=args.num_rounds, return_rounds=True, **common_kwargs
    )
    if scheduler_lp_rand.schedule_aux_data is not None:
        plot_schedule(scheduler_lp_rand, False, save_file=scratch_dir / "CHECKM8_RAND_APPROX.png")
        data.append(
            {
                "Strategy": str(scheduler_lp_rand.solve_strategy.value),
                "Name": "CHECKM8_RAND_APPROX",
                "CPU": scheduler_lp_rand.schedule_aux_data.cpu,
                "Activation RAM": scheduler_lp_rand.schedule_aux_data.activation_ram,
            }
        )

    # Plot solution memory usage vs cpu scatter plot
    sns.set()
    sns.set_style("white")

    plt.figure(figsize=(4, 4))
    plt.xlabel("Activation memory usage (GB)")
    plt.ylabel("GPU time (ms)")

    color, marker, markersize = SolveStrategy.get_plot_params(scheduler_result_all.solve_strategy)
    plt.axhline(
        y=scheduler_result_all.schedule_aux_data.cpu / 1000, color=color, linestyle="--", label="Checkpoint all (ideal)"
    )

    if args.model_name in LINEAR_MODELS:
        color, marker, markersize = SolveStrategy.get_plot_params(scheduler_result_sqrtn.solve_strategy)
        plt.scatter(
            [b2gb(scheduler_result_sqrtn.schedule_aux_data.activation_ram)],
            [scheduler_result_sqrtn.schedule_aux_data.cpu / 1000],
            s=markersize ** 2,
            color=color,
            marker=marker,
            label="Chen $\sqrt{n}$",
        )

    _, marker, markersize = SolveStrategy.get_plot_params(scheduler_lp_rand.solve_strategy)
    plt.scatter(
        b2gb(rounding_stats["activation_ram"]),
        np.array(rounding_stats["cpu"]) / 1000,
        s=markersize ** 2,
        color="lightcoral",
        marker=marker,
        label="Randomized rounding",
    )
    plt.axhline(y=np.mean(rounding_stats["cpu"]) / 1000, color="lightcoral", linestyle=":")

    color, marker, markersize = SolveStrategy.get_plot_params(scheduler_lp_det.solve_strategy)
    plt.scatter(
        [b2gb(scheduler_lp_det.schedule_aux_data.activation_ram)],
        [scheduler_lp_det.schedule_aux_data.cpu / 1000],
        s=markersize ** 2,
        color="royalblue",
        marker=marker,
        label="Deterministic rounding",
    )

    if not args.skip_ilp:
        color, marker, markersize = SolveStrategy.get_plot_params(scheduler_result_ilp.solve_strategy)
        plt.scatter(
            [b2gb(scheduler_result_ilp.schedule_aux_data.activation_ram)],
            [scheduler_result_ilp.schedule_aux_data.cpu / 1000],
            s=markersize ** 2,
            color=color,
            marker=marker,
            label="ILP",
        )

    plt.legend()
    plt.savefig(scratch_dir / "scatter.pdf", bbox_inches="tight", format="pdf")
    plt.savefig(scratch_dir / "scatter.png", bbox_inches="tight", dpi=300)

    # Save results
    df = pd.DataFrame(data)
    df.to_csv(scratch_dir / "data.csv")
    df.plot.barh(y="CPU", x="Name")
    plt.savefig(scratch_dir / "barplot.png")

    # Save data
    with open(scratch_dir / "data.pickle", "wb") as f:
        pickle.dump({"data": data, "rounding_stats": rounding_stats}, f, protocol=pickle.HIGHEST_PROTOCOL)
