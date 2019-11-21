import argparse
import logging

from experiments.common.definitions import remat_data_dir
from experiments.common.graph_plotting import plot
from remat.core.dfgraph import gen_linear_graph
from remat.core.enum_strategy import ImposedSchedule
from remat.core.solvers.lower_bound_lp import lower_bound_lp_relaxation
from remat.core.solvers.strategy_approx_lp import solve_approx_lp_deterministic_sweep
from remat.core.solvers.strategy_griewank import solve_griewank
from remat.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", "-n", default=16, type=int)
    parser.add_argument("--imposed-schedule", default=ImposedSchedule.FULL_SCHEDULE,
                        type=ImposedSchedule, choices=list(ImposedSchedule))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Set parameters
    args = parse_args()
    N = args.num_layers
    IMPOSED_SCHEDULE = args.imposed_schedule
    APPROX = False
    EPS_NOISE = 0
    SOLVE_R = False

    # Compute integrality gap for each budget
    for B in reversed(range(4, N + 3)):  # Try several budgets
        g = gen_linear_graph(N)
        scratch_dir = remat_data_dir() / f"scratch_integrality_gap_linear" / f"{N}_layers" / str(
            IMPOSED_SCHEDULE) / f"{B}_budget"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        data = []

        griewank = solve_griewank(g, B)

        logging.info("--- Solving LP relaxation for lower bound")
        lb_lp = lower_bound_lp_relaxation(g, B, approx=APPROX, eps_noise=EPS_NOISE, imposed_schedule=IMPOSED_SCHEDULE)
        plot(lb_lp, False, save_file=scratch_dir / "CHECKMATE_LB_LP.png")

        logging.info("--- Solving ILP")
        ilp = solve_ilp_gurobi(g, B, approx=APPROX, eps_noise=EPS_NOISE, imposed_schedule=IMPOSED_SCHEDULE,
                               solve_r=SOLVE_R)
        ilp_feasible = ilp.schedule_aux_data.activation_ram <= B
        plot(ilp, False, save_file=scratch_dir / "CHECKMATE_ILP.png")

        integrality_gap = ilp.schedule_aux_data.cpu / lb_lp.schedule_aux_data.cpu
        speedup = ilp.solve_time_s / lb_lp.solve_time_s

        approx_ratio_actual, approx_ratio_ub = float("inf"), float("inf")
        try:
            logging.info("--- Solving deterministic rounting of LP")
            approx_lp_determinstic = solve_approx_lp_deterministic_sweep(g, B, approx=APPROX, eps_noise=EPS_NOISE,
                                                                         imposed_schedule=IMPOSED_SCHEDULE)
            if approx_lp_determinstic.schedule_aux_data:
                approx_ratio_ub = approx_lp_determinstic.schedule_aux_data.cpu / lb_lp.schedule_aux_data.cpu
                approx_ratio_actual = approx_lp_determinstic.schedule_aux_data.cpu / ilp.schedule_aux_data.cpu
        except Exception as e:
            logging.error("WARN: exception in solve_approx_lp_deterministic")
            logging.exception(e)

        logging.info(f">>> N={N} B={B} ilp_feasible={ilp.feasible} lb_lp_feasible={lb_lp.feasible}"
                     f" integrality_gap={integrality_gap:.3f} approx_ratio={approx_ratio_actual:.3f}-{approx_ratio_ub:.3f}"
                     f" time_ilp={ilp.solve_time_s:.3f} time_lp={lb_lp.solve_time_s:.3f} speedup={speedup:.3f}")
