from itertools import chain
import pickle
from typing import NamedTuple, List, Optional

import numpy as np
import ray

from evaluation.util.solve_strategy import SolveStrategy
from solvers.scheduler import ScheduleBuilder
from remat.core.schedule import Schedule
from remat.core.graph import Graph


class PartialRSResult(NamedTuple):
    solve_strategy: SolveStrategy
    cost_file: str
    input_shape: List[int]  # input_shape[0] is batch size
    R: np.ndarray
    S: np.ndarray
    U: Optional[np.ndarray] = None
    Free_E: Optional[np.ndarray] = None
    solver_budget: float = None
    solve_time_s: float = None
    ilp_feasible: bool = True
    ilp_num_variables: int = None
    ilp_num_constraints: int = None


class RSResult(NamedTuple):
    solve_strategy: SolveStrategy
    cost_file: str
    input_shape: List[int]  # input_shape[0] is batch size
    R: np.ndarray
    S: np.ndarray
    U: Optional[np.ndarray]
    Free_E: Optional[np.ndarray]
    solver_budget: float
    solve_time_s: float
    ilp_feasible: bool
    ilp_num_variables: int
    ilp_num_constraints: int
    schedule: Schedule
    peak_ram: int
    activation_ram: int
    cpu: int
    mem_grid: np.ndarray
    mem_timeline: List[int]
    schedule_time_s: float

    def dumps(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(serialized_result: bytes):
        return pickle.loads(serialized_result)

    @staticmethod
    def verify_solution(G: Graph, R: np.ndarray, S: np.ndarray):
        if R is None or S is None:
            return None, None, None, None, None, None
        T = G.size

        def _used_after(t_, u_, i_):
            """Returns True if v_u is used after v_i in stage t"""
            is_retained_snapshot = t_ < T - 1 and S[t_ + 1, u_] == 1
            is_used_by_successor = not all([R[t_, v] == 0 or v <= i_ for v in G.successors(u_)])
            return is_retained_snapshot or is_used_by_successor

        # compute last usage to determine whether to update auxiliary variables
        last_used = {i: max([t for t in range(T) if R[t, i] == 1]) for i in range(T)}

        mem_usage = np.zeros((T, T), dtype=np.int)
        # Schedule builder accounts for all memory, including fixed memory
        sb = ScheduleBuilder(G, verbosity=1)
        for t in range(T):
            # Free unused checkpoints
            for i in filter(lambda x: sb.is_op_cached(x), range(T)):
                if not _used_after(t, i, i):
                    sb.deallocate_register(i)

            for i in range(T):
                if R[t, i] == 1:
                    sb.run_operator(i, last_used[i] == t)
                mem_usage[t, i] = sb.current_mem() + G.cost_ram_fixed

                # Free memory
                for u in filter(lambda x: sb.is_op_cached(x), chain(G.predecessors(i), [i])):
                    if not _used_after(t, u, i):
                        sb.deallocate_register(u)
        max_ram = sb.max_ram + G.cost_ram_fixed
        ram_timeline = [mem + G.cost_ram_fixed for mem in sb.ram_timeline]
        return sb.schedule, max_ram, max_ram - G.cost_ram_fixed, sb.total_cpu, mem_usage, ram_timeline

    @staticmethod
    @ray.remote(num_cpus=1, num_return_vals=5)
    def remote_verify_solution(G: Graph, R: np.ndarray, S: np.ndarray):
        return RSResult.verify_solution(G, R, S)

    @staticmethod
    def verify_memory(G: Graph, R: np.ndarray, S: np.ndarray):
        """
        Lightweight version of verify_solution that only computes maximum memory usage
        """
        mem_usage = np.zeros((G.size, G.size), dtype=np.int)
        M = G.cost_ram_np

        check_free = []
        for k in range(G.size):
            check_free.append(list(G.predecessors(k)))
            check_free[k].append(k)

        hazards = []
        for k in range(G.size):
            hazards.append(np.array(list(G.successors(k)), dtype=np.int))

        for t in range(G.size):
            # Memory usage at start of stage and including fixed memory
            mem_usage[t, 0] = R[t, 0] * M[0] + G.cost_ram_fixed
            for i in range(G.size):
                NumUses = (t + 1 < G.size and S[t + 1, i]) + np.sum(R[t, hazards[i]])
                mem_usage[t, 0] += M[i] * S[t, i] * (NumUses > 0)

            for k in range(t):
                mem_usage[t, k + 1] = mem_usage[t, k] + R[t, k + 1] * M[k + 1]

                if not R[t, k]:
                    continue

                # Free dependencies
                for i in check_free[k]:
                    if t + 1 == G.size or not S[t + 1, i]:
                        if not np.any(R[t, hazards[i][hazards[i] > k]]):
                            mem_usage[t, k + 1] -= M[i]

        return mem_usage.max(), mem_usage

    @staticmethod
    def test_verify_memory(G: Graph, R: np.ndarray, S: np.ndarray,
                           U: np.ndarray = None, tol: float = 1e-6):
        _, mem_usage = RSResult.verify_memory(G, R, S)
        _, __, ___, mem_usage_verify, ____ = RSResult.verify_solution(G, R, S)

        # Check if memory measures match
        if not np.allclose(mem_usage, mem_usage_verify):
            print("WARNING: Fast RAM computation and verifier mismatch")

        if U is not None and not np.allclose(U, mem_usage_verify):
            print("WARNING: Solver U and verifier RAM mismatch")
            if np.any(U + tol < mem_usage_verify):
                print("ERROR: Solver U does not dominate verifier RAM")

                for t in range(G.size):
                    if np.any(U[t] + tol < mem_usage[t]):
                        print("  Lack of domination at t={}".format(t))

        if U is not None and not np.allclose(U, mem_usage):
            print("WARNING: Solver U and fast RAM computation mismatch")
            if np.any(U + tol < mem_usage):
                print("ERROR: Solver U does not dominate fast RAM computation")

                for t in range(G.size):
                    if np.any(U[t] + tol < mem_usage[t]):
                        print("  Lack of domination at t={}".format(t))

    @staticmethod
    def plot(G: Graph, R: np.ndarray, S: np.ndarray, U: np.ndarray = None, plot_mem_usage: bool = False,
             timeline: np.ndarray = None, save_file: str = None, show: bool = False, plt=None):
        if plt is None:
            import matplotlib.pyplot as plt
        if plot_mem_usage:
            fig, axs = plt.subplots(1, 5)
            _, mem_usage = RSResult.verify_memory(G, R, S)
            _, _, _, _, mem_usage_verify, _ = RSResult.verify_solution(G, R, S)
            vmax = max(np.max(mem_usage), np.max(mem_usage_verify))
            vmax = max(vmax, np.max(U)) if U is not None else vmax

            # Plot fast checker memory usage
            axs[2].invert_yaxis()
            axs[2].pcolormesh(mem_usage, cmap="Greys", vmin=0, vmax=vmax)  # edgecolors='0.5', linewidth=1,
            axs[2].set_title("Memory usage (fast)")

            # Plot slow verifier memory usage
            axs[3].invert_yaxis()
            axs[3].pcolormesh(mem_usage_verify, cmap="Greys", vmin=0, vmax=vmax)
            axs[3].set_title("Memory usage (verifier)")

            # Plot solver memory usage variables
            axs[4].invert_yaxis()
            axs[4].set_title("Memory usage (solved)")
            if U is not None:
                axs[4].pcolormesh(U, cmap="Greys", vmin=0, vmax=vmax)

            fig.set_size_inches(28, 6)
        elif timeline is not None:
            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(24, 6)
            axs[2].plot(timeline)
        else:
            fig, axs = plt.subplots(1, 2)
            fig.set_size_inches(18, 6)

        axs[0].invert_yaxis()
        axs[0].pcolormesh(R, cmap="Greys")
        axs[0].set_title("R")

        axs[1].invert_yaxis()
        axs[1].pcolormesh(S, cmap="Greys")
        axs[1].set_title("S")

        if show:
            plt.show()
        if save_file:
            # print("Saving schedule plot to {}".format(save_file))
            fig.savefig(save_file)
            plt.close(fig)


