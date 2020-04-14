import logging
import os

import cvxpy as cp
import numpy as np

from checkmate.core.dfgraph import DFGraph
from checkmate.core.utils.timer import Timer


class POETSolverCVXPY:
    def __init__(self, g: DFGraph, budget: int, cpu_cost_vec, page_in_cost_vec, page_out_cost_vec, integral=True, paging=True, remat=True):
        self.budget = budget
        self.g = g
        self.T = self.g.size

        self.R = cp.Variable((self.T, self.T), name="R", integer=integral)
        self.Sram = cp.Variable((self.T, self.T), name="S_RAM", integer=integral)
        self.Ssd = cp.Variable((self.T, self.T), name="S_SD", integer=integral)
        self.Min = cp.Variable((self.T, self.T), name="M_in", integer=integral)
        self.Mout = cp.Variable((self.T, self.T), name="M_out", integer=integral)
        self.Free_E = cp.Variable((self.T, len(self.g.edge_list)), name="FREE_E", integer=integral)
        self.U = cp.Variable((self.T, self.T), name="U")

        assert cpu_cost_vec.shape == (self.T, 1)
        objective = cp.Minimize(cp.sum(self.R @ cpu_cost_vec + self.Min @ page_in_cost_vec + self.Mout @ page_out_cost_vec))
        constraints = self.make_constraints(budget)
        if not paging:
            constraints.append(self.Mout == 0)
            constraints.append(self.Min == 0)
            constraints.append(self.Ssd == 0)

        if not remat:
            constraints.append(self.R == np.eye(self.T))

        self.problem = cp.Problem(objective, constraints)
        self.num_vars = self.problem.size_metrics.num_scalar_variables
        self.num_constraints = self.problem.size_metrics.num_scalar_eq_constr + self.problem.size_metrics.num_scalar_leq_constr

    def make_constraints(self, budget):
        constraints = []
        T = self.T
        ram_costs = self.g.cost_ram
        ram_cost_vec = np.asarray([ram_costs[i] for i in range(T)])

        with Timer("Var bounds"):
            constraints.extend([self.R >= 0, self.R <= 1])
            constraints.extend([self.Sram >= 0, self.Sram <= 1])
            constraints.extend([self.Ssd >= 0, self.Ssd <= 1])
            constraints.extend([self.Min >= 0, self.Min <= 1])
            constraints.extend([self.Mout >= 0, self.Mout <= 1])
            constraints.extend([self.Free_E >= 0, self.Free_E <= 1])
            constraints.extend([self.U >= 0, self.U <= budget])
            constraints.append(cp.diag(self.R) == 1)
            constraints.append(cp.upper_tri(self.R) == 0)
            constraints.append(cp.diag(self.Sram) == 0)
            constraints.append(cp.upper_tri(self.Sram) == 0)
            constraints.append(cp.diag(self.Ssd) == 0)
            constraints.append(cp.upper_tri(self.Ssd) == 0)
            constraints.append(cp.upper_tri(self.Min) == 0)
            constraints.append(cp.upper_tri(self.Mout) == 0)

        with Timer("Correctness constraints"):
            # ensure all computations are possible
            for (u, v) in self.g.edge_list:
                constraints.append(self.R[:, v] <= self.R[:, u] + self.Sram[:, u])

            # ensure all checkpoints are in memory
            constraints.append(self.Sram[1:, :] <= self.R[:-1, :] + self.Sram[:-1, :] + self.Min[:-1, :])
            constraints.append(self.Ssd[1:, :] <= self.Ssd[:-1, :] + self.Mout[:-1, :])
            constraints.append(self.Min <= self.Ssd)
            constraints.append(self.Mout <= self.Sram)

        with Timer("Free_E constraints"):
            # Constraint: sum_k Free_{t,i,k} <= 1
            for i in range(T):
                frees = [self.Free_E[:, eidx] for eidx, (j, _) in enumerate(self.g.edge_list) if i == j]
                if frees:
                    constraints.append(cp.sum(frees, axis=0) <= 1)

            # Constraint: Free_{t,i,k} <= 1 - S_{t+1, i}
            for eidx, (i, k) in enumerate(self.g.edge_list):
                constraints.append(self.Free_E[:-1, eidx] + self.Sram[1:, i] <= 1)

            # Constraint: Free_{t,i,k} <= 1 - R_{t, j}
            for eidx, (i, k) in enumerate(self.g.edge_list):
                for j in self.g.successors(i):
                    if j > k:
                        constraints.append(self.Free_E[:, eidx] + self.R[:, j] <= 1)

        with Timer("U constraints"):
            constraints.append(self.U[:, 0] == self.R[:, 0] * ram_costs[0] + ram_cost_vec @ self.Sram.T)
            for k in range(T - 1):
                mem_freed = cp.sum([ram_costs[i] * self.Free_E[:, eidx] for (eidx, i) in self.g.predecessors_indexed(k)])
                constraints.append(self.U[:, k + 1] == self.U[:, k] + self.R[:, k + 1] * ram_costs[k + 1] - mem_freed)
        return constraints

    def solve(self, solver_override=None, verbose=False, num_threads=os.cpu_count()):
        installed_solvers = cp.installed_solvers()
        with Timer("Solve", print_results=verbose) as solve_timer:
            if solver_override is not None:
                self.problem.solve(verbose=verbose, solver=solver_override)
            elif "MOSEK" in installed_solvers:
                self.problem.solve(verbose=verbose, solver=cp.MOSEK)
            elif "GUROBI" in installed_solvers:
                self.problem.solve(verbose=verbose, solver=cp.GUROBI)
            elif "CBC" in installed_solvers:
                self.problem.solve(verbose=verbose, solver=cp.CBC, numberThreads=num_threads)
            else:
                self.problem.solve(verbose=verbose)
        self.solve_time = solve_timer.elapsed
        if self.problem.status in ["infeasible", "unbounded"]:
            raise ValueError("Model infeasible")
        return self.R.value, self.Sram.value, self.Ssd.value, self.Min.value, self.Mout.value, self.U.value, self.Free_E.value


def extract_costs_from_dfgraph(g: DFGraph, sd_card_multipler=5.0):
    T = g.size
    cpu_cost_vec = np.asarray([g.cost_cpu[i] for i in range(T)])[np.newaxis, :].T
    page_in_cost_vec = cpu_cost_vec * sd_card_multipler
    page_out_cost_vec = cpu_cost_vec * sd_card_multipler
    return cpu_cost_vec, page_in_cost_vec, page_out_cost_vec


def solve_poet_cvxpy(g, budget, cpu_cost, page_in_cost, page_out_cost, integral=True, solver_override=None, verbose=False,
                     paging=True, remat=True):
    poet_solver = POETSolverCVXPY(g, budget, cpu_cost, page_in_cost, page_out_cost, integral=integral, paging=paging, remat=remat)
    try:
        r, s_ram, s_sd, m_in, m_out, u, free_e = poet_solver.solve(solver_override=solver_override, verbose=verbose)
        lp_feasible = True
        # r_ = solve_r_opt(g, s)
        # schedule, aux_data = schedule_from_rs(g, r, s_ram)
    except ValueError as e:
        logging.exception(e)
        r, s_ram, s_sd, m_in, m_out, u, free_e = (None,) * 7
        lp_feasible, schedule, aux_data = False, None, None
    return r, s_ram, s_sd, m_in, m_out, u
    # return ScheduledResult(
    #     solve_strategy=SolveStrategy.OPTIMAL_ILP_GC,
    #     solver_budget=budget,
    #     feasible=lp_feasible,
    #     schedule=schedule,
    #     schedule_aux_data=aux_data,
    #     solve_time_s=poet_solver.solve_time,
    #     ilp_aux_data=ILPAuxData(
    #         U=u,
    #         Free_E=free_e,
    #         ilp_approx=False,
    #         ilp_time_limit=None,
    #         ilp_eps_noise=0.0,
    #         ilp_num_constraints=poet_solver.num_vars,
