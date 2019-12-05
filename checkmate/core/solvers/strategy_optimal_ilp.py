import logging
import math
import os
from typing import Dict, Any, Optional

import numpy as np

# noinspection PyPackageRequirements
from gurobipy import GRB, Model, quicksum

import checkmate.core
from checkmate.core.dfgraph import DFGraph
from checkmate.core.enum_strategy import SolveStrategy, ImposedSchedule
from checkmate.core.schedule import ScheduledResult, ILPAuxData
from checkmate.core.utils.definitions import PathLike
from checkmate.core.utils.scheduler import schedule_from_rs
from checkmate.core.utils.solver_common import solve_r_opt
from checkmate.core.utils.timer import Timer


class ILPSolver:
    def __init__(
        self,
        g: DFGraph,
        budget: int,
        eps_noise=None,
        seed_s=None,
        integral=True,
        imposed_schedule: ImposedSchedule = ImposedSchedule.FULL_SCHEDULE,
        solve_r=True,
        write_model_file: Optional[PathLike] = None,
        gurobi_params: Dict[str, Any] = None,
    ):
        self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT = 300  # todo (paras): read this from gurobi_params
        self.gurobi_params = gurobi_params
        self.num_threads = self.gurobi_params.get("Threads", 1)
        self.model_file = write_model_file
        self.seed_s = seed_s
        self.integral = integral
        self.imposed_schedule = imposed_schedule
        self.solve_r = solve_r
        self.eps_noise = eps_noise
        self.budget = budget
        self.g: DFGraph = g
        self.solve_time = None

        if not self.integral:
            assert not self.solve_r, "Can't solve for R if producing a fractional solution"

        self.init_constraints = []  # used for seeding the model

        self.m = Model("checkpointmip_gc_{}_{}".format(self.g.size, self.budget))
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        T = self.g.size
        self.ram_gcd = self.g.ram_gcd(self.budget)
        if self.integral:
            self.R = self.m.addVars(T, T, name="R", vtype=GRB.BINARY)
            self.S = self.m.addVars(T, T, name="S", vtype=GRB.BINARY)
            self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.BINARY)
        else:
            self.R = self.m.addVars(T, T, name="R", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
            self.S = self.m.addVars(T, T, name="S", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
            self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0)
        gcd = float(budget) / self.ram_gcd
        self.U = self.m.addVars(T, T, name="U", lb=0.0, ub=gcd)
        for x in range(T):
            for y in range(T):
                self.m.addLConstr(self.U[x, y], GRB.GREATER_EQUAL, 0)
                self.m.addLConstr(self.U[x, y], GRB.LESS_EQUAL, float(budget) / self.ram_gcd)

    def build_model(self):
        T = self.g.size
        dict_val_div = lambda cost_dict, divisor: {k: v / divisor for k, v in cost_dict.items()}
        permute_ram = dict_val_div(self.g.cost_ram, self.ram_gcd)
        budget = self.budget / self.ram_gcd

        permute_eps = lambda cost_dict, eps: {k: v * (1.0 + eps * np.random.randn()) for k, v in cost_dict.items()}
        permute_cpu = dict_val_div(self.g.cost_cpu, self.g.cpu_gcd())
        if self.eps_noise:
            permute_cpu = permute_eps(permute_cpu, self.eps_noise)

        with Timer("Gurobi model construction", extra_data={"T": str(T), "budget": str(budget)}):
            with Timer("Objective construction", extra_data={"T": str(T), "budget": str(budget)}):
                # seed solver with a baseline strategy
                if self.seed_s is not None:
                    for x in range(T):
                        for y in range(T):
                            if self.seed_s[x, y] < 1:
                                self.init_constraints.append(self.m.addLConstr(self.S[x, y], GRB.EQUAL, 0))
                    self.m.update()

                # define objective function
                self.m.setObjective(quicksum(self.R[t, i] * permute_cpu[i] for t in range(T) for i in range(T)), GRB.MINIMIZE)

            with Timer("Variable initialization", extra_data={"T": str(T), "budget": str(budget)}):
                if self.imposed_schedule == ImposedSchedule.FULL_SCHEDULE:
                    self.m.addLConstr(quicksum(self.R[t, i] for t in range(T) for i in range(t + 1, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.S[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                    self.m.addLConstr(quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T)
                elif self.imposed_schedule == ImposedSchedule.COVER_ALL_NODES:
                    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                    for i in range(T):
                        self.m.addLConstr(quicksum(self.R[t, i] for t in range(T)), GRB.GREATER_EQUAL, 1)
                elif self.imposed_schedule == ImposedSchedule.COVER_LAST_NODE:
                    self.m.addLConstr(quicksum(self.S[0, i] for i in range(T)), GRB.EQUAL, 0)
                    # note: the integrality gap is very large as this constraint
                    # is only applied to the last node (last column of self.R).
                    self.m.addLConstr(quicksum(self.R[t, T - 1] for t in range(T)), GRB.GREATER_EQUAL, 1)

            with Timer("Correctness constraints", extra_data={"T": str(T), "budget": str(budget)}):
                # ensure all checkpoints are in memory
                for t in range(T - 1):
                    for i in range(T):
                        self.m.addLConstr(self.S[t + 1, i], GRB.LESS_EQUAL, self.S[t, i] + self.R[t, i])
                # ensure all computations are possible
                for (u, v) in self.g.edge_list:
                    for t in range(T):
                        self.m.addLConstr(self.R[t, v], GRB.LESS_EQUAL, self.R[t, u] + self.S[t, u])

            # define memory constraints
            def _num_hazards(t, i, k):
                if t + 1 < T:
                    return 1 - self.R[t, k] + self.S[t + 1, i] + quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)
                return 1 - self.R[t, k] + quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)

            def _max_num_hazards(t, i, k):
                num_uses_after_k = sum(1 for j in self.g.successors(i) if j > k)
                if t + 1 < T:
                    return 2 + num_uses_after_k
                return 1 + num_uses_after_k

            with Timer("Constraint: upper bound for 1 - Free_E", extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for eidx, (i, k) in enumerate(self.g.edge_list):
                        self.m.addLConstr(1 - self.Free_E[t, eidx], GRB.LESS_EQUAL, _num_hazards(t, i, k))
            with Timer("Constraint: lower bound for 1 - Free_E", extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for eidx, (i, k) in enumerate(self.g.edge_list):
                        self.m.addLConstr(
                            _max_num_hazards(t, i, k) * (1 - self.Free_E[t, eidx]), GRB.GREATER_EQUAL, _num_hazards(t, i, k)
                        )
            with Timer(
                "Constraint: initialize memory usage (includes spurious checkpoints)",
                extra_data={"T": str(T), "budget": str(budget)},
            ):
                for t in range(T):
                    self.m.addLConstr(
                        self.U[t, 0],
                        GRB.EQUAL,
                        self.R[t, 0] * permute_ram[0] + quicksum(self.S[t, i] * permute_ram[i] for i in range(T)),
                    )
            with Timer("Constraint: memory recurrence", extra_data={"T": str(T), "budget": str(budget)}):
                for t in range(T):
                    for k in range(T - 1):
                        mem_freed = quicksum(
                            permute_ram[i] * self.Free_E[t, eidx] for (eidx, i) in self.g.predecessors_indexed(k)
                        )
                        self.m.addLConstr(
                            self.U[t, k + 1], GRB.EQUAL, self.U[t, k] + self.R[t, k + 1] * permute_ram[k + 1] - mem_freed
                        )

        if self.model_file is not None and self.g.size < 200:  # skip for big models to save runtime
            with Timer("Saving model", extra_data={"T": str(T), "budget": str(budget)}):
                self.m.write(self.model_file)
        return None  # return value ensures ray remote call can be chained

    def solve(self):
        T = self.g.size
        with Timer("Gurobi model optimization", extra_data={"T": str(T), "budget": str(self.budget)}):
            if self.seed_s is not None:
                self.m.Params.TimeLimit = self.GRB_CONSTRAINED_PRESOLVE_TIME_LIMIT
                self.m.optimize()
                if self.m.status == GRB.INFEASIBLE:
                    print(f"Infeasible ILP seed at budget {self.budget:.2E}")
                self.m.remove(self.init_constraints)
            self.m.Params.TimeLimit = self.gurobi_params.get("TimeLimit", 0)
            self.m.message("\n\nRestarting solve\n\n")
            with Timer("ILPSolve") as solve_ilp:
                self.m.optimize()
            self.solve_time = solve_ilp.elapsed

        infeasible = self.m.status == GRB.INFEASIBLE
        if infeasible:
            raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")

        if self.m.solCount < 1:
            raise ValueError(f"Model status is {self.m.status} (not infeasible), but solCount is {self.m.solCount}")

        Rout = np.zeros((T, T), dtype=checkmate.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Sout = np.zeros((T, T), dtype=checkmate.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Uout = np.zeros((T, T), dtype=checkmate.core.utils.solver_common.SOLVER_DTYPE if self.integral else np.float)
        Free_Eout = np.zeros((T, len(self.g.edge_list)), dtype=checkmate.core.utils.solver_common.SOLVER_DTYPE)
        solver_dtype_cast = int if self.integral else float
        try:
            for t in range(T):
                for i in range(T):
                    try:
                        Rout[t][i] = solver_dtype_cast(self.R[t, i].X)
                    except (AttributeError, TypeError) as e:
                        Rout[t][i] = solver_dtype_cast(self.R[t, i])

                    try:
                        Sout[t][i] = solver_dtype_cast(self.S[t, i])
                    except (AttributeError, TypeError) as e:
                        Sout[t][i] = solver_dtype_cast(self.S[t, i].X)

                    try:
                        Uout[t][i] = self.U[t, i].X * self.ram_gcd
                    except (AttributeError, TypeError) as e:
                        Uout[t][i] = self.U[t, i] * self.ram_gcd
                for e in range(len(self.g.edge_list)):
                    try:
                        Free_Eout[t][e] = solver_dtype_cast(self.Free_E[t, e].X)
                    except (AttributeError, TypeError) as e:
                        Free_Eout[t][e] = solver_dtype_cast(self.Free_E[t, e])
        except AttributeError as e:
            logging.exception(e)
            return None, None, None, None

        # prune R using closed-form solver
        if self.solve_r and self.integral:
            Rout = solve_r_opt(self.g, Sout)

        return Rout, Sout, Uout, Free_Eout


def solve_ilp_gurobi(
    g: DFGraph,
    budget: int,
    seed_s: Optional[np.ndarray] = None,
    approx=True,
    imposed_schedule: ImposedSchedule = ImposedSchedule.FULL_SCHEDULE,
    solve_r=True,
    time_limit: Optional[int] = None,
    write_log_file: Optional[PathLike] = None,
    print_to_console=True,
    write_model_file: Optional[PathLike] = None,
    eps_noise=0.01,
    solver_cores=os.cpu_count(),
):
    """
    Memory-accurate solver with garbage collection.
    :param g: DFGraph -- graph definition extracted from model
    :param budget: int -- budget constraint for solving
    :param seed_s: np.ndarray -- optional parameter to set warm-start for solver, defaults to empty S
    :param approx: bool -- set true to return as soon as a solution is found that is within 1% of optimal
    :param imposed_schedule -- selects a set of constraints on R and S that impose a schedule or require some nodes to be computed
    :param solve_r -- if set, solve for the optimal R 
    :param time_limit: int -- time limit for solving in seconds
    :param write_log_file: if set, log gurobi to this file
    :param print_to_console: if set, print gurobi logs to the console
    :param write_model_file: if set, write output model file to this location
    :param eps_noise: float -- if set, inject epsilon noise into objective weights, default 0.5%
    :param solver_cores: int -- if set, use this number of cores for ILP solving
    """
    param_dict = {
        "LogToConsole": 1 if print_to_console else 0,
        "LogFile": str(write_log_file) if write_log_file is not None else "",
        "Threads": solver_cores,
        "TimeLimit": math.inf if time_limit is None else time_limit,
        "OptimalityTol": 1e-2 if approx else 1e-4,
        "IntFeasTol": 1e-3 if approx else 1e-5,
        "Presolve": 2,
        "StartNodeLimit": 10000000,
    }
    ilpsolver = ILPSolver(
        g,
        budget,
        gurobi_params=param_dict,
        seed_s=seed_s,
        eps_noise=eps_noise,
        imposed_schedule=imposed_schedule,
        solve_r=solve_r,
        write_model_file=write_model_file,
    )
    ilpsolver.build_model()
    try:
        r, s, u, free_e = ilpsolver.solve()
        ilp_feasible = True
    except ValueError as e:
        logging.exception(e)
        r, s, u, free_e = (None, None, None, None)
        ilp_feasible = False
    ilp_aux_data = ILPAuxData(
        U=u,
        Free_E=free_e,
        ilp_approx=approx,
        ilp_time_limit=time_limit,
        ilp_eps_noise=eps_noise,
        ilp_num_constraints=ilpsolver.m.numConstrs,
        ilp_num_variables=ilpsolver.m.numVars,
        ilp_imposed_schedule=imposed_schedule,
    )
    schedule, aux_data = schedule_from_rs(g, r, s)
    return ScheduledResult(
        solve_strategy=SolveStrategy.OPTIMAL_ILP_GC,
        solver_budget=budget,
        feasible=ilp_feasible,
        schedule=schedule,
        schedule_aux_data=aux_data,
        solve_time_s=ilpsolver.solve_time,
        ilp_aux_data=ilp_aux_data,
    )
