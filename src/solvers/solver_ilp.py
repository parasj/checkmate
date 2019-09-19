from typing import Dict, Any

import gurobipy
import numpy as np
import ray
from gurobi import GRB as GRB

import remat.core.solvers.common
import solvers.solver
from remat.core.dfgraph import DFGraph
from utils.setup_logger import setup_logger
from remat.core.utils.timer import Timer


class ILPSolver:
    def __init__(self, g: DFGraph, budget: int, eps_noise=None, seed_s=None, model_file=None, remote=True,
                 gurobi_params: Dict[str, Any] = None):
        self.remote = remote
        self.profiler = Timer if not self.remote else ray.profile
        self.gurobi_params = gurobi_params
        self.num_threads = self.gurobi_params.get('Threads', 1)
        self.model_file = model_file
        self.seed_s = seed_s
        self.eps_noise = eps_noise
        self.budget = budget
        self.g: DFGraph = g
        self.solve_time = None

        self.init_constraints = []  # used for seeding the model

        self.m = gurobipy.Model("checkpointmip_gc_{}_{}".format(self.g.size, self.budget))
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        T = self.g.size
        self.ram_gcd = self.g.ram_gcd(self.budget)
        self.R = self.m.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.m.addVars(T, T, name="S", vtype=GRB.BINARY)
        self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.BINARY)
        self.U = self.m.addVars(T, T, name="U", lb=0.0, ub=float(budget) / self.ram_gcd)
        for x in range(T):
            for y in range(T):
                self.m.addLConstr(self.U[x, y], GRB.GREATER_EQUAL, 0)
                self.m.addLConstr(self.U[x, y], GRB.LESS_EQUAL, float(budget) / self.ram_gcd)

    def build_model(self):
        T = self.g.size
        dict_val_div = lambda cost_dict, divisor: {k: v / divisor for k, v in cost_dict.items()}
        permute_ram = dict_val_div(self.g.cost_ram, self.ram_gcd)
        budget = self.budget / self.ram_gcd

        permute_eps = lambda cost_dict, eps: {k: v * (1. + eps * np.random.randn()) for k, v in cost_dict.items()}
        permute_cpu = dict_val_div(self.g.cost_cpu, self.g.cpu_gcd())
        if self.eps_noise:
            permute_cpu = permute_eps(permute_cpu, self.eps_noise)

        with self.profiler("Gurobi model construction", extra_data={'T': str(T), 'budget': str(budget)}):
            with self.profiler("Objective construction", extra_data={'T': str(T), 'budget': str(budget)}):
                # seed solver with a baseline strategy
                if self.seed_s is not None:
                    for x in range(T):
                        for y in range(T):
                            if self.seed_s[x, y] < 1:
                                self.init_constraints.append(self.m.addLConstr(self.S[x, y], GRB.EQUAL, 0))
                    self.m.update()

                # define objective function
                self.m.setObjective(gurobipy.quicksum(
                    self.R[t, i] * permute_cpu[i] for t in range(T) for i in range(T)),
                    GRB.MINIMIZE)

            with self.profiler("Variable initialization", extra_data={'T': str(T), 'budget': str(budget)}):
                self.m.addLConstr(gurobipy.quicksum(self.R[t, i] for t in range(T) for i in range(t + 1, T)), GRB.EQUAL,
                                  0)
                self.m.addLConstr(gurobipy.quicksum(self.S[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                self.m.addLConstr(gurobipy.quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T)

            with self.profiler("Correctness constraints", extra_data={'T': str(T), 'budget': str(budget)}):
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
                    return 1 - self.R[t, k] + self.S[t + 1, i] + gurobipy.quicksum(
                        self.R[t, j] for j in self.g.successors(i) if j > k)
                return 1 - self.R[t, k] + gurobipy.quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)

            def _max_num_hazards(t, i, k):
                num_uses_after_k = sum(1 for j in self.g.successors(i) if j > k)
                if t + 1 < T:
                    return 2 + num_uses_after_k
                return 1 + num_uses_after_k

            with self.profiler("Constraint: upper bound for 1 - Free_E",
                               extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for eidx, (i, k) in enumerate(self.g.edge_list):
                        self.m.addLConstr(1 - self.Free_E[t, eidx], GRB.LESS_EQUAL, _num_hazards(t, i, k))
            with self.profiler("Constraint: lower bound for 1 - Free_E",
                               extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for eidx, (i, k) in enumerate(self.g.edge_list):
                        self.m.addLConstr(_max_num_hazards(t, i, k) * (1 - self.Free_E[t, eidx]),
                                          GRB.GREATER_EQUAL, _num_hazards(t, i, k))
            with self.profiler("Constraint: initialize memory usage (includes spurious checkpoints)",
                               extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    self.m.addLConstr(self.U[t, 0], GRB.EQUAL,
                                      self.R[t, 0] * permute_ram[0] + gurobipy.quicksum(
                                          self.S[t, i] * permute_ram[i] for i in range(T)))
            with self.profiler("Constraint: memory recurrence", extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for k in range(T - 1):
                        mem_freed = gurobipy.quicksum(
                            permute_ram[i] * self.Free_E[t, eidx] for (eidx, i) in self.g.predecessors_indexed(k))
                        self.m.addLConstr(self.U[t, k + 1], GRB.EQUAL,
                                          self.U[t, k] + self.R[t, k + 1] * permute_ram[k + 1] - mem_freed)

        if self.model_file is not None and self.g.size < 200:  # skip for big models to save runtime
            with self.profiler("Saving model", extra_data={'T': str(T), 'budget': str(budget)}):
                self.m.write(self.model_file)
        return None  # return value ensures ray remote call can be chained

    def solve(self):
        T = self.g.size
        with self.profiler('Gurobi model optimization', extra_data={'T': str(T), 'budget': str(self.budget)}):
            if self.seed_s is not None:
                self.m.Params.TimeLimit = 300
                self.m.optimize()
                if self.m.status == GRB.INFEASIBLE:
                    print(f"Infeasible ILP seed at budget {self.budget:.2E}")
                self.m.remove(self.init_constraints)
            self.m.Params.TimeLimit = self.gurobi_params.get('TimeLimit', 3600)  # todo remove this check, and default to np.inf
            self.m.message("\n\nRestarting solve\n\n")
            with Timer("ILPSolve") as solve_ilp:
                self.m.optimize()
            self.solve_time = solve_ilp.elapsed

        infeasible = (self.m.status == GRB.INFEASIBLE)
        if infeasible:
            raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")
        
        if self.m.solCount < 1:
            raise ValueError(f"Model status is {self.m.status} (not infeasible), but solCount is {self.m.solCount}")

        Rout = np.zeros((T, T), dtype=remat.core.solvers.common.SOLVER_DTYPE)
        Sout = np.zeros((T, T), dtype=remat.core.solvers.common.SOLVER_DTYPE)
        Uout = np.zeros((T, T), dtype=remat.core.solvers.common.SOLVER_DTYPE)
        Free_Eout = np.zeros((T, len(self.g.edge_list)), dtype=remat.core.solvers.common.SOLVER_DTYPE)
        try:
            for t in range(T):
                for i in range(T):
                    try:
                        Rout[t][i] = int(self.R[t, i].X)
                    except (AttributeError, TypeError) as e:
                        Rout[t][i] = int(self.R[t, i])

                    try:
                        Sout[t][i] = int(self.S[t, i])
                    except (AttributeError, TypeError) as e:
                        Sout[t][i] = int(self.S[t, i].X)

                    try:
                        Uout[t][i] = self.U[t, i].X * self.ram_gcd
                    except (AttributeError, TypeError) as e:
                        Uout[t][i] = self.U[t, i] * self.ram_gcd
                for e in range(len(self.g.edge_list)):
                    try:
                        Free_Eout[t][e] = int(self.Free_E[t, e].X)
                    except (AttributeError, TypeError) as e:
                        Free_Eout[t][e] = int(self.Free_E[t, e])
        except AttributeError as e:
            logger = setup_logger("ILPSolver", self.gurobi_params.get('LogFile'))
            logger.exception(e)
            return None, None, None, None

        Rout = solvers.solver.CheckpointSolver.solve_r_opt(self.g, Sout)  # prune R using optimal recomputation solver
        return Rout, Sout, Uout, Free_Eout
