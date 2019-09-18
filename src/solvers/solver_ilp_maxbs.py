import functools
from typing import Dict, Any

import gurobipy
from gurobipy import quicksum
import numpy as np
from gurobi import GRB as GRB

import solvers.solver
from utils.graph import Graph
from utils.setup_logger import setup_logger
from utils.timer import Timer


class MaxBatchILPSolver:
    def __init__(self, g: Graph, budget: int, eps_noise=None,model_file=None, remote=False, gurobi_params: Dict[str, Any] = None, cpu_fwd_factor: int = 2):
        self.cpu_fwd_factor = cpu_fwd_factor
        self.logger = setup_logger("MaxBatchILPSolve")
        self.remote = remote
        self.profiler = functools.partial(Timer, print_results=True)
        self.gurobi_params = gurobi_params
        self.num_threads = self.gurobi_params.get('Threads', 1)
        self.model_file = model_file
        self.eps_noise = eps_noise
        self.budget = budget
        self.g: Graph = g
        self.solve_time = None
        self.init_constraints = []  # used for seeding the model

        self.m = gurobipy.Model(f"checkpointmip_gc_maxbs_{self.g.size}_{self.budget}")
        if gurobi_params is not None:
            for k, v in gurobi_params.items():
                setattr(self.m.Params, k, v)

        T = self.g.size
        CPU_VALS = list(self.g.cost_cpu.values())
        RAM_VALS = list(self.g.cost_ram.values())
        self.logger.info(f"RAM: [{np.min(RAM_VALS):.2E}, {np.max(RAM_VALS):.2E}], {np.mean(RAM_VALS):.2E} +- {np.std(RAM_VALS):.2E}")
        self.logger.info(f"CPU: [{np.min(CPU_VALS):.2E}, {np.max(CPU_VALS):.2E}], {np.mean(CPU_VALS):.2E} +- {np.std(CPU_VALS):.2E}")
        self.ram_gcd = int(max(self.g.ram_gcd(self.budget), 1))
        self.cpu_gcd = int(max(self.g.cpu_gcd(), 1))
        self.logger.info(f"ram_gcd = {self.ram_gcd} cpu_gcd = {self.cpu_gcd}")

        self.batch_size = self.m.addVar(lb=1, ub=1024 * 16, name="batch_size")
        self.R = self.m.addVars(T, T, name="R", vtype=GRB.BINARY)
        self.S = self.m.addVars(T, T, name="S", vtype=GRB.BINARY)
        self.Free_E = self.m.addVars(T, len(self.g.edge_list), name="FREE_E", vtype=GRB.BINARY)
        self.U = self.m.addVars(T, T, name="U", lb=0.0, ub=self.budget)
        for x in range(T):
            for y in range(T):
                self.m.addLConstr(self.U[x, y], GRB.GREATER_EQUAL, 0)
                self.m.addLConstr(self.U[x, y], GRB.LESS_EQUAL, float(budget) / self.ram_gcd)

    def build_model(self):
        T = self.g.size
        dict_val_div = lambda cost_dict, divisor: {k: np.ceil(v / divisor) for k, v in cost_dict.items()}
        permute_ram = dict_val_div(self.g.cost_ram, self.ram_gcd)
        budget = self.budget / self.ram_gcd

        permute_eps = lambda cost_dict, eps: {k: v * (1. + eps * np.random.randn()) for k, v in cost_dict.items()}
        permute_cpu = dict_val_div(self.g.cost_cpu, self.cpu_gcd)
        if self.eps_noise:
            permute_cpu = permute_eps(permute_cpu, self.eps_noise)

        with self.profiler("Gurobi model construction", extra_data={'T': str(T), 'budget': str(budget)}):
            with self.profiler("Objective construction", extra_data={'T': str(T), 'budget': str(budget)}):
                # define objective function
                self.m.setObjective(self.batch_size, GRB.MAXIMIZE)

            with self.profiler("Variable initialization", extra_data={'T': str(T), 'budget': str(budget)}):
                self.m.addLConstr(quicksum(self.R[t, i] for t in range(T) for i in range(t + 1, T)), GRB.EQUAL, 0)
                self.m.addLConstr(quicksum(self.S[t, i] for t in range(T) for i in range(t, T)), GRB.EQUAL, 0)
                self.m.addLConstr(quicksum(self.R[t, t] for t in range(T)), GRB.EQUAL, T)

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
                    return 1 - self.R[t, k] + self.S[t + 1, i] + quicksum(
                        self.R[t, j] for j in self.g.successors(i) if j > k)
                return 1 - self.R[t, k] + quicksum(self.R[t, j] for j in self.g.successors(i) if j > k)

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
                        self.m.addLConstr(_max_num_hazards(t, i, k) * (1 - self.Free_E[t, eidx]), GRB.GREATER_EQUAL,
                                          _num_hazards(t, i, k))
            with self.profiler("Constraint: initialize memory usage (includes spurious checkpoints)",
                               extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    self.m.addConstr(self.U[t, 0] ==
                                      self.batch_size * (self.R[t, 0] * permute_ram[0] + quicksum(
                                          self.S[t, i] * permute_ram[i] for i in range(T))), name="init_mem")
            with self.profiler("Constraint: memory recurrence", extra_data={'T': str(T), 'budget': str(budget)}):
                for t in range(T):
                    for k in range(T - 1):
                        mem_freed = quicksum(
                            permute_ram[i] * self.Free_E[t, eidx] for (eidx, i) in self.g.predecessors_indexed(k))
                        self.m.addConstr(self.U[t, k + 1] ==
                                          self.U[t, k] + self.batch_size * (
                                                      self.R[t, k + 1] * permute_ram[k + 1] - mem_freed), name="update_mem")

            if self.cpu_fwd_factor:
                with self.profiler("Constraint: recomputation overhead"):
                    compute_fwd = sum([permute_cpu[i] for i in self.g.vfwd])
                    bwd_compute = sum([permute_cpu[i] for i in self.g.v if i not in self.g.vfwd])
                    max_mem = (self.cpu_fwd_factor * compute_fwd + bwd_compute)
                    self.logger.info(f"Solver using compute overhead ceiling of {max_mem}")
                    self.m.addConstr(quicksum(self.R[t, i] * permute_cpu[i] for t in range(T) for i in range(T)) <= max_mem, name="limit_cpu")
            else:
                self.logger.info("No compute limit")

        with self.profiler("Model update"):
            self.m.update()

        if self.model_file is not None and self.g.size < 200:  # skip for big models to save runtime
            with self.profiler("Saving model", extra_data={'T': str(T), 'budget': str(budget)}):
                self.m.write(self.model_file)
        return None  # return value ensures ray remote call can be chained

    def solve(self):
        T = self.g.size
        with self.profiler('Gurobi model optimization', extra_data={'T': str(T), 'budget': str(self.budget)}):
            with Timer("ILPSolve") as solve_ilp:
                self.m.optimize()
            self.solve_time = solve_ilp.elapsed

        infeasible = (self.m.status == GRB.INFEASIBLE)
        try:
            _ = self.R[0, 0].X
            _ = self.S[0, 0].X
            _ = self.U[0, 0].X
            _ = self.batch_size.X
        except AttributeError as e:
            infeasible = True

        if infeasible:
            raise ValueError("Infeasible model, check constraints carefully. Insufficient memory?")

        Rout = np.zeros((T, T), dtype=solvers.solver.SOLVER_DTYPE)
        Sout = np.zeros((T, T), dtype=solvers.solver.SOLVER_DTYPE)
        Uout = np.zeros((T, T), dtype=solvers.solver.SOLVER_DTYPE)
        Free_Eout = np.zeros((T, len(self.g.edge_list)), dtype=solvers.solver.SOLVER_DTYPE)
        batch_size = self.batch_size.X
        try:
            for t in range(T):
                for i in range(T):
                    Rout[t][i] = int(self.R[t, i].X)
                    Sout[t][i] = int(self.S[t, i].X)
                    Uout[t][i] = self.U[t, i].X * self.ram_gcd
                for e in range(len(self.g.edge_list)):
                    Free_Eout[t][e] = int(self.Free_E[t, e].X)
        except AttributeError as e:
            logger = setup_logger("ILPSolver", self.gurobi_params.get('LogFile'))
            logger.exception(e)
            return None, None, None, None

        Rout = solvers.solver.CheckpointSolver.solve_r_opt(self.g, Sout)  # prune R using optimal recomputation solver
        return Rout, Sout, Uout, Free_Eout, batch_size
