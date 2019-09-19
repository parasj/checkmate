import math
import os
import pathlib
import urllib.request

import numpy as np
import pandas as pd

from remat.core.dfgraph import DFGraph
from remat.core.solvers.common import setup_implied_s_backwards, gen_s_matrix_fixed_checkpoints, solve_r_opt
from solvers.solver_ilp import ILPSolver
from utils.setup_logger import setup_logger


class CheckpointSolver:
    @staticmethod
    def solve_r_opt(G: DFGraph, S: np.ndarray):
        return solve_r_opt(G, S)

    @staticmethod
    def schedule_ilp_gurobi(g: DFGraph, budget: int, seed_s: np.ndarray = None, approx: bool = True, time_limit=None,
                            log_file=None, print_to_console=True, model_file=None,
                            remote=False, eps_noise=0.01, solver_cores=1):
        """
        Memory-accurate solver with garbage collection.
        :param budget: int -- budget constraint for solving
        :param seed_s: np.ndarray -- optional parameter to set warm-start for solver, defaults to empty S
        :param approx: bool -- set true to return as soon as a solution is found that is within 1% of optimal
        :param time_limit: int -- time limit for solving in seconds
        :param log_file: if set, log gurobi to this file
        :param print_to_console: if set, print gurobi logs to the console
        :param model_file: if set, write output model file to this location
        :param remote: bool -- if true, run using Ray configuration
        :param eps_noise: float -- if set, inject epsilon noise into objective weights, default 0.5%
        :param solver_cores: int -- if set, use this number of cores for ILP solving
        """
        param_dict = {'LogToConsole': 1 if print_to_console else 0,
                      'LogFile': log_file if log_file is not None else "",
                      'Threads': solver_cores,
                      'TimeLimit': math.inf if time_limit is None else time_limit,
                      'OptimalityTol': 1e-2 if approx else 1e-4,
                      'IntFeasTol': 1e-3 if approx else 1e-5,
                      'Presolve': 2,
                      'StartNodeLimit': 10000000}
        ilpsolver = ILPSolver(g, budget, gurobi_params=param_dict, seed_s=seed_s, eps_noise=eps_noise, remote=remote,
                              model_file=model_file)
        ilpsolver.build_model()
        try:
            sol = ilpsolver.solve()
            ilp_feasible = True
        except ValueError as exc:
            print("ValueError in ILP solve!", exc)
            sol = (None, None, None, None)
            ilp_feasible = False
        return_vals = {"solve_time_s": ilpsolver.solve_time,
                       "ilp_feasible": ilp_feasible,
                       "ilp_num_variables": ilpsolver.m.numVars,
                       "ilp_num_constraints": ilpsolver.m.numConstrs}
        return sol, return_vals

    @staticmethod
    def schedule_griewank(g: DFGraph, budget: int):
        S = np.zeros((g.size, g.size), dtype=np.int32)
        S = setup_implied_s_backwards(g, S)
        np.fill_diagonal(S[1:], 1)

        ap_points = list(sorted(g.checkpoint_set))
        metaTfwd = len(ap_points)
        ap_points = ap_points + [g.forward_to_backward(p) for p in reversed(ap_points)]
        meta_to_real_v = {ap_points.index(ap_point): ap_point for ap_point in ap_points}
        try:
            regranges_all = CheckpointSolver.load_griewank(metaTfwd)
        except Exception as e:
            logger = setup_logger("griewank")
            logger.exception(e)
            return None, None
        if regranges_all is None:
            return None, None
        regranges = regranges_all[regranges_all['budget'] == budget]
        if len(regranges.index) < 1:
            return None, None

        def map_time(_t: int) -> int: return min(meta_to_real_v.get(_t, np.inf), g.size)
        for index, reg_range in regranges.iterrows():
            for t in range(map_time(reg_range['timestart']), map_time(reg_range['timeend'] + 1)):
                if reg_range['nodeid'] > 0:
                    S[t, meta_to_real_v[reg_range['nodeid']]] = 1
        R = CheckpointSolver.solve_r_opt(g, S)
        return R, S

    @classmethod
    def load_griewank(cls, graph_size: int) -> pd.DataFrame:
        fname = f'{graph_size}.pkl.gz'
        local_path_base = os.path.join('/tmp', 'griewank_cache')
        local_path = os.path.join(local_path_base, fname)
        remote_path = f"https://optimalcheckpointing.s3.amazonaws.com/griewank_solutions/pickle/{fname}"
        if os.path.exists(local_path):
            try:
                return pd.read_pickle(local_path)
            except Exception as e:
                logger = setup_logger("griewank")
                logger.exception(e)
                logger.warn("Error loading cached griewank solution, corrupt file? Reloading from S3")
        pathlib.Path(local_path_base).mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(remote_path, local_path)
        return pd.read_pickle(local_path)
