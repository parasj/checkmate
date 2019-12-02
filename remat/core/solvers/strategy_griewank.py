import logging
import pathlib
import shutil
import urllib.request

import numpy as np
import pandas as pd

from remat.core.dfgraph import DFGraph
from remat.core.utils.solver_common import solve_r_opt, setup_implied_s_backwards
from remat.core.utils.timer import Timer


def solve_griewank(g: DFGraph, budget: int):
    # todo check graph is chain graph
    raise NotImplementedError("Griewank's checkpointing strategy does not support the nonlinear structure of low-level Tensorflow graphs.")
    # with Timer('solve_griewank') as timer_solve:
    #     r, s = _solve_griewank_to_rs(g, budget)
    # schedule, aux_data = schedule_from_rs(g, r, s)
    # griewank_feasible = (r is not None and s is not None)  # griewank load from FS
    # return ScheduledResult(
    #     solve_strategy=SolveStrategy.GRIEWANK_LOGN,
    #     solver_budget=budget,
    #     feasible=griewank_feasible,
    #     schedule=schedule,
    #     schedule_aux_data=aux_data,
    #     solve_time_s=timer_solve.elapsed  # this is technically just filesystem load time
    # )


def _solve_griewank_to_rs(g: DFGraph, budget: int):
    S = np.zeros((g.size, g.size), dtype=np.int32)
    S = setup_implied_s_backwards(g, S)
    np.fill_diagonal(S[1:], 1)

    ap_points = list(sorted(g.checkpoint_set))
    metaTfwd = len(ap_points)
    ap_points = ap_points + [g.forward_to_backward(p) for p in reversed(ap_points)]
    meta_to_real_v = {ap_points.index(ap_point): ap_point for ap_point in ap_points}
    try:
        regranges_all = _load_griewank(metaTfwd)
    except Exception as e:
        logging.exception(e)
        return None, None

    if regranges_all is None:
        return None, None

    max_budget = max(regranges_all["budget"])
    regranges = regranges_all[regranges_all["budget"] == min(budget, max_budget)]
    if len(regranges.index) < 1:
        return None, None

    def map_time(_t: int) -> int:
        return min(meta_to_real_v.get(_t, np.inf), g.size)

    for index, reg_range in regranges.iterrows():
        for t in range(map_time(reg_range['timestart']), map_time(reg_range['timeend'] + 1)):
            if reg_range['nodeid'] > 0:
                S[t, meta_to_real_v[reg_range['nodeid']]] = 1
    R = solve_r_opt(g, S)
    return R, S


def _load_griewank(graph_size: int) -> pd.DataFrame:
    fname = f'{graph_size}.pkl.gz'
    local_path_base = pathlib.Path('/tmp') / 'remat_cache' / 'griewank_solutions'
    local_path = local_path_base / fname
    remote_path = f"https://optimalcheckpointing.s3.amazonaws.com/griewank_solutions/pickle/{fname}"
    if local_path.exists():
        try:
            return pd.read_pickle(local_path)
        except Exception as e:
            logging.exception(e)
            logging.warning("Error loading cached griewank solution, corrupt file? Reloading from S3")
    with Timer('griewank_dl') as dl_timer:
        local_path_base.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(remote_path, local_path)
    logging.info(f"Loaded graph from {remote_path} and saving to {local_path} in {dl_timer.elapsed:.2f}s")
    return pd.read_pickle(local_path)


def clean_griewank_cache():
    local_path_base = pathlib.Path('/tmp') / 'remat_cache' / 'griewank_solutions'
    shutil.rmtree(local_path_base, ignore_errors=True)
