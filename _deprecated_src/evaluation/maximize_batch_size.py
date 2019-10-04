from __future__ import division

import math
import os

from integration.tf2.TF2ExtractorParams import TF2ExtractorParams
from experiments.common.keras_extractor import get_keras_model
from solvers.result import RSResult
from experiments.solver_ilp_max_batchsize import MaxBatchILPSolver
from utils.setup_logger import setup_logger


MODELS = ["VGG16", "vgg_unet"]


def eval_maximize_batch_size(args, log_base: str, key_prefix: str, cost_file: str):
    logger = setup_logger("eval_maximize_batch_size", os.path.join(log_base, "eval_maximize_batch_size.log"))
    model_name = args.model_name
    model = get_keras_model(model_name)
    params = TF2ExtractorParams(model, batch_size=1, log_base=log_base, loss_cpu_cost=1000)
    logger.info(f"Running model {model_name}")
    model_file = os.path.join(log_base, f"max_bs_{model_name}.mps")
    param_dict = {'LogToConsole': 1,
                  'LogFile': os.path.join(log_base, f"max_bs_{model_name}.solve.log"),
                  'Threads': os.cpu_count(),
                  'TimeLimit': math.inf}
    ilp_solver = MaxBatchILPSolver(params.g, budget=16 * GB - params.g.cost_ram_fixed, model_file=model_file,
                                   gurobi_params=param_dict, cpu_fwd_factor=2)
    ilp_solver.build_model()
    R, S, U, Free_E, batch_size = ilp_solver.solve()
    logger.info(f"Max batch size = {batch_size}")

    params_new = TF2ExtractorParams(model, batch_size=int(batch_size), log_base=log_base)
    sched, peak_ram, act_ram, cpu, mem_usage, mem_timeline = RSResult.verify_solution(params_new.g, R, S)
    logger.info(f"Solution peak_ram={peak_ram:.2E} cpu={cpu:.2E}")

    save_file = os.path.join(log_base, f'{model}_plot.png')
    RSResult.plot(params.g, R, S, U, plot_mem_usage=True, timeline=mem_timeline, save_file=save_file)
