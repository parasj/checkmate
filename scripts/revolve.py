import numpy as np
import pyrevolve.crevolve as pr

from utils.setup_logger import setup_logger
logger = setup_logger("revolvenb")

from utils.graph import Graph
from solvers.solver import CheckpointSolver

from typing import Dict
from dataclasses import dataclass

from tqdm import tqdm
import pickle


@dataclass
class RegisterRange:
    node_id: int
    reg_id: int
    start_time: int
    end_time: int = None

    def gc(self, timestep):
        self.end_time = self.end_time if self.end_time is not None else timestep

    def as_dict(self):
        return {
            'reg_id': self.reg_id,
            'node_id': self.node_id,
            'live_range': (self.start_time, self.end_time)
        }


class RegisterPool:
    def __init__(self):
        self.next_reg_alloc = 0
        self.temp_reg_mapping = {}  # map from slot to current reg id
        self.regs: Dict[int, RegisterRange] = {}

    def assign_checkpoint(self, time_step, node_id, reg_slot):
        self._free_reg_id(time_step, reg_slot)
        reg = self._alloc_reg_id(time_step, node_id, reg_slot)
        self.temp_reg_mapping[reg_slot] = reg.reg_id

    def _alloc_reg_id(self, time_step, node_id, reg_slot):
        reg_id = self.next_reg_alloc
        self.next_reg_alloc += 1
        self.regs[reg_id] = RegisterRange(node_id, reg_id, time_step + 1)
        return self.regs[reg_id]

    def _free_reg_id(self, time_step, reg_slot):
        if reg_slot in self.temp_reg_mapping:
            reg_id = self.temp_reg_mapping[reg_slot]
            self.regs[reg_id].gc(time_step)
            return reg_id

    def get_register_ranges(self):
        return list(self.regs.values())


def run_revolve(N, b=None, debug=False):
    if b is None:
        b = pr.adjust(N)
    if debug:  logger.info(f"Running with {N} steps and {b} snapshots")
    c = pr.CRevolve(b, N)
    pool = RegisterPool()

    t = 0
    while True:
        action = c.revolve()
        if action == pr.Action.advance:
            for i in range(c.oldcapo + 1, c.capo + 1):
                t = max(t, i)
                if debug:  logger.debug(f"{t}\tFWD {i}")
        elif action == pr.Action.takeshot:
            if debug:  logger.debug(f"{t}\tregs[{c.check}] = {c.capo}")
            pool.assign_checkpoint(t, c.capo, c.check)
        elif action == pr.Action.restore:
            if debug:  logger.debug(f"{t}\tFWD {c.capo} = regs[{c.check}]")
        elif action == pr.Action.firstrun:
            fwd_node = c.capo
            t = max(t, N * 2 + 1 - fwd_node)
            if debug:  logger.debug(f"{t}\tBWD {fwd_node}")
        elif action == pr.Action.youturn:
            for i in range(c.capo, c.oldcapo - 1, -1):
                t = max(t, N * 2 + 1 - i)
                if debug:  logger.debug(f"{t}\tBWD {i}")
        if action == pr.Action.terminate:
            break
    regs = pool.get_register_ranges()
    for reg in regs:
        reg.gc(t)
        if debug:  print(reg)

    T = N * 2 + 1
    S = np.zeros((T, T), dtype=np.int32)
    np.fill_diagonal(S[1:], 1)
    for reg_range in regs:
        for t in range(reg_range.start_time, min(T, reg_range.end_time + 1)):
            if reg_range.node_id >= 0:
                S[t, reg_range.node_id] = 1

    g = Graph.gen_linear_graph(N)
    R = CheckpointSolver.solve_r_opt(g, S)

    range_out = [regrange.as_dict() for regrange in regs]
    return range_out, R, S


for N in tqdm(range(1, 1000)):
    configs = {}
    for b in range(1, N):
        range_out, R, S = run_revolve(N, b, debug=False)
        configs[(N, b)] = range_out
    pickle.dump(configs, open('logn_soln/griewank_{}.pickle'.format(N), 'wb'))
