import pickle
from typing import NamedTuple, Dict, List, Union, Optional

import numpy as np

from remat.core.solvers.strategy import SolveStrategy


class OperatorEvaluation(NamedTuple):
    id: int
    arg_regs: Dict[int, int]
    out_register: int
    operator_cost: int
    is_backwards: bool = False
    update_aux_vars: bool = True  # will be true if this is the last time this node is evaluated


class AllocateRegister(NamedTuple):
    register_id: int
    for_operation_id: int
    register_size: int


class DeallocateRegister(NamedTuple):
    op_id: int
    register_id: int


Schedule = List[Union[OperatorEvaluation, AllocateRegister, DeallocateRegister]]


class ScheduledResult(NamedTuple):
    solve_strategy: SolveStrategy
    cost_file: str
    input_shape: List[int]  # input_shape[0] is batch size
    R: np.ndarray
    S: np.ndarray
    solver_budget: float

    schedule: Schedule
    peak_ram: int
    activation_ram: int
    cpu: int

    mem_grid: np.ndarray
    mem_timeline: List[int]

    # ILP-only data
    U: Optional[np.ndarray] = None
    Free_E: Optional[np.ndarray] = None
    ilp_feasible: Optional[bool] = None
    ilp_num_variables: Optional[int] = None
    ilp_num_constraints: Optional[int] = None

    # Profiling data
    solve_time_s: Optional[float] = None
    schedule_time_s: Optional[float] = None

    def dumps(self) -> bytes:
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def loads(serialized_result: bytes):
        return pickle.loads(serialized_result)
