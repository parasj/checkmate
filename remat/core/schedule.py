from typing import NamedTuple, Dict, List, Union


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
