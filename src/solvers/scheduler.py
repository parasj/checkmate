from typing import NamedTuple, List, Dict, Union


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


class ScheduleBuilder:
    def __init__(self, g, verbosity: int = 2):
        self.max_ram = 0
        self.total_cpu = 0
        self.g = g
        self.schedule: Schedule = []
        self.live_registers: Dict[int, int] = {}
        self.next_free_register_id = 0
        self.verbosity = verbosity
        self.ram_timeline: List[int] = []
        # todo live ranges and garbage collection

    def is_op_cached(self, op_id: int):
        return op_id in self.live_registers.keys()

    def allocate_register(self, op_id: int):
        """
        Schedule a register allocation
        :param op_id: ID for operation whose output will be stored in this register,
        :return: the newly allocated register ID
        """
        if op_id in self.live_registers.keys():
            if self.verbosity >= 2:
                print("WARNING! Double allocating output register for op #{}, skipping allocation to reuse reg #{}"
                      .format(op_id, self.live_registers[op_id]))
            return self.live_registers[op_id]
        reg = AllocateRegister(self.next_free_register_id, op_id, self.g.cost_ram[op_id])
        self.live_registers[op_id] = reg.register_id
        self.schedule.append(reg)
        self.next_free_register_id += 1
        self.max_ram = max(self.max_ram, self.current_mem())
        self.ram_timeline.append(self.current_mem())
        return reg.register_id

    def run_operator(self, op_id: int, update_aux_vars: bool):
        debug_str = "Dependency not fulfilled for op #{}, ops in ram now are {} but I need {}".format(
            op_id, set(self.live_registers.keys()), self.g.predecessors(op_id))
        assert all([pred in self.live_registers.keys() for pred in self.g.predecessors(op_id)]), debug_str
        out_reg = self.allocate_register(op_id)
        in_regs = {pred_id: self.live_registers[pred_id] for pred_id in self.g.predecessors(op_id)}
        eval_op = OperatorEvaluation(op_id, in_regs, out_reg, self.g.cost_cpu[op_id],
                                     update_aux_vars=update_aux_vars, is_backwards=op_id not in self.g.vfwd)
        self.schedule.append(eval_op)
        self.total_cpu += self.g.cost_cpu[op_id]
        self.ram_timeline.append(self.current_mem())

    def deallocate_register(self, op_id: int):
        """
        Schedule a register deallocation
        :param op_id: ID for operation whose output will be stored in this register
        """
        if op_id not in self.live_registers.keys():
            print("WARNING! Double free output register for op #{}".format(op_id))
        reg_id = self.live_registers.pop(op_id)
        self.schedule.append(DeallocateRegister(op_id, reg_id))
        self.ram_timeline.append(self.current_mem())

    def current_mem(self):
        return sum(map(self.g.cost_ram.get, self.live_registers.keys()))

