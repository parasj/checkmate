import os
from typing import Optional

from checkmate.core.dfgraph import DFGraph
from checkmate.core.enum_strategy import ImposedSchedule
from checkmate.core.utils.definitions import PathLike


class ILPSolverORTools:
    def __init__(
            self,
            g: DFGraph,
            budget: int,
            seed_s=None,
            integral=True,
            imposed_schedule: ImposedSchedule = ImposedSchedule.FULL_SCHEDULE,
            solve_r=True,
            write_model_file: Optional[PathLike] = None,
            num_threads=os.cpu_count()
    ):
        self.num_threads = num_threads
        self.model_file = write_model_file
        self.seed_s = seed_s
        self.integral = integral
        self.imposed_schedule = imposed_schedule
        self.solve_r = solve_r
        self.budget = budget
        self.g = g

        if not self.integral:
            assert not self.solve_r, "Can't solve for R if producing a fractional solution"
