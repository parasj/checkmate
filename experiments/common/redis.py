import logging
import os
from typing import Optional, Tuple, NamedTuple

import dotenv
from redis import StrictRedis

from remat.core.schedule import ScheduledResult
from remat.core.enum_strategy import SolveStrategy


class RedisCacheKey(NamedTuple):
    platform: str
    model_name: str
    batch_size: int
    solve_strategy: SolveStrategy
    model_version: Optional[str] = None
    cost_file: Optional[str] = None
    solver_budget: Optional[float] = None
    input_shape: Optional[Tuple[int]] = None
    global_project_version: str = "2.0"

    def key(self, *extra_args):
        def join(*args, delimiter="/"):
            return delimiter.join(map(lambda s: str(s).strip("/ \t\n\r"), args))
        return join(self.global_project_version, self.platform, self.model_name, self.model_version,
                    "bs" + str(self.batch_size), tuple(self.input_shape or []), self.solve_strategy.value,
                    SolveStrategy.get_version(self.solve_strategy), self.cost_file, self.solver_budget, *extra_args)


class RedisCache:
    def __init__(self, host=None, port=None, db=None, password=None, key_prefix=""):
        dotenv_location = dotenv.find_dotenv()
        if len(dotenv_location):
            logging.info(f'Loading dotenv config from {dotenv_location}')
            dotenv.load_dotenv(dotenv_location)
        else:
            logging.warning("Failed to load dotenv config!")

        self.key_prefix = key_prefix
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port or int(os.environ.get("REDIS_PORT", 6379))
        self.db = db or int(os.environ.get("REDIS_DB", 0))
        self.password = password or os.environ.get("REDIS_PASSWORD", "")
        self.redis_conn = StrictRedis(host=self.host, port=self.port, db=self.db, password=self.password)

    # def query_results(self, key_pattern: str) -> Tuple[List[ScheduledResult], List[str]]:
    #     result_list = []
    #     keys = []
    #     with self.make_client() as c:
    #         for key in c.scan_iter(key_pattern):
    #             result_bytes = c.get(key)
    #             if result_bytes:
    #                 result_list.append(ScheduledResult.loads(result_bytes))
    #                 keys.append(key)
    #     return result_list, keys

    # def read_results(self, solver: SolveStrategy, cost_file: str) -> Tuple[List[ScheduledResult], List[str]]:
    #     cost_file = cost_file if cost_file is not None else "flops"
    #     key_pattern = self.join(self.key_prefix, solver.value, SolveStrategy.get_version(solver), cost_file + "*")
    #     print("key pattern", key_pattern)
    #     return self.query_results(key_pattern)

    def read_result(self, cache_key: RedisCacheKey, ilp_time_limit: int = -1) -> Optional[ScheduledResult]:
        key = cache_key.key(ilp_time_limit)
        result_bytes = self.redis_conn.get(key)
        if result_bytes:
            res = ScheduledResult.loads(result_bytes)
            if res.solve_strategy == SolveStrategy.OPTIMAL_ILP_GC:
                if res.ilp_aux_data is not None and (res.ilp_aux_data.ilp_time_limit >= ilp_time_limit):
                    return res
            elif res.schedule_aux_data is not None:
                return res
        return None

    def write_result(self, key: RedisCacheKey, result: ScheduledResult, ilp_time_limit: int = -1):
        return self.redis_conn.set(key.key(ilp_time_limit), result.dumps())

    def __del__(self):
        self.redis_conn.close()