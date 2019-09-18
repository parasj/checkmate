import os
from typing import List, Optional, Iterable, Tuple

from redis import StrictRedis

from evaluation.util.solve_strategy import SolveStrategy
from global_version import GLOBAL_PROJECT_VERSION
from solvers.result import RSResult


class RedisCache:
    def __init__(self, host=None, port=None, db=None, password=None, key_prefix=""):
        self.key_prefix = key_prefix
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port or int(os.environ.get("REDIS_PORT", 6379))
        self.db = db or int(os.environ.get("REDIS_DB", 0))
        self.password = password or os.environ.get("REDIS_PASSWORD", "")

    def make_client(self):
        return StrictRedis(host=self.host, port=self.port, db=self.db, password=self.password,
                           single_connection_client=True)

    def query_results(self, key_pattern: str) -> Tuple[List[RSResult], List[str]]:
        result_list = []
        keys = []
        with self.make_client() as c:
            for key in c.scan_iter(key_pattern):
                result_bytes = c.get(key)
                if result_bytes:
                    result_list.append(RSResult.loads(result_bytes))
                    keys.append(key)
        return result_list, keys

    def read_results(self, solver: SolveStrategy, cost_file: str) -> Tuple[List[RSResult], List[str]]:
        cost_file = cost_file if cost_file is not None else "flops"
        key_pattern = self.join(self.key_prefix, solver.value, SolveStrategy.get_version(solver), cost_file + "*")
        print("key pattern", key_pattern)
        return self.query_results(key_pattern)

    def read_result(self, solver: SolveStrategy, budget: int, cost_file: str) -> Optional[RSResult]:
        key = self.join(self.key_prefix, solver.value, SolveStrategy.get_version(solver), cost_file, budget)
        with self.make_client() as c:
            result_bytes = c.get(key)
        if result_bytes:
            return RSResult.loads(result_bytes)
        return None

    def write_result(self, result: RSResult):
        strategy = result.solve_strategy
        key = self.join(self.key_prefix, strategy.value, SolveStrategy.get_version(strategy), result.cost_file,
                        result.solver_budget)
        with self.make_client() as c:
            return c.set(key, result.dumps())

    @staticmethod
    def join(*args, delimiter="/"):
        return delimiter.join(map(lambda s: str(s).strip("/ \t\n\r"), args))

    @staticmethod
    def make_key(platform, model_name, model_version, batch_size, input_shape, delimiter="/"):
        return RedisCache.join(GLOBAL_PROJECT_VERSION,
                               platform,
                               model_name,
                               model_version,
                               "bs" + str(batch_size),
                               input_shape if input_shape else "defaultshape",
                               delimiter=delimiter)
