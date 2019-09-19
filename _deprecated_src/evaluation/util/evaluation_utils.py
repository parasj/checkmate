import math
import sys
from typing import List, Dict

import numpy as np
import pandas
import ray
from tqdm import tqdm

from remat.core.solvers.enum_strategy import SolveStrategy
from solvers.result import RSResult

RSResultDict = Dict[SolveStrategy, List[RSResult]]


def compute_cpu_prefix_min(in_tups: List[RSResult]):
    """Given list of RSResults, this will yield the sorted prefix min for runtime"""
    current_min_cpu = math.inf
    for t in sorted(in_tups, key=lambda t: t.peak_ram):
        if t.cpu < current_min_cpu:
            current_min_cpu = t.cpu
            yield (current_min_cpu, t)


def prefix_min_np(values: np.ndarray):
    assert values.ndim == 1
    values_min = np.copy(values)
    for i in range(1, values.shape[0]):
        values_min[i] = min(values_min[i - 1], values[i])
    return values_min


def lookup_closest_s(sorted_list, key):
    """return value that is closest (but no greater) than key in a sorted list of key-value tuples"""
    last_value = None
    for list_key, list_value in sorted_list:
        if list_key > key:
            return last_value.S if last_value is not None else None
        last_value = list_value


def confirm(query):
    if not input(query + " (y/n): ").lower().strip()[:1] == "y":
        print("Ok. Exiting.")
        sys.exit(0)


def result_dict_to_dataframe(results: RSResultDict):
    df = []
    for strategy, results_item in results.items():
        result_list = [results_item] if isinstance(results_item, RSResult) else list(results_item)
        for result in result_list:
            d = result._asdict()
            for key in ['R', 'S', 'U', 'schedule', 'mem_grid', 'mem_timeline']:
                del d[key]
            df.append(d)
    return pandas.DataFrame(df)


def get_futures(futures, desc="Jobs", progress_bar=True):
    if progress_bar:
        results = []
        with tqdm(total=len(futures), desc=desc) as pbar:
            while len(futures):
                done_results, futures = ray.wait(futures)
                results.extend(ray.get(done_results))
                pbar.update((len(done_results)))
        return results
    else:
        return ray.get(futures)
