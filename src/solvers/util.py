from typing import Set

from solvers import solver
from remat.core.graph import Graph
import numpy as np


def setup_implied_s_backwards(g: Graph, S: np.ndarray = None):
    """
    Given a backward graph, this function will set the appropriate items in S to 1 in order
    to satisfy no-recompute rules during backwards optimization.
    """
    S = S if S is not None else np.zeros((g.size, g.size), dtype=solver.SOLVER_DTYPE)
    Vbwd = set(g.v) - set(g.vfwd)
    for (start, end) in g.induce_subgraph(Vbwd):
        for t in range(start + 1, end + 1):
            S[t, start] = 1
    return S


def gen_s_matrix_fixed_checkpoints(g: Graph, segment_set: Set[int]):
    """
    Given a list of checkpoint locations, this function will generate
    as output S matrices denoting checkpoint schedule, given a set of
    fixed segments (only recompute once).
    """
    T = g.size_fwd
    Ttotal = g.size
    segment_set = list(sorted(segment_set))
    S = setup_implied_s_backwards(g)
    # set minimum input requirements
    for v in g.v:
        for u in g.predecessors(v):
            S[v, u] = 1

    # stripe every k nodes
    for t in range(1, Ttotal):
        for i in segment_set:
            if i < t:
                S[t, i] = 1

    # checkpoint ladders
    starts = [0] + list(map(lambda x: x, segment_set))
    ends = segment_set + [T + 1]
    for start, end in zip(starts, ends):
        for t in filter(lambda t: t < Ttotal, map(lambda x: Ttotal - x - 1, range(start, end))):
            for i in range(start, min(t, end)):
                S[t, i] = 1

    # forward checkpoint block
    for start, end in zip(starts, ends):
        for t in filter(lambda t: t < Ttotal, range(start, end + 1)):
            for i in range(start, min(t, end)):
                S[t, i] = 1

    # backward checkpoint block
    for start, end in zip(starts, ends):
        for t in filter(lambda _t: _t < Ttotal, range(start, end + 1)):
            back_t = Ttotal - 1 - t
            for i in range(start, end):
                back_i = g.forward_to_backward(i)
                if back_i is not None and back_i < back_t:
                    S[back_t, back_i] = 1

    return S
