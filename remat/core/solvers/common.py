from typing import Set

import numpy as np

import remat.core
from remat.core.dfgraph import DFGraph

SOLVER_DTYPE = np.int


def setup_implied_s_backwards(g: DFGraph, S: np.ndarray = None):
    """
    Given a backward graph, this function will set the appropriate items in S to 1 in order
    to satisfy no-recompute rules during backwards optimization.
    """
    S = S if S is not None else np.zeros((g.size, g.size), dtype=remat.core.solvers.common.SOLVER_DTYPE)
    Vbwd = set(g.v) - set(g.vfwd)
    for (start, end) in g.induce_subgraph(Vbwd):
        for t in range(start + 1, end + 1):
            S[t, start] = 1
    return S


def gen_s_matrix_fixed_checkpoints(g: DFGraph, segment_set: Set[int]):
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


def solve_r_opt(G: DFGraph, S: np.ndarray):
    """Find the optimal recomputation pattern given caching decisions.
    Given S, E = [(i, j)] where node j depends on the result of node i,
    find R that minimizes cost, satisfies constraints. Assumes recomputation
    costs are nonnegative.

    NOTE: Does NOT check if memory limits are exceeded.
    Enforcing R[t,i] != S[t,i] does not seem to be necessary.
    """
    T = S.shape[0]
    assert S.shape[1] == T

    R = np.eye(T, dtype=S.dtype)  # Enforce R_t,t = 1
    # Enforce S_{t+1,v} <= S_{t,v} + R_{t,v},
    # i.e. R_{t,v} >= S_{t+1,v} - S_{t,v}
    S_diff = S[1:] - S[:-1]
    R[:-1] = R[:-1] | (R[:-1] < S_diff)
    # Create reverse adjacency list (child -> parents, i.e. node -> dependencies)
    adj = [[] for v in range(T)]
    for (u, v) in G.edge_list:
        adj[v].append(u)
    # Enforce R_{t,v} <= R_{t,u} + S_{t,u} for all (u, v) \in E
    for t in range(T):
        for v in range(t, -1, -1):
            for u in adj[v]:
                if R[t, v] > R[t, u] + S[t, u]:
                    R[t, u] = 1
    return R
