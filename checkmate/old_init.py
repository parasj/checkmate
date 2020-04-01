import tensorflow as tf
from checkmate.tf2.util.load_keras_model import get_keras_model
from checkmate.tf2.extraction import dfgraph_from_tf_function
from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_chen import solve_chen_greedy, solve_chen_sqrtn
from checkmate.core.solvers.strategy_optimal_ilp import solve_ilp_gurobi
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.wrapper import _get_gpu_memory, get_function
import pickle

"""
Intended Usage:
    Get a model
    use get_function on that model to grab a concrete function
    pass into one of the checkpoint functions
    run the new function
"""

print("Warning: importing checkmate currently turns off a single optimization in tensorflow on import")

from collections import namedtuple

def set_opts():
    opts = {}
    #tf.config.optimizer.set_jit(False)
    #opts["dependency"] = False
    #
    opts["remapper"] = False
    tf.config.optimizer.set_experimental_options(opts)

set_opts()

def norm_checkpoint (fn, solver="sqrtn"):
    fxn = fn.get_concrete_function(tf.TensorSpec(shape=fxn.input_spec, dtype=tf.float32))
    g = dfgraph_from_tf_function(fxn)
    if solver.lower() == "sqrtn" :
        sol = solve_chen_sqrtn(g, use_actuation_points=False)
    elif  solver.lower()=='all' :
        sol = solve_checkpoint_all(g)
    else:
        raise NotImplementedError("solver {} not implemented.  Please coose from sqrtn, all".format(solver))
    sched = sol.schedule
    new_fxn = edit_graph(fxn, g.op_dict, sched)
    return new_fxn

def ilp_checkpoint(fn,  budget="auto", fname= 'last_result.pkl', mode = "save", **kwargs):
    if budget=='auto' :
        budget = _get_gpu_memory()
    fxn = fn.get_concrete_function(tf.TensorSpec(shape=input_shape, dtype=tf.float32))
    g = dfgraph_from_tf_function(fxn)
    if mode=='load' :
        sched = pickle.load(fname)
    else:
        sol = solve_ilp_gurobi(g, budget, **kwargs)
        sched = sol.schedule
    if mode=='save':
        pickle.dump(fname)
    new_fxn = edit_graph(fxn, g.op_dict, sched)
    return new_fxn

