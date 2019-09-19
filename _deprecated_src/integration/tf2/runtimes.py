import tensorflow as tf
import numpy as np
import importlib
import os.path as osp
#from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.python.client import timeline
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf1
import json
import sys
#sys.path.append('')

from extraction import MODEL_NAMES, get_keras_model
import argparse
#from extraction import extract_graph_from_keras, get_keras_model, MODEL_NAMES


def get_exec_time_loss(loss_fn, logits_shape):
    run_opts = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
    logits = tf.random.normal(logits_shape)
    labels = tf.random.normal(logits_shape)
    @tf.function
    def run_loss():
        return loss_fn(logits, labels)
    conc = run_loss.get_concrete_function()
    
    with tf1.Session() as sess:
        run_meta = tf1.RunMetadata()
        sess.run(tf1.global_variables_initializer())
        out = conc()
        sess.run(out, options=run_opts, run_metadata=run_meta)
        t1 = timeline.Timeline(run_meta.step_stats)
        lctf = t1.generate_chrome_trace_format()

    return convert_string_to_time(lctf)

    

def get_exec_time_timeline(mod, batch_size, get_grads=False):
    run_opts = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
    if type(mod.input) != list:
        input_shapes = [(batch_size,) + tuple(mod.input.shape[1:])]
        output_shapes = [(batch_size,) + tuple(mod.output.shape[1:])]
    else:
        input_shapes = [(batch_size,) + tuple(inp.shape[1:]) for inp in mod.input]
        output_shapes = [(batch_size,) + tuple(inp.shape[1:]) for inp in mod.output]
    inputs = [tf.random.normal(shp) for shp in input_shapes]
    outputs = [tf.random.normal(shp) for shp in output_shapes]
    func = tf.function(mod)
    if len(inputs) == 1:
        conc = func.get_concrete_function(tf.TensorSpec(shape=input_shapes[0], 
            dtype=tf.float32))
    else:
        conc = func.get_concrete_function([tf.TensorSpec(shape=shp, dtype=tf.float32)
        for shp in input_shapes])
    
    with tf1.Session() as sess:
        run_meta = tf1.RunMetadata()
        sess.run(tf1.global_variables_initializer())
        out = conc(*inputs)
        if not get_grads:
            sess.run(out, options=run_opts, run_metadata=run_meta)
            t1 = timeline.Timeline(run_meta.step_stats)
            ctf = t1.generate_chrome_trace_format()
        else:
            grads = tf.gradients(out, inputs, grad_ys = outputs)
            run_meta = tf1.RunMetadata()
            sess.run(grads, options=run_opts, run_metadata=run_meta)
            t1 = timeline.Timeline(run_meta.step_stats)
            ctf = t1.generate_chrome_trace_format()
        

    return convert_string_to_time(ctf)

def get_exec_time_profile(lyr, batch_size, get_grads=False):  #must 

    print(lyr.__class__.__name__)
    
    run_opts = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
    if type(lyr.input) != list:
        input_shapes = [(batch_size,) + tuple(lyr.input.shape[1:])]
    else:
        input_shapes = [(batch_size,) + tuple(inp.shape[1:]) for inp in lyr.input]
    inputs = [tf.random.normal(shp) for shp in input_shapes]
    func = tf.function(lyr)
    conc = func.get_concrete_function(*[tf.TensorSpec(shape=shp, dtype=tf.float32)
        for shp in input_shapes])
    run_meta = tf1.RunMetadata()
    
    with tf1.Session() as sess:
        sess.run(tf1.global_variables_initializer())
        out = conc(*inputs)
        sess.run(out, options=run_opts, run_metadata=run_meta)
        profile = tf1.profiler.Profiler(sess.graph)
        profile.add_step(0,run_meta)
        profiler_options = (tf1.profiler.ProfileOptionBuilder(
                tf1.profiler.ProfileOptionBuilder.time_and_memory(
                    min_cpu_micros=int(0)
                )).with_step(0).with_empty_output().build())
        prof = profile.profile_graph(options=profiler_options)
        micro_s = prof.total_exec_micros
        if get_grads:
            out_grads = tf.random.normal(tf.shape(out))
            loss = tf.losses.mean_squared_error(out, out_correct)
            grads = tf.gradients(loss, inp)
    return micro_s, prof

def convert_string_to_time(s):
    events = json.loads(s)['traceEvents']
    names_and_times = {e['name'] : e['dur'] for e in events if e['ph'] == 'X'}
    names_and_times['RandomStandardNormal'] = 0 #remove normal initialization
    longest_event = max(names_and_times, key = lambda x: names_and_times[x])
    print(longest_event)
    #will also try some of the rest
    return names_and_times[longest_event]




def main():
    tf1.logging.set_verbosity('ERROR')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model-name', default='MobileNet', choices=MODEL_NAMES)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-o', '--output-file', default=None)
    parser.add_argument('-f', '--folder', default='profiles') 
    parser.add_argument('-l', '--loss-function',
            default='softmax_cross_entropy')
    args = parser.parse_args()
    output_file = args.output_file

    model_name = args.model_name
    batch_size = args.batch_size
    
    if output_file is None:
        output_file = model_name + "_runtimes"
    output_file = osp.join(args.folder, output_file)
    print("WARNING: Using default input shape")
    model = get_keras_model(model_name, input_shape=None)
    loss_fn = eval("tf1.losses.{}".format(args.loss_function))
    #ctf, t1 = get_exec_time_timeline(model, batch_size)
    forwards  = [
            get_exec_time_timeline(lyr, batch_size)
            for lyr in model.layers[1:]]
    backwards = [get_exec_time_timeline(lyr, batch_size, get_grads=True)
            for lyr in model.layers[1:]]
    logits_shape = list(model.output.shape)
    logits_shape[0] = batch_size
    logits_shape = tuple(logits_shape)
    loss_time = [get_exec_time_loss(loss_fn, logits_shape)]

    runtimes = forwards + loss_time + list(reversed(backwards))

    np.save(output_file, runtimes)

   

if __name__ == "__main__":    
    tf1.disable_eager_execution()
    main()
