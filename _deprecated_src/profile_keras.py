import argparse
import json
import os.path as osp

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.client import timeline
import tensorflow.compat.v1 as tf1

from experiments.keras_extractor import MODEL_NAMES, get_keras_model


def get_names(timeline):
    print()
    events = json.loads(timeline)['traceEvents']
    names_and_times = {e['name']: e['dur'] for e in events if e['ph'] == 'X'}
    print(names_and_times.keys())
    # will also try some of the rest


def get_exec_time_loss(loss_fn, logits_shape, num_runs=1):
    run_opts = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)

    times = []

    @tf.function
    def run_loss(logits, labels):
        return loss_fn(logits, labels)

    conc = run_loss.get_concrete_function(tf.TensorSpec(logits_shape),
                                          tf.TensorSpec(logits_shape))

    for run in range(num_runs + 1):
        with tf1.Session() as sess:
            run_meta = tf1.RunMetadata()
            sess.run(tf1.global_variables_initializer())
            logits = tf.random.normal(logits_shape)
            labels = tf.random.normal(logits_shape)
            out = conc(logits, labels)
            sess.run(out, options=run_opts, run_metadata=run_meta)
            t1 = timeline.Timeline(run_meta.step_stats)
            lctf = t1.generate_chrome_trace_format()

            del logits
            del labels

        time = convert_string_to_time(lctf)
        times.append(time)
    if np.std(times) <= np.std(times[1:]):
        return np.average(times), np.std(times)
    # Filter first run
    return np.average(times[1:]), np.std(times[1:])


def get_shapes(model, batch_size):
    if type(model.input) != list:
        input_shapes = [(batch_size,) + tuple(model.input.shape[1:])]
    else:
        input_shapes = [(batch_size,) + tuple(inp.shape[1:]) for inp in model.input]
    if type(model.output) != list:
        output_shapes = [(batch_size,) + tuple(model.output.shape[1:])]
    else:
        output_shapes = [(batch_size,) + tuple(inp.shape[1:]) for inp in model.output]
    return input_shapes, output_shapes


def get_concrete_function(model, input_shapes):
    func = tf.function(model)
    if len(input_shapes) == 1:
        tensor_specs = tf.TensorSpec(shape=input_shapes[0], dtype=tf.float32)
    else:
        tensor_specs = [tf.TensorSpec(shape=shp, dtype=tf.float32) for shp in input_shapes]
    concrete_func = func.get_concrete_function(tensor_specs)
    return concrete_func


# def get_exec_time_timeline(log_device_placement=True):
def get_exec_time_timeline(model, batch_size, get_grads=False, num_runs=1, return_timeline=False):
    print("get_exec_time_timeline", model.__class__.__name__)
    run_opts = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
    input_shapes, output_shapes = get_shapes(model, batch_size)
    concrete_function = get_concrete_function(model, input_shapes)

    # input_names = [f"input_random_normal_{i}" for i in range(len(input_shapes))]
    # output_names = [f"output_random_normal_{i}" for i in range(len(output_shapes))]
    # inputs = [tf.random.normal(shp, name=name) for name, shp in zip(input_names, input_shapes)]
    # outputs = [tf.random.normal(shp, name=name) for name, shp in zip(output_names, output_shapes)]
    times = []

    for run in range(num_runs + 1):
        # with tf1.Session(config=config) as sess:
        with tf1.Session() as sess:
            run_meta = tf1.RunMetadata()
            sess.run(tf1.global_variables_initializer())
            inputs = [tf.random.normal(shp) for shp in input_shapes]
            outputs = [tf.random.normal(shp) for shp in output_shapes]
            out = concrete_function(*inputs)
            if not get_grads:
                sess.run(out, options=run_opts, run_metadata=run_meta)
                t1 = timeline.Timeline(run_meta.step_stats)
                ctf = t1.generate_chrome_trace_format()
            else:
                grads = tf.gradients(out, inputs, grad_ys=outputs)
                run_meta = tf1.RunMetadata()
                sess.run(grads, options=run_opts, run_metadata=run_meta)
                t1 = timeline.Timeline(run_meta.step_stats)
                ctf = t1.generate_chrome_trace_format()
            if return_timeline:
                return ctf

            # for i in inputs:
            #    del i
            # del inputs
            # for o in outputs:
            #    del o
            # del outputs

        time = convert_string_to_time(ctf)
        times.append(time)

    # for handle in inputs:
    #    tf1.delete_session_tensor(handle)
    # for handle in output_names:
    #    tf1.delete_session_tensor(handle)
    if np.std(times) <= np.std(times[1:]):
        return np.average(times), np.std(times)
    # Filter first run
    return np.average(times[1:]), np.std(times[1:])


# def get_exec_time_profile(lyr, batch_size, get_grads=False):  # must
#     print(lyr)
#     run_opts = tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE)
#     input_shapes, _ = get_shapes(lyr, batch_size)
#     inputs = [tf.random.normal(shp) for shp in input_shapes]
#     concrete_function = get_concrete_function(lyr, input_shapes)
#     run_meta = tf1.RunMetadata()
#
#     with tf1.Session() as sess:
#         sess.run(tf1.global_variables_initializer())
#         out = concrete_function(*inputs)
#         sess.run(out, options=run_opts, run_metadata=run_meta)
#         profile = tf1.profiler.Profiler(sess.graph)
#         profile.add_step(0, run_meta)
#         profiler_options = (tf1.profiler.ProfileOptionBuilder(
#             tf1.profiler.ProfileOptionBuilder.time_and_memory(
#                 min_cpu_micros=int(0)
#             )).with_step(0).with_empty_output().build())
#         prof = profile.profile_graph(options=profiler_options)
#         micro_s = prof.total_exec_micros
#         if get_grads:
#             out_grads = tf.random.normal(tf.shape(out))
#             loss = tf.losses.mean_squared_error(out, out_correct)
#             grads = tf.gradients(loss, inp)
#     return micro_s, prof


def convert_string_to_time(s):
    print()
    events = json.loads(s)['traceEvents']
    names_and_times = {e['name']: e['dur'] for e in events if e['ph'] == 'X'}

    for key in names_and_times.keys():
        if 'Conv2DBackpropInput' in key:
            return names_and_times[key]

    names_and_times['RandomStandardNormal'] = 0  # remove normal initialization
    ban_keys = ['unknown', 'RandomStandardNormal', 'NoOp']
    longest_event = max(names_and_times, key=lambda x: names_and_times[x])

    while longest_event == '' or any([b in longest_event for b in ban_keys]):
        print("names_and_times[longest_event]", names_and_times[longest_event])
        names_and_times[longest_event] = 0
        longest_event = max(names_and_times, key=lambda x: names_and_times[x])

    print("longest_event", longest_event)
    print("names_and_times[longest_event]", names_and_times[longest_event])

    print()
    # will also try some of the rest
    return names_and_times[longest_event]


def main():
    tf1.logging.set_verbosity('ERROR')
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--model-name', default='MobileNet', choices=MODEL_NAMES)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('-s', '--input-shape', type=int, nargs='+', default=[])
    parser.add_argument('-o', '--output-file', default=None)
    parser.add_argument('-f', '--folder', default='profiles')
    parser.add_argument('-l', '--loss-function',
                        default='softmax_cross_entropy')
    parser.add_argument('-c', '--num-runs', type=int, default=1, help='Number of runs of the operator. '
                                                                      'Increase to reduce variance')
    args = parser.parse_args()
    input_shape = args.input_shape if args.input_shape else None
    output_file = args.output_file
    model_name = args.model_name
    batch_size = args.batch_size

    if output_file is None:
        output_file = model_name + "_runtimes"
    output_file = osp.join(args.folder, output_file)

    model = get_keras_model(model_name, input_shape=input_shape)
    loss_fn = eval("tf1.losses.{}".format(args.loss_function))
    print("Num layers:", len(model.layers))

    # Run first layer a few times (4). On GPUs, it seems the first graph run has some additional overhead.
    print("Dummy runs of the first layer...")
    get_exec_time_timeline(model.layers[1], batch_size, num_runs=3)

    # Profile forward pass
    print("Profile network start...")
    forwards = [get_exec_time_timeline(lyr, batch_size, num_runs=args.num_runs)
                for lyr in model.layers[1:]]
    forwards_times, forwards_stds = map(list, zip(*forwards))

    # Profile backward pass
    backwards = [get_exec_time_timeline(lyr, batch_size, get_grads=True, num_runs=args.num_runs)
                 for lyr in reversed(model.layers[1:])]
    backwards_times, backwards_stds = map(list, zip(*backwards))

    # Profile loss
    logits_shape = (batch_size, *model.output.shape[1:])
    print("logits_shape", logits_shape, "model output shape", model.output.shape, "batch size", batch_size)
    loss_time, loss_std = get_exec_time_loss(loss_fn, logits_shape, num_runs=args.num_runs)

    runtimes = forwards_times + [loss_time] + backwards_times
    stds = forwards_stds + [loss_std] + backwards_stds

    print()
    for t, std, lyr in zip(forwards_times, forwards_stds, model.layers[1:]):
        print("fwd", t, "+-", std / t * 100, "%", lyr.__class__.__name__)
    for t, std, lyr in zip(backwards_times, backwards_stds, reversed(model.layers[1:])):
        print("bwd", t, "+-", std / t * 100, "%", lyr.__class__.__name__)
    print("loss", loss_time, "+-", loss_std / loss_time * 100, "%")
    print()

    np.save(output_file, (runtimes, stds))


if __name__ == "__main__":
    tf1.disable_eager_execution()
    main()
