import argparse
import pickle
from collections import OrderedDict
from glob import glob

import pandas as pd

from evaluation.util.solve_strategy import SolveStrategy


def extract_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-folder-suffix", default="_gradless_eagerfalse", type=str,
                        help="Suffix of log folders. Will search in data/*[log-folder-suffix]/metrics.pickle")
    args = parser.parse_args()
    return args




def collect_execution_pickles(log_folder_suffix=""):
    # Load metric dictionaries
    metric_pickle_pattern = f"data/run_single_model/*{log_folder_suffix}/metrics.pickle"
    files = glob(metric_pickle_pattern)
    print("Pickles:", files)
    all_metrics = []
    for file in files:
        with open(file, "rb") as f:
            all_metrics.append(pickle.load(f))

    # Generate dataframe with diff. strategies on diff. rows
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    print("All keys:", all_keys)
    aggregated = {k: [] for k in all_keys}
    for metrics in all_metrics:
        for k in all_keys:
            if k in metrics:
                aggregated[k].append(metrics[k])
            else:
                aggregated[k].append(None)
    df = pd.DataFrame.from_dict(aggregated)
    outfile = f"data/run_single_model/aggregated_metrics{log_folder_suffix}.csv"
    df.to_csv(outfile)
    print(f">> Saved to {outfile}")

    # Aggregate by model, batch size, platform
    grouped = df.groupby(["model_name", "platform", "batch_size"])
    print(grouped.first())
    summary_dicts = []
    for (model_name, platform, batch_size), group in grouped:
        configuration_metrics = OrderedDict(
            model_name=model_name,
            batch_size=batch_size,
            platform=platform,
        )
        for strategy in SolveStrategy:
            results = group.loc[group.solve_strategy == strategy]
            if len(results) > 1:
                print(f"WARNING: Multiple results for strategy {strategy.name} ({len(results)}), picking max throughput")
                throughput = max(map(float, list(results.throughput_it_per_s)))
                configuration_metrics[f"Throughput of {strategy.name}"] = throughput
            elif len(results == 1):
                throughput = float(list(results.throughput_it_per_s)[0])
                configuration_metrics[f"Throughput of {strategy.name}"] = throughput
            else:
                configuration_metrics[f"Throughput of {strategy.name}"] = None
        summary_dicts.append(configuration_metrics)
    summary_df = pd.DataFrame(summary_dicts, columns=summary_dicts[0].keys())
    print(summary_df)
    outfile = f"data/run_single_model/throughputs{log_folder_suffix}.csv"
    summary_df.to_csv(outfile)
    print(f">> Saved summary to {outfile}")


if __name__ == "__main__":
    args = extract_params()
    collect_execution_pickles(args.log_folder_suffix)
