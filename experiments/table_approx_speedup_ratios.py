from experiments.common.definitions import remat_data_dir
import numpy as np
import pandas as pd

import glob
import re

# compute aggregated tables of max and geomean lp approximation ratios
exp_name_re = re.compile(r"^(?P<platform>.+?)_(?P<model_name>.+?)_(?P<batch_size>[0-9]+?)_(?P<input_shape>None|.+?)$")
dfs = []
for path in (remat_data_dir() / 'budget_sweep').glob('**/slowdowns.csv'):
    slowdown_df = pd.read_csv(path)
    matches = exp_name_re.match(path.parents[0].name)
    model_name = matches.group('model_name')
    slowdown_df['Model name'] = [model_name] * len(slowdown_df)
    dfs.append(slowdown_df)
df = pd.concat(dfs)
del df['Unnamed: 0']
for valuekey in ['geomean_slowdown', 'max']:
    pivot_df = pd.pivot_table(df, values=valuekey, index=['Model name'], columns=['method'])
    pivot_df.to_csv(remat_data_dir() / 'budget_sweep' / f"{valuekey}_aggr.csv")

# compute lp relaxation speedups
ilp_runtime_dict = {}
lp_runtime_dict = {}
for model in ['p32xlarge_vgg_unet_32_None', 'p32xlarge_ResNet50_256_None', 'p32xlarge_MobileNet_512_None', 'p32xlarge_VGG16_256_None', 'p32xlarge_VGG19_256_None']:
    ilp_matcher = re.compile(r"Explored [0-9]+ nodes \([0-9]+ simplex iterations\) in (?P<ilp_runtime>[0-9\.]+) seconds")
    lp_matcher = re.compile(r"Solved in [0-9]+ iterations and (?P<lp_runtime>[0-9\.]+) seconds")
    ilp_runtimes = []
    for path in (remat_data_dir() / 'budget_sweep' / model / 'ilp_log').glob('./*.log'):
        with path.open('r') as f:
            file_contents = f.read()
        if 'Model is infeasible' in file_contents:
            continue
        match = ilp_matcher.search(file_contents)
        ilp_runtimes.append(float(match.group('ilp_runtime')))

    lp_runtimes = []
    for path in (remat_data_dir() / 'budget_sweep' / 'p32xlarge_vgg_unet_32_None' / 'lp_det_05').glob('./*.log'):
        with path.open('r') as f:
            file_contents = f.read()
        if 'Model is infeasible' in file_contents:
            continue
        match = lp_matcher.search(file_contents)
        lp_runtimes.append(float(match.group('lp_runtime')))
    
    
    print("Speedup for {} is {:0.2f} ({:.2f} versus {:.2f}, count {} vs {})".format(model, np.median(ilp_runtimes) / np.median(lp_runtimes), np.mean(ilp_runtimes), np.mean(lp_runtimes), len(ilp_runtimes), len(lp_runtimes)))
    ilp_runtime_dict[model] = ilp_runtimes
    lp_runtime_dict[model] = lp_runtimes
