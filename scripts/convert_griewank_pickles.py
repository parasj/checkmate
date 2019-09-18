import glob
import os
import pickle
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm


def load_file(file):
    data = []
    obj = pickle.load(open(file, 'rb'))
    for (N, b), regs in obj.items():
        for reg in regs:
            d = dict(graph_size=N, budget=b, timestart=reg['live_range'][0], timeend=reg['live_range'][1],
                       nodeid=reg['node_id'], regid=reg['reg_id'])
            data.append(d)
    return pd.DataFrame(data)

files = glob.glob("./*.pickle")
with Pool(os.cpu_count()) as p:
    maps = tqdm(p.imap(load_file, files), total=len(files), desc="Load")
    df = pd.concat(maps)

group_N = df.groupby('graph_size')
for graph_size, df in tqdm(((x, group_N.get_group(x)) for x in group_N.groups), total=len(group_N.groups), desc="WriteOut"):
    df.to_parquet(f'out/logn_{graph_size}.parquet')