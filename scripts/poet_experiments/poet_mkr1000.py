import logging
import shutil

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from checkmate.core.graph_builder import GraphBuilder
from checkmate.poet.poet_solver import solve_poet_cvxpy
from checkmate.core.utils.definitions import PathLike
from checkmate.plot.definitions import checkmate_data_dir
from checkmate.poet.power_computation import MKR1000, make_linear_network, get_net_costs, GradientLayer


def make_dfgraph_costs(net, device):
    power_specs = get_net_costs(device, net)
    per_layer_specs = pd.DataFrame(power_specs).to_dict(orient='records')

    layer_names, power_cost_dict, page_in_cost_dict, page_out_cost_dict = {}, {}, {}, {}
    gb = GraphBuilder()
    for idx, (layer, specs) in enumerate(zip(net, per_layer_specs)):
        layer_name = "layer{}_{}".format(idx, layer.__class__.__name__)
        layer_names[layer] = layer_name
        gb.add_node(layer_name, cpu_cost=specs['compute'], ram_cost=specs['memory'], backward=isinstance(layer, GradientLayer))
        gb.set_parameter_cost(gb.parameter_cost + specs['param_memory'])
        page_in_cost_dict[layer_name] = specs['pagein_cost']
        page_out_cost_dict[layer_name] = specs['pageout_cost']
        power_cost_dict[layer_name] = specs['compute']
        for dep in layer.depends_on:
            gb.add_deps(layer_name, layer_names[dep])
    g = gb.make_graph()

    ordered_names = [(topo_idx, name) for topo_idx, name in g.node_names.items()]
    ordered_names.sort(key=lambda x: x[0])
    ordered_names = [x for _, x in ordered_names]

    compute_costs = np.asarray([power_cost_dict[name] for name in ordered_names]).reshape((-1, 1))
    page_in_costs = np.asarray([page_in_cost_dict[name] for name in ordered_names]).reshape((-1, 1))
    page_out_costs = np.asarray([page_out_cost_dict[name] for name in ordered_names]).reshape((-1, 1))
    return g, compute_costs, page_in_costs, page_out_costs


def solve(budget):
    net = make_linear_network()
    device = MKR1000
    g, compute_costs, page_in_costs, page_out_costs = make_dfgraph_costs(net, device)
    solution = solve_poet_cvxpy(g, budget, compute_costs, page_in_costs, page_out_costs,
                                solver_override="GUROBI", verbose=True)
    return dict(budget=budget, solution=solution, dfgraph=g, cpu_cost=compute_costs,
                page_in_cost=page_in_costs, page_out_cost=page_out_costs)


def write_visualization(solution, out_path: PathLike):
    plt.figure()
    fig, axarr = plt.subplots(1, 5, figsize=(25, 5))
    for arr, ax, name in zip(solution, axarr, ['R', 'S_RAM', 'S_SD', 'M_sd2ram', 'M_ram2sd']):
        ax.invert_yaxis()
        ax.pcolormesh(arr, cmap="Greys", vmin=0, vmax=1)
        ax.set_title(name)
    fig.savefig(out_path)


def featurize_row(data_row):
    out_vec = dict(budget=data_row['budget'])
    R, S_RAM, S_SD, M_sd2ram, M_ram2sd = data_row['solution']
    out_vec['total_compute_runtime'] = np.sum(R @ data_row['cpu_cost'])
    out_vec['total_page_cost'] = np.sum(M_sd2ram @ data_row['page_out_cost']) + np.sum(M_ram2sd @ data_row['page_in_cost'])
    return out_vec


if __name__ == "__main__":
    sns.set('notebook')
    sns.set_style('dark')

    data_dir = checkmate_data_dir() / "poet_mkr1000"
    shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    data = []
    for budget in tqdm(np.linspace(1000, 1600, num=10)):
        try:
            solution_dict = solve(budget)
            write_visualization(solution_dict['solution'], data_dir / "budget_{}.png".format(budget))
            data.append(solution_dict)
        except Exception as e:
            logging.exception(e)

    df = pd.DataFrame(map(featurize_row, data))
    df.to_pickle(str((data_dir / "results.pkl").resolve()))
