from __future__ import division

import logging
import os
import pathlib
import urllib.request
import urllib.error
from collections import defaultdict
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from remat.core.utils.definitions import PathLike

# BATCH_SIZES_LOAD = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BATCH_SIZES_LOAD = [32, 64, 128, 256, 512, 1024, 2048]


class CostModel:
    def __init__(self, model_name: str, platform: str, log_base: PathLike, quantization: int):
        self.log_base = log_base
        self.logger = logging.getLogger("CostModel")

        self.model_name = model_name
        self.platform = platform
        self.quantization = quantization

        # Make cost file paths
        self.batch_sizes_to_load = []
        self.cost_files_to_load = []
        for batch_size in BATCH_SIZES_LOAD:
            cost_file = self.load_profile_s3(model_name, batch_size, platform)
            if cost_file is not None:
                self.batch_sizes_to_load.append(batch_size)
                self.cost_files_to_load.append(cost_file)
            else:
                self.logger.warning(f"Missing cost file {cost_file} for batch size {batch_size}")

        # Cost model parameters
        self.fits = []
        self.slopes_np: Optional[np.ndarray] = None
        self.intercepts_np: Optional[np.ndarray] = None

    def fit(self):
        import scipy.stats
        self.logger.info("Loading measured costs")
        costs_by_layer = defaultdict(list)
        batch_sizes_by_layer = defaultdict(list)
        for batch_size, cost_file in zip(self.batch_sizes_to_load, self.cost_files_to_load):
            costs = self.load_costs(cost_file)
            if costs is None:
                self.logger.error(f"Error loading cost file {cost_file}, skipping")
                continue
            for layer, cost in enumerate(costs):
                costs_by_layer[layer].append(cost)
                batch_sizes_by_layer[layer].append(batch_size)

        self.logger.info("Fitting cost model for each layer linear in batch size")
        self.fits = []
        for layer, costs in sorted(costs_by_layer.items()):
            fit = scipy.stats.linregress(batch_sizes_by_layer[layer], costs)
            slope, intercept, rvalue, pvalue, stderr = fit
            self.fits.append(fit)

            if intercept / 1000 >= 100:
                # Greater than 100 ms overhead for the layer
                self.logger.warn(f"Layer {layer} has overhead (bs=0 cost) of {intercept / 1000} ms. "
                                 f"r={rvalue}, p={pvalue}, stderr={stderr}, for cost model {slope}*bs+{intercept}")
            if rvalue < 0.8:
                self.logger.warn(
                    f"Poor fit: layer {layer} has r={rvalue}, p={pvalue}, stderr={stderr}, for cost model {slope}*bs+{intercept}")

        # Collect models into ndarrays
        nlayer = len(self.fits)
        self.slopes_np = np.zeros(nlayer, dtype=float)
        self.intercepts_np = np.zeros(nlayer, dtype=float)
        for layer, (slope, intercept, _, __, ___) in enumerate(self.fits):
            # Costs should increase with batch size
            self.slopes_np[layer] = max(0, slope)
            self.intercepts_np[layer] = max(0, intercept)

    def quantize_costs(self, costs: np.ndarray) -> np.ndarray:
        rounded = np.around(costs / self.quantization)
        rounded = np.array(rounded, dtype=np.int)
        rounded = rounded * self.quantization
        return rounded

    def get_costs(self, batch_size: int) -> Optional[np.ndarray]:
        # Attempt to load costs if available
        if batch_size in self.batch_sizes_to_load:
            cost_file = self.cost_files_to_load[self.batch_sizes_to_load.index(batch_size)]
            costs = self.load_costs(cost_file)
            if costs is not None:
                self.logger.info(f"Using measured costs {cost_file} for batch size {batch_size}")
                self.logger.info(f"Quantizing costs")
                return self.quantize_costs(costs)

        self.logger.info(f"Using linear cost model for batch size {batch_size}")
        # raise NotImplementedError("Linear cost model disabled. Must have a profile .npy file")
        return self.slopes_np * batch_size + self.intercepts_np

    def load_costs(self, cost_file: str, withdevs=False):
        try:
            cost_list, stds = np.load(cost_file)
        except ValueError as exc:
            if not withdevs:
                cost_list = np.load(cost_file)
                return cost_list
            self.logger.error(f"Error loading cost file {cost_file}: %s", exc)
            if withdevs:
                return None, None
            return None
        if withdevs:
            return cost_list, stds
        return cost_list

    def plot_costs(self):
        self.logger.info("Plotting cost model")
        data_by_layer = defaultdict(lambda: ([], [], []))
        for batch_size, cost_file in zip(self.batch_sizes_to_load, self.cost_files_to_load):
            import datetime
            threshutc = datetime.datetime.utcnow() - datetime.timedelta(days=3)
            if datetime.datetime.fromtimestamp(int(os.path.getmtime(cost_file))) < threshutc:
                self.logger.warn(f"Skipping {cost_file} for plotting, too old")
                continue
            costs, stds = self.load_costs(cost_file, withdevs=True)
            if costs is None:
                continue
            for layer, (cost, std) in enumerate(zip(costs, stds)):
                data_by_layer[layer][0].append(batch_size)
                data_by_layer[layer][1].append(cost)
                data_by_layer[layer][2].append(std)

        sns.set()
        fig, ax = plt.subplots(1, 1, figsize=(16, 20))
        ax.set_title(f"Cost model for {self.model_name}")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Layer cost (microseconds)")

        n = len(data_by_layer)
        cmap = sns.cubehelix_palette(n)
        batch_sizes = set()
        for layer, (x, y, stds) in sorted(data_by_layer.items()):
            color = cmap[layer]
            # if self.fits[layer][2] <= 0.95:
            #     color = "red"
            # Plot measurements
            try:
                batch_sizes.update(x)
                ax.plot(x, y, label=layer, color=color, marker=".")
                ax.errorbar(x, y, yerr=stds, label=layer, color=color, marker=".")
            except ValueError as exc:
                self.logger.error("Plotting error %s", exc)
                self.logger.error("x: %s", x)
                self.logger.error("y: %s", y)

            # Plot cost model
            # interp_x = np.linspace(0, max(self.batch_sizes_to_load))
            # cost_model_pts = self.slopes_np[layer] * interp_x + self.intercepts_np[layer]
            # ax.plot(interp_x, cost_model_pts, ":", color=color)

        ax.legend(bbox_to_anchor=(1.04, 1))
        plt.xticks(sorted(list(batch_sizes)))

        fig.savefig(self.log_base / "!plot_costs.pdf", format='pdf', bbox_inches='tight')
        fig.savefig(self.log_base / "!plot_costs.png", bbox_inches='tight', dpi=300)

    @staticmethod
    def load_profile_s3(model_name: str, batch_size: int, platform: str) -> Optional[str]:
        local_base = pathlib.Path("/tmp") / "remat_cache" / "profiles"
        local_path = local_base / f"{model_name}_{batch_size}_{platform}.npy"
        remote_path = f"https://optimalcheckpointing.s3.amazonaws.com/profiles/{model_name}/b{batch_size}_{platform}.npy"
        if os.path.exists(local_path):
            try:
                _ = np.load(local_path)
                return local_path
            except Exception as e:
                logging.exception(e)
                logging.warning("Error loading cached profile solution, corrupt file? Reloading from S3")
        pathlib.Path(local_base).mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(remote_path, local_path)
            return local_path
        except urllib.error.HTTPError:
            return None
