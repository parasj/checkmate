import os
from typing import Optional

import graphviz
import numpy as np
import tensorflow as tf

from integration.tf2.extraction import extract_graph_from_keras
from remat.core.graph import Graph
from utils.setup_logger import setup_logger


class TF2ExtractorParams:
    # todo (paras) override self.g in case it is already computed (e.g. evaluation_old.py)
    # todo (paras) how does this handle compiled keras models
    # todo solve outside the constructor
    def __init__(self, keras_model: tf.keras.models.Model, batch_size: int = 1,
                 loss_cpu_cost: int = 0, loss_ram_cost: int = 4, log_base: str = None,
                 costs_np: Optional[np.ndarray] = None):
        self.log_base = log_base
        self.logger = setup_logger("TF2Wrapper", os.path.join(log_base, 'TF2Wrapper.log'))
        self.input_shape = list(keras_model.input_shape)
        self.input_shape[0] = batch_size
        self.output_shape = list(keras_model.output_shape)
        self.output_shape[0] = batch_size

        self.g: Graph = extract_graph_from_keras(keras_model, batch_size=batch_size, loss_cpu_cost=loss_cpu_cost,
                                                 loss_ram_cost=loss_ram_cost, costs_np=costs_np)
        self.logger.info(f"Extracted graph {keras_model.name} with {self.g.size} nodes")

        if self.log_base is not None:
            try:
                self.g.write_graphviz(log_base)
                tf.keras.utils.plot_model(keras_model, show_shapes=True, show_layer_names=True,
                                          to_file=os.path.join(log_base, "model.png"))
            except (FileNotFoundError, graphviz.ExecutableNotFound) as e:
                self.logger.exception(e)
                self.logger.warn("GraphViz is not installed, continuing without plotting.")
