import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization
from tqdm import tqdm

from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.definitions import checkmate_data_dir

logging.basicConfig(level=logging.INFO)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def get_data(batch_size=32):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds


def make_model():
    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.bn = BatchNormalization()
            self.conv1 = Conv2D(32, 3, activation="relu")
            self.flatten = Flatten()
            self.d1 = Dense(128, activation="relu")
            self.d2 = Dense(10, activation="softmax")

        def call(self, x):
            x = self.bn(x)
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)

    return MyModel()


def train_model(train_ds, test_ds, train_step, test_step, train_loss, train_accuracy, test_loss, test_accuracy, n_epochs=1):
    train_losses = []
    for epoch in range(n_epochs):
        # Reset the metrics at the start of the next epoch

        for images, labels in tqdm(train_ds, "Train", total=len(list(train_ds))):
            train_step(images, labels)
            train_losses.append(train_loss.result())

        for images, labels in tqdm(test_ds, "Test", total=len(list(test_ds))):
            test_step(images, labels)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        print(
            template.format(
                epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100
            )
        )
    return train_losses


def plot_losses(loss_curves):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")
    for loss_name, loss_data in loss_curves.items():
        plt.plot(loss_data, label=loss_name)
    plt.legend(loc="upper right")
    (checkmate_data_dir() / "exec").mkdir(parents=True, exist_ok=True)
    plt.savefig(checkmate_data_dir() / "exec" / "test.pdf")
    plt.savefig(checkmate_data_dir() / "exec" / "test.png")


def test_baseline(train_ds, test_ds, epochs=5):
    logging.info("Configuring basic MNIST model")
    model = make_model()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    logging.info("Training baseline model")
    orig_losses = train_model(
        test_ds, test_ds, train_step, test_step, train_loss, train_accuracy, test_loss, test_accuracy, n_epochs=epochs
    )
    return orig_losses


def test_checkpointed(train_ds, test_ds, solver, epochs=5):
    logging.info("Configuring basic MNIST model")
    model_check = make_model()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    logging.info("Building checkpointed model via checkmate")

    @tf.function(input_signature=train_ds.element_spec)
    def grads_check(images, labels):
        with tf.GradientTape() as check_tape:
            predictions = model_check(images)
            loss = loss_object(labels, predictions)
        gradients = check_tape.gradient(loss, model_check.trainable_variables)
        train_loss(loss)
        train_accuracy(labels, predictions)
        return gradients

    fn = grads_check.get_concrete_function()
    g = dfgraph_from_tf_function(fn)
    sqrtn_fn = edit_graph(fn, g.op_dict, solver(g).schedule)

    @tf.function
    def train_step_check(images, labels):
        gradients = sqrtn_fn(images, labels)
        optimizer.apply_gradients(zip(gradients, model_check.trainable_variables))

    @tf.function
    def test_step_check(images, labels):
        predictions = model_check(images)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    logging.info("Training checkpointed model")
    sqrtn_losses = train_model(
        test_ds,
        test_ds,
        train_step_check,
        test_step_check,
        train_loss,
        train_accuracy,
        test_loss,
        test_accuracy,
        n_epochs=epochs,
    )
    return sqrtn_losses


if __name__ == "__main__":
    train_ds, test_ds = get_data()
    EPOCHS = 1
    data = {
        "baseline": test_baseline(train_ds, test_ds, EPOCHS),
        "checkpoint_all": test_checkpointed(train_ds, test_ds, solve_checkpoint_all, epochs=EPOCHS),
        "checkpoint_sqrtn_ap": test_checkpointed(train_ds, test_ds, lambda g: solve_chen_sqrtn(g, False), epochs=EPOCHS),
    }
    plot_losses(data)
