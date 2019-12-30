import logging
import os
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from checkmate.core.solvers.strategy_checkpoint_all import solve_checkpoint_all
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from checkmate.core.utils.definitions import PathLike
from checkmate.tf2.execution import edit_graph
from checkmate.tf2.extraction import dfgraph_from_tf_function
from experiments.common.definitions import checkmate_data_dir
from experiments.common.load_keras_model import get_keras_model

logging.basicConfig(level=logging.INFO)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

solve_chen_sqrtn_noap = lambda g: solve_chen_sqrtn(g, False)
solve_chen_sqrtn_ap = lambda g: solve_chen_sqrtn(g, True)


def get_data(dataset: str, batch_size=32):
    if dataset in ["mnist", "cifar10", "cifar100"]:
        dataset = eval("tf.keras.datasets.{}".format(dataset))
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        if dataset is "mnist":
            x_train = x_train[..., tf.newaxis]
            x_test = x_test[..., tf.newaxis]
        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
        return train_ds, test_ds
    else:
        raise ValueError("Invalid dataset " + str(dataset))


def make_model(dataset: str, model: str = "test"):
    if model == "test":
        from tensorflow.keras import Model
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization

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
    else:
        shapes = {
            "mnist": ((28, 28, 1), 10),
            "cifar10": ((32, 32, 3), 10),
            "cifar100": ((32, 32, 3), 100),
            "imagenet": ((224, 224, 3), 1000),
        }
        return get_keras_model(model, input_shape=shapes[dataset][0], num_classes=shapes[dataset][1])


def train_model(train_ds, test_ds, train_step, test_step, train_loss, train_accuracy, test_loss, test_accuracy, n_epochs=1):
    train_losses = []
    for epoch in range(n_epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in tqdm(train_ds, "Train", total=len(list(train_ds))):
            train_step(images, labels)
            train_losses.append(train_loss.result())

        for images, labels in tqdm(test_ds, "Test", total=len(list(test_ds))):
            test_step(images, labels)

        print(
            f"Epoch {epoch + 1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}, ",
            f"Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result() * 100}",
        )
    return train_losses


def _build_model_via_solver(dataset: str, model_name: str, train_signature, solver):
    logging.info("Configuring model " + str(model_name))
    model_check = make_model(dataset, model_name)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    test_loss = tf.keras.metrics.Mean(name="test_loss")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    logging.info("Building checkpointed model via checkmate")

    @tf.function(input_signature=train_signature)
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

    return sqrtn_fn, train_step_check, test_step_check, train_loss, train_accuracy, test_loss, test_accuracy


def save_checkpoint_chrome_trace(dataset: str, model_name: str, log_base: PathLike, batch_size: int = 32):
    def trace_solver_solution(save_path: PathLike, train_ds, solver):
        import tensorflow.compat.v1 as tf1
        from tensorflow.python.client import timeline

        data_iter = train_ds.__iter__()
        data_list = [x.numpy() for x in data_iter.next()]
        with tf1.Session() as sess:
            sqrtn_fn, *_ = _build_model_via_solver(dataset, model_name, train_ds.element_spec, solver)
            out = sqrtn_fn(*[tf1.convert_to_tensor(x) for x in data_list])

            run_meta = tf1.RunMetadata()
            sess.run(tf1.global_variables_initializer())
            sess.run(out, options=tf1.RunOptions(trace_level=tf1.RunOptions.FULL_TRACE), run_metadata=run_meta)
            t1 = timeline.Timeline(run_meta.step_stats)
            lctf = t1.generate_chrome_trace_format()

        with Path(save_path).open("w") as f:
            f.write(lctf)

    log_base = Path(log_base)
    log_base.mkdir(parents=True, exist_ok=True)
    train_ds, test_ds = get_data(dataset, batch_size=batch_size)
    trace_solver_solution(log_base / "check_all.json", train_ds, solve_checkpoint_all)
    trace_solver_solution(log_base / "check_sqrtn_noap.json", train_ds, solve_chen_sqrtn_noap)


def compare_checkpoint_loss_curves(dataset: str = "mnist", model_name: str = "test", n_epochs: int = 1, batch_size: int = 32):
    def test_baseline(train_ds, test_ds, epochs=5):
        logging.info("Configuring basic MNIST model")
        model = make_model(dataset=dataset, model=model_name)
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
            train_ds, test_ds, train_step, test_step, train_loss, train_accuracy, test_loss, test_accuracy, n_epochs=epochs
        )
        return orig_losses

    def test_checkpointed(train_ds, test_ds, solver, epochs=1):
        check_model = _build_model_via_solver(dataset, model_name, train_ds.element_spec, solver)
        _, train_step_check, test_step_check, train_loss, train_accuracy, test_loss, test_accuracy = check_model
        logging.info("Training checkpointed model")
        sqrtn_losses = train_model(
            train_ds,
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

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("darkgrid")

    train_ds, test_ds = get_data(dataset, batch_size=batch_size)
    data = {
        "baseline": (test_baseline(train_ds, test_ds, n_epochs)),
        "checkpoint_all": (test_checkpointed(train_ds, test_ds, solve_checkpoint_all, epochs=n_epochs)),
        "checkpoint_sqrtn_ap": (test_checkpointed(train_ds, test_ds, solve_chen_sqrtn_ap, epochs=n_epochs)),
        # "checkpoint_sqrtn_noap": (test_checkpointed(train_ds, test_ds, solve_chen_sqrtn_noap, epochs=n_epochs)),
    }

    for loss_name, loss_data in data.items():
        plt.plot(loss_data, label=loss_name)
    plt.legend(loc="upper right")
    (checkmate_data_dir() / "exec").mkdir(parents=True, exist_ok=True)
    plt.savefig(checkmate_data_dir() / "exec" / "{}_{}_bs{}_epochs{}.pdf".format(dataset, model_name, batch_size, n_epochs))


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity("ERROR")
    # save_checkpoint_chrome_trace(checkmate_data_dir() / "profile_exec")
    # compare_checkpoint_loss_curves(dataset='mnist', model_name='test', n_epochs=1)
    compare_checkpoint_loss_curves(dataset="cifar10", model_name="ResNet50", n_epochs=1, batch_size=1)
