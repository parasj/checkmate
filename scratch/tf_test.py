import logging

import tensorflow as tf
from tensorflow.keras import layers

logging.basicConfig(level=logging.DEBUG)

# tf.compat.v1.disable_eager_execution()


class Linear(layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class MLPBlock(layers.Layer):
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        return x


def main():
    mlp = MLPBlock()
    x = tf.ones(shape=(3, 64))
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = mlp(x)
        loss = tf.reduce_mean(y)
    logging.debug(f'x: {x}')
    logging.debug(f'y: {y}')
    logging.debug(f'loss: {loss}')
    logging.debug(f'weights: {len(mlp.weights)}')
    logging.debug(f'trainable weights: {len(mlp.trainable_weights)}')

    @tf.function
    def grad(ys, xs):
        return tape.gradient(ys, xs)

    # first backprop loss to last activation
    logging.debug(f'dldy ref: {tape.gradient(loss, mlp.trainable_weights[0])}')
    logging.debug(f'dldy: {grad(loss, mlp.trainable_weights[0])}')
    logging.debug(f'dldy: {grad(loss, mlp.trainable_weights[0])}')


if __name__ == "__main__":
    main()
