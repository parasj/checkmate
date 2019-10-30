import logging

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.python.ops import gradients_util as tfg

tf.compat.v1.disable_eager_execution()
logging.basicConfig(level=logging.DEBUG)



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
    x = tf.ones(shape=(3, 64), name="x")
    y = mlp(x)
    loss = tf.reduce_mean(y, name="loss")
    logging.debug(f'loss: {loss}')

    dldy = tf.compat.v2.gradients(loss, y, name="dldy")
    dldtheta = tf.compat.v2.gradients(y, mlp.trainable_variables, dldy, name="dldtheta")
    print(dldtheta)

    # del loss
    # loss = tf.reduce_mean(mlp(x))
    # dldtheta = tf.gradients
    # out_g = dldtheta(loss, mlp.trainable_weights[0])

    import tfgraphviz
    g = tfgraphviz.board(tf.compat.v1.get_default_graph())
    g.view()

    # with tf.compat.v1.Session() as sess:
    #     sess.run(out_g)

if __name__ == "__main__":
    main()
