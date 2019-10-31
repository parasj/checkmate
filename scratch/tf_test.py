import logging

import tensorflow as tf
from tensorflow.keras import layers

from tensorflow.python.ops import gradients_util as tfg

#tf.compat.v1.disable_eager_execution()
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
        #self.linear_2 = Linear(32)
        #self.linear_3 = Linear(1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        z = tf.nn.relu(x)
        y = tf.matmul(z,x)
        y = tf.reduce_mean(y)
        return y  #loss
@tf.function
def get_grads(f, inputs, tvars):
    y = f(inputs)
    return tf.gradients(y, tvars)

mlp = MLPBlock()

x = tf.ones(shape=(32, 32), name="x")
spec = tf.TensorSpec((32,32))
conc = tf.function(mlp).get_concrete_function(x)
grad_conc = get_grads.get_concrete_function(mlp, 
            x, 
            mlp.trainable_variables)
    
op_list = grad_conc.graph.get_operations()
op_dict = dict(zip(op_list, range(len(op_list))))

adj_list = {}
for op in op_list:
    in_id = op_dict[op]
    adj_list[in_id] = []
    for out in op.outputs:
        adj_list[in_id].append(op_dict[out])

#schedule



