import tensorflow as tf

from checkmate.tf2.extraction import dfgraph_from_tf_function
from checkmate.tf2.load_keras_model import get_keras_model


def test_mlpblock_extract():
    vgg = get_keras_model("VGG16")

    @tf.function
    def get_grads(inputs):
        y = vgg(inputs)
        y = tf.reduce_mean(y)
        return tf.gradients(y, vgg.trainable_variables)

    x = tf.ones(shape=(1, 224, 224, 3), name="input")
    grad_conc = get_grads.get_concrete_function(x)
    g = dfgraph_from_tf_function(grad_conc)
    # todo actually run some kind of tests here...
