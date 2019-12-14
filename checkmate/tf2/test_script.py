from extraction import dfgraph_from_tf_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.resnet import ResNet50
from checkmate.core.solvers.strategy_chen import solve_chen_sqrtn
from execution import edit_graph

samp_inp = keras.layers.Input(shape=(224,224,3))
resnet = ResNet50(input_tensor=samp_inp)
fxn =tf.function(resnet).get_concrete_function(inputs=samp_inp)
g = dfgraph_from_tf_function(fxn)
sqrtn = solve_chen_sqrtn(g, False)[3]
new_fxn = edit_graph(fxn, g.op_dict, sqrtn)
