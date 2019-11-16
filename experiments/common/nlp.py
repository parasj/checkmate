import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from transformers.modeling_tf_bert import TFBertSelfAttention
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Activation
from transformers.configuration_bert import BertConfig

#unroll the structure manually 
class BertModel(keras.layers.Layer):
    
    def __init__(self, num_layers, seq_length, hidden_size, heads):
        super(BertModel, self).__init__()
        config = BertConfig(hidden_size=hidden_size, 
                            num_hidden_layers=num_layers , 
                            num_attention_heads=heads, 
                            intermediate_size=4 * hidden_size,
                           hidden_act = "relu",
                           max_position_embeddings=seq_length)
        self.num_layers = num_layers
        self.layers = []
        for i in range(num_layers):
            self.layers.append(TFBertSelfAttention(config))
            self.layers.append(Dense(config.hidden_size))
            self.layers.append(LayerNormalization())
            self.layers.append(Dense(config.intermediate_size))
            self.layers.append(Activation("relu"))
            self.layers.append(Dense(config.hidden_size))
            self.layers.append(LayerNormalization())
        
    def call(self, inputs):
        mask = tf.fill(tf.shape(inputs), 1.0)
        tids = tf.fill(tf.shape(inputs), 0.0)
        x = inputs
        for i in range(0, 7 * self.num_layers, 7):
            lyr = self.layers[i:i+7]
            att = lyr[0]((x, None, None))[0]
            x = lyr[2](x + lyr[1](att))
            x = lyr[6](x + lyr[5](lyr[4](lyr[3](x))))
        return x
