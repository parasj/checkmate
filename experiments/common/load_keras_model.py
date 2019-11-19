from typing import Optional, List

import keras_segmentation
import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Activation, Lambda, Reshape

KERAS_APPLICATION_MODEL_NAMES = ['InceptionV3', 'VGG16', 'VGG19', 'ResNet50',
                                 'Xception', 'MobileNet', 'MobileNetV2', 'DenseNet121',
                                 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                                 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2',
                                 'ResNet152V2']
SEGMENTATION_MODEL_NAMES = list(keras_segmentation.models.model_from_name.keys())
MODEL_NAMES = KERAS_APPLICATION_MODEL_NAMES + SEGMENTATION_MODEL_NAMES + ["test", "bert"]
CHAIN_GRAPH_MODELS = ["VGG16", "VGG19", "MobileNet"]
NUM_SEGMENTATION_CLASSES = 19  # Cityscapes has 19 evaluation classes


def pretty_model_name(model_name: str):
    mapping = {
        "vgg_unet": "U-Net with VGG16",
    }
    if model_name in mapping:
        return mapping[model_name]
    return model_name


def simple_model():
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='in_conv')(inputs)
    a = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', name='conv1')(x)
    b = tf.keras.layers.Conv2D(128, (1, 1), activation='relu', name='conv2')(x)
    c = tf.keras.layers.Add(name='addc1c2')([a, b])
    d = tf.keras.layers.GlobalAvgPool2D(name='flatten')(c)
    predictions = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(d)
    return tf.keras.Model(inputs=inputs, outputs=predictions)


def get_keras_model(model_name: str, input_shape: Optional[List[int]] = None):
    if model_name == "test":
        model = simple_model()
    elif model_name == "bert":
        model = bertModel(12, 16, input_shape)
    elif model_name in KERAS_APPLICATION_MODEL_NAMES:
        model = eval("tf.keras.applications.{}".format(model_name))
        model = model(input_shape=input_shape)
    elif model_name in SEGMENTATION_MODEL_NAMES:
        model = keras_segmentation.models.model_from_name[model_name]
        if input_shape is not None:
            assert input_shape[2] == 3, "Can only segment 3-channel, channel-last images"
            model = model(n_classes=NUM_SEGMENTATION_CLASSES, input_height=input_shape[0], input_width=input_shape[1])
        else:
            model = model(n_classes=NUM_SEGMENTATION_CLASSES)
    else:
        raise NotImplementedError("Model {} not available".format(model_name))

    return model


def get_input_shape(model_name: str, batch_size: Optional[int] = None):
    model = get_keras_model(model_name, input_shape=None)
    shape = model.layers[0].input_shape
    if batch_size is not None:
        shape[0] = batch_size
    return shape

def bertModel(num_layers,  heads, input_size):
    
    hidden_size = input_size[1]
    intermediate_size = 4 * hidden_size
    seq_length = input_size[0]
    #batch = input_size[0]

    #config = BertConfig(hidden_size=hidden_size, 
    #                    num_hidden_layers=num_layers , 
    #                    num_attention_heads=heads, 
    #                    intermediate_size=4 * hidden_size,
    #                   hidden_act = "relu",
    #                   max_position_embeddings=seq_length)
    num_layers = num_layers
    layer = []
    inputs = keras.Input(shape=( input_size))
    mask = tf.fill(tf.shape(inputs), 1.0)
    tids = tf.fill(tf.shape(inputs), 0.0)
    x = inputs
    for i in range(num_layers):
        query = Dense(hidden_size, name = "query_{}".format(i))(x)
        key = Dense(hidden_size, name = "key_{}".format(i))(x)
        value = Dense(hidden_size, name = "value_{}".format(i))(x)
        query = Reshape( (heads, seq_length, hidden_size//heads))(query)
        key= Reshape( ( heads, hidden_size//heads, seq_length))(key)
        value = Reshape( (heads, seq_length, hidden_size//heads))(value)
        acts = Lambda(lambda x : tf.matmul(x[0],x[1]), name="acts_{}".format(i))([query, key])
        fin = Lambda(lambda x : tf.matmul(x[0], x[1]), name="fin_{}".format(i))([acts, value])
        fin = Reshape( ( seq_length, hidden_size))(fin)
        #layer.append(TFBertSelfAttention(config, name="layer_{}".format(i)))
        att = Dense(hidden_size, name="att_{}".format(i))(fin)
        relu = Activation("relu", name="relu0_{}".format(i))(att)
        x = LayerNormalization(name="f_att_{}".format(i))(relu + x)
        inter = Dense(intermediate_size,name = "inter_{}".format(i))(x)
        relu1 = Activation("relu", name="relu1_{}".format(i))(inter)
        shrink = Dense(hidden_size, name="shrink_{}".format(i))(relu1)
        relu2 = Activation("relu", name="relu2_{}".format(i))(shrink)
        x = LayerNormalization(name="layer_out_{}".format(i))(x + relu2)

    return keras.Model(inputs=inputs, outputs=x)
