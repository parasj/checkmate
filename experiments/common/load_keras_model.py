import logging
import re
from typing import Optional, List

import tensorflow as tf

KERAS_APPLICATION_MODEL_NAMES = ['InceptionV3', 'VGG16', 'VGG19', 'ResNet50',
                                 'Xception', 'MobileNet', 'MobileNetV2', 'DenseNet121',
                                 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                                 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2',
                                 'ResNet152V2']
LINEAR_MODEL_NAMES = ["linear" + str(i) for i in range(32)]
MODEL_NAMES = KERAS_APPLICATION_MODEL_NAMES + ["test"] + LINEAR_MODEL_NAMES
CHAIN_GRAPH_MODELS = ["VGG16", "VGG19", "MobileNet"] + LINEAR_MODEL_NAMES

try:
    import keras_segmentation
    SEGMENTATION_MODEL_NAMES = list(keras_segmentation.models.model_from_name.keys())
    MODEL_NAMES.extend(SEGMENTATION_MODEL_NAMES)
    NUM_SEGMENTATION_CLASSES = 19  # Cityscapes has 19 evaluation classes
except ImportError as e:
    logging.exception(e)
    logging.error("Failed to load segmentation model names, skipping...")
    SEGMENTATION_MODEL_NAMES = []


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
    d = tf.keras.layers.GlobalAveragePooling2D(name='flatten')(c)
    predictions = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(d)
    return tf.keras.Model(inputs=inputs, outputs=predictions)


def linear_model(i):
    input = tf.keras.Input(shape=(224, 224, 3))
    x = input
    for i in range(i):
        x = tf.keras.layers.Conv2D(64, (3, 3), activation=None, use_bias=False, name='conv' + str(i))(x)
    d = tf.keras.layers.GlobalAveragePooling2D(name='flatten')(x)
    predictions = tf.keras.layers.Dense(1000, activation='softmax', name='predictions')(d)
    return tf.keras.Model(inputs=input, outputs=predictions)


def get_keras_model(model_name: str, input_shape: Optional[List[int]] = None):
    if model_name == "test":
        model = simple_model()
    elif model_name in LINEAR_MODEL_NAMES:
        i = int(re.search(r'\d+$', model_name).group())
        model = linear_model(i)
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
