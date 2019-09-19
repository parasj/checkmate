from typing import Optional, List

import keras_segmentation
import tensorflow.compat.v2 as tf

try:
    from tensorflow.python.keras.utils.layer_utils import count_params  # TF r2.0
except ImportError as e:
    from tensorflow.keras.backend import count_params  # TF r1.14

KERAS_APPLICATION_MODEL_NAMES = ['InceptionV3', 'VGG16', 'VGG19', 'ResNet50',
                                 'Xception', 'MobileNet', 'MobileNetV2', 'DenseNet121',
                                 'DenseNet169', 'DenseNet201', 'NASNetMobile', 'NASNetLarge',
                                 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2',
                                 'ResNet152V2']
SEGMENTATION_MODEL_NAMES = list(keras_segmentation.models.model_from_name.keys())
MODEL_NAMES = KERAS_APPLICATION_MODEL_NAMES + SEGMENTATION_MODEL_NAMES + ["test"]

CHAIN_GRAPH_MODELS = ["VGG16", "VGG19", "MobileNet"]

NUM_SEGMENTATION_CLASSES = 19  # Cityscapes has 19 evaluation classes


def pretty_model_name(model_name: str):
    mapping = {
        "vgg_unet": "U-Net with VGG16",
    }
    if model_name in mapping:
        return mapping[model_name]
    return model_name


def pretty_platform_name(platform: str):
    mapping = {
        "p32xlarge": "V100",
        "p32xlarge_fp16": "V100, fp16",
        "p2xlarge": "K80",
        "flops": "FLOPs",
    }
    if platform in mapping:
        return mapping[platform]
    return platform


def platform_memory(platform: str):
    mapping = {
        "p32xlarge": 16 * 1000 * 1000 * 1000,
        "p32xlarge_fp16": 16 * 1000 * 1000 * 1000,
        "p2xlarge": 12 * 1000 * 1000 * 1000,
        "flops": 12 * 1000 * 1000 * 1000,
    }
    if platform in mapping:
        return mapping[platform]
    return platform


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
    if model_name is "test":
        model = simple_model()
    elif model_name in KERAS_APPLICATION_MODEL_NAMES:
        # Pre-trained Keras applications
        model = eval("tf.keras.applications.{}".format(model_name))
        model = model(input_shape=input_shape)
    elif model_name in SEGMENTATION_MODEL_NAMES:
        # Segmentation models
        model = keras_segmentation.models.model_from_name[model_name]
        if input_shape is not None:
            assert input_shape[2] == 3, "Can only segment 3-channel, channel-last images"

            model = model(n_classes=NUM_SEGMENTATION_CLASSES,
                          input_height=input_shape[0],
                          input_width=input_shape[1])
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


def count_params_keras(model: tf.keras.models.Model):
    model._check_trainable_weights_consistency()
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    elif hasattr(model, '_unique_trainable_weights'):
        trainable_count = count_params(model._unique_trainable_weights)  # TF r2.0
    else:
        trainable_count = count_params(model.trainable_weights)  # TF r1.14
    # print("Trainable params:", trainable_count)

    non_trainable_count = count_params(model.non_trainable_weights)
    # print("Non-trainable params:", non_trainable_count)
    return trainable_count, non_trainable_count


