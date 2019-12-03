import tensorflow as tf


def random_batch(
    batch_size,
    data_format="channels_last",
    num_classes=1000,
    img_h=224,
    img_w=224,
    num_channels=3,
):
    shape = (
        (num_channels, img_h, img_w)
        if data_format == "channels_first"
        else (img_h, img_w, num_channels)
    )
    shape = (batch_size,) + shape
    images = tf.keras.backend.random_uniform(shape)
    labels = tf.keras.backend.random_uniform(
        [batch_size], minval=0, maxval=num_classes, dtype=tf.int32
    )
    one_hot = tf.one_hot(labels, num_classes)
    return images, one_hot
