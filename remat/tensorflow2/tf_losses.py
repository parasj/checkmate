import tensorflow as tf


# noinspection PyUnresolvedReferences
def categorical_cross_entropy(pred_logits, labels, model_losses=[]):
    loss = tf.keras.losses.categorical_crossentropy(labels, pred_logits, from_logits=True)
    loss += 0 if not model_losses else tf.add_n(model_losses)  # regularization
    return loss
