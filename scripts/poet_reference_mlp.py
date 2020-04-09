# import dependencies
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load mnist dataset
batch_size = 1024
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
x_test, y_test = x_test.astype(np.float32), y_test.astype(np.float32)
print(x_train.shape)
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)


# Deine a new model
model = models.Sequential()
model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(120, activation='relu', use_bias=False))
model.add(layers.Dense(120, activation='relu', use_bias=False))
model.add(layers.Dense(100, activation='relu', use_bias=False))
model.add(layers.Dense(60, activation='relu', use_bias=False))
model.add(layers.Dense(10, activation='relu', use_bias=False))
model.add(layers.Dense(10))
loss = loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss=loss)

# Visualize the Model
model.summary()

# Train
print('Training')
model.fit(x_train, y_train, epochs=10)

#Test
print('Test')
model.evaluate(x_test,  y_test, verbose=2)
