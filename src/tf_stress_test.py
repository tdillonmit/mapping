import tensorflow as tf
import numpy as np
data = np.random.random((1000, 1000, 3)).astype(np.float32)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), input_shape=(1000, 1000, 3))])
for _ in range(1000):
    model.predict(np.expand_dims(data, axis=0))