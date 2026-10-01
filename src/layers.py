import tensorflow as tf
from tensorflow.keras import layers


class FixedRFFT(layers.Layer):
    def call(self, inputs):
        return tf.abs(tf.signal.rfft(inputs))


class WeightedRFFT(layers.Layer):
    def build(self, input_shape):
        self.freq_weights = self.add_weight(
            name="freq_w",
            shape=(input_shape[-1] // 2 + 1,),
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, inputs):
        magnitude = tf.abs(tf.signal.rfft(inputs))
        return magnitude * self.freq_weights
