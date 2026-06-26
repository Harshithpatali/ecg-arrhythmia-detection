"""
Custom Attention Layer for CNN + BiLSTM ECG Classification.

This layer learns attention weights over the BiLSTM
time-step outputs and produces a context vector.

Input Shape
-----------
(batch_size, time_steps, features)

Output Shape
------------
(batch_size, features)
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import Layer


class AttentionLayer(Layer):
    """
    Bahdanau-style additive attention layer.

    Learns which time steps are most informative
    for heartbeat classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Create trainable parameters.
        """

        feature_dim = input_shape[-1]

        self.W = self.add_weight(
            name="attention_weight",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )

        self.u = self.add_weight(
            name="attention_context",
            shape=(feature_dim,),
            initializer="glorot_uniform",
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs):
        """
        Forward pass.

        Parameters
        ----------
        inputs:
            Shape (batch, time_steps, features)

        Returns
        -------
        Context vector
            Shape (batch, features)
        """

        # Alignment scores
        score = tf.tanh(
            tf.tensordot(inputs, self.W, axes=1) + self.b
        )

        # Importance of each time step
        attention_weights = tf.nn.softmax(
            tf.tensordot(score, self.u, axes=1),
            axis=1,
        )

        # Expand dimensions for broadcasting
        attention_weights = tf.expand_dims(
            attention_weights,
            axis=-1,
        )

        # Weighted sum
        context = tf.reduce_sum(
            inputs * attention_weights,
            axis=1,
        )

        return context

    def get_config(self):
        """
        Required for saving/loading the model.
        """

        config = super().get_config()

        return config