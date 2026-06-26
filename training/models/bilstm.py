"""
BiLSTM Sequence Learning Module for ECG Arrhythmia Detection.

This module learns temporal dependencies from the
CNN feature maps using stacked Bidirectional LSTMs.
"""

from tensorflow.keras.layers import (
    Bidirectional,
    LSTM,
    Dropout,
)


def bilstm_feature_extractor(inputs):
    """
    BiLSTM feature extraction block.

    Parameters
    ----------
    inputs : keras Tensor
        Shape: (batch, time_steps, features)

    Returns
    -------
    keras Tensor
        Shape: (batch, time_steps, 128)
    """

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True,
        )
    )(inputs)

    x = Dropout(0.30)(x)

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True,
        )
    )(x)

    x = Dropout(0.30)(x)

    return x