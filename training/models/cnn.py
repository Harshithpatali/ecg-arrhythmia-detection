"""
CNN Feature Extractor for ECG Arrhythmia Detection.

This module extracts local ECG morphology features
using stacked Conv1D blocks.
"""

from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    MaxPooling1D,
    Dropout,
)


def cnn_feature_extractor(inputs):
    """
    CNN feature extraction block.

    Parameters
    ----------
    inputs : keras.Input

    Returns
    -------
    Tensor
    """

    x = Conv1D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(inputs)

    x = BatchNormalization()(x)

    x = MaxPooling1D(
        pool_size=2,
    )(x)

    x = Dropout(0.20)(x)

    x = Conv1D(
        filters=64,
        kernel_size=5,
        padding="same",
        activation="relu",
    )(x)

    x = BatchNormalization()(x)

    x = MaxPooling1D(
        pool_size=2,
    )(x)

    x = Dropout(0.25)(x)

    x = Conv1D(
        filters=128,
        kernel_size=3,
        padding="same",
        activation="relu",
    )(x)

    x = BatchNormalization()(x)

    x = Dropout(0.30)(x)

    return x