"""
Complete CNN + BiLSTM + Attention Model for
ECG Arrhythmia Detection.
"""

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from training.models.cnn import (
    cnn_feature_extractor,
)

from training.models.bilstm import (
    bilstm_feature_extractor,
)

from training.models.attention import (
    AttentionLayer,
)


def build_model(
    input_shape=(180, 1),
    num_classes=5,
    learning_rate=1e-3,
):
    """
    Build CNN + BiLSTM + Attention model.

    Parameters
    ----------
    input_shape : tuple

    num_classes : int

    learning_rate : float

    Returns
    -------
    keras.Model
    """

    inputs = Input(shape=input_shape)

    # ----------------------------------
    # CNN Feature Extraction
    # ----------------------------------

    x = cnn_feature_extractor(inputs)

    # ----------------------------------
    # BiLSTM Sequence Learning
    # ----------------------------------

    x = bilstm_feature_extractor(x)

    # ----------------------------------
    # Attention
    # ----------------------------------

    x = AttentionLayer()(x)

    # ----------------------------------
    # Classification Head
    # ----------------------------------

    x = Dense(
        128,
        activation="relu",
    )(x)

    x = Dropout(0.50)(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
        name="prediction",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN_BiLSTM_Attention",
    )

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
    )

    return model