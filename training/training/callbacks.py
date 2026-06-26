"""
Training callbacks.
"""

from pathlib import Path

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    CSVLogger,
)

from configs.config import MODEL_DIR


def get_callbacks():
    """
    Create training callbacks.
    """

    Path(MODEL_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    callbacks = [

        ModelCheckpoint(
            filepath=str(
                Path(MODEL_DIR) / "best_model.keras"
            ),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),

        CSVLogger(
            str(
                Path(MODEL_DIR) / "training_log.csv"
            )
        ),

    ]

    return callbacks