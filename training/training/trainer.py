"""
Model Trainer.
"""

import pickle
from pathlib import Path

import numpy as np

from configs.config import (
    MODEL_DIR,
)

from configs.constants import (
    BATCH_SIZE,
    EPOCHS,
)

from training.models.cnn_bilstm_attention import (
    build_model,
)

from training.training.callbacks import (
    get_callbacks,
)


class ModelTrainer:

    def __init__(self):

        self.model = build_model()

    def train(self):

        data_dir = Path("data/processed")

        X_train = np.load(
            data_dir / "X_train_dl.npy"
        )

        y_train = np.load(
            data_dir / "y_train_dl.npy"
        )

        X_val = np.load(
            data_dir / "X_val_dl.npy"
        )

        y_val = np.load(
            data_dir / "y_val_dl.npy"
        )

        history = self.model.fit(

            X_train,

            y_train,

            validation_data=(

                X_val,

                y_val,

            ),

            batch_size=BATCH_SIZE,

            epochs=EPOCHS,

            callbacks=get_callbacks(),

            verbose=1,

        )

        with open(

            Path(MODEL_DIR) / "history.pkl",

            "wb",

        ) as file:

            pickle.dump(

                history.history,

                file,

            )

        return history