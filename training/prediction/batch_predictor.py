"""
Batch heartbeat prediction.
"""

from __future__ import annotations

import numpy as np

from training.inference.model_loader import ModelLoader
from training.utils.serialization import load_pickle
from configs.config import LABEL_ENCODER_PATH


class BatchPredictor:

    def __init__(self):

        self.model = ModelLoader.get_model()

        self.encoder = load_pickle(
            LABEL_ENCODER_PATH
        )

    def predict(
        self,
        beats: np.ndarray,
    ):

        beats = beats.reshape(
            -1,
            180,
            1,
        )

        print(f"beats shape = {beats.shape}")
        print("STARTING MODEL PREDICT")

        probabilities = self.model.predict(
            beats,
            verbose=0,
            batch_size=16,
        )

        print("MODEL PREDICT FINISHED")

        prediction_indices = np.argmax(
            probabilities,
            axis=1,
        )

        predictions = self.encoder.inverse_transform(
            prediction_indices
        )

        confidences = np.max(
            probabilities,
            axis=1,
        )

        print(f"Total predictions = {len(predictions)}")
        print(f"Average confidence = {np.mean(confidences):.4f}")

        return (
            predictions.tolist(),
            confidences.tolist(),
            probabilities.tolist(),
        )