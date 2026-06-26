"""
Single heartbeat prediction.
"""

from __future__ import annotations

import numpy as np

from training.inference.model_loader import (
    ModelLoader,
)

from training.utils.serialization import (
    load_pickle,
)

from configs.config import (
    LABEL_ENCODER_PATH,
)

from training.inference.prediction_utils import (
    confidence_score,
    probability_dict,
)


class BeatPredictor:

    def __init__(self):

        self.model = (
            ModelLoader.get_model()
        )

        self.encoder = load_pickle(
            LABEL_ENCODER_PATH
        )

    def predict(
        self,
        beat: np.ndarray,
    ) -> dict:

        beat = beat.reshape(
            1,
            180,
            1,
        )

        probabilities = self.model.predict(
            beat,
            verbose=0,
        )[0]

        prediction_index = int(
            np.argmax(probabilities)
        )

        predicted_class = (
            self.encoder.inverse_transform(
                [prediction_index]
            )[0]
        )

        return {

            "predicted_class":
                predicted_class,

            "confidence":
                confidence_score(
                    probabilities
                ),

            "probabilities":
                probability_dict(
                    probabilities,
                    self.encoder.classes_,
                ),
        }