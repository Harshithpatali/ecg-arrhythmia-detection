"""
Full ECG Record Prediction.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from training.preprocessing.heartbeat_segmenter import (
    HeartbeatSegmenter,
)

from training.preprocessing.normalizer import (
    HeartbeatNormalizer,
)

from training.inference.beat_predictor import (
    BeatPredictor,
)


class FullRecordPredictor:

    def __init__(self):

        self.segmenter = (
            HeartbeatSegmenter()
        )

        self.predictor = (
            BeatPredictor()
        )

    def predict_record(
        self,
        signal: np.ndarray,
        annotations: pd.DataFrame,
    ):

        beats, _ = self.segmenter.segment(

            signal,

            annotations,

        )

        beats = (
            HeartbeatNormalizer.normalize(
                beats
            )
        )

        predictions = []

        for beat in beats:

            result = self.predictor.predict(
                beat
            )

            predictions.append(
                result["predicted_class"]
            )

        summary = dict(
            Counter(predictions)
        )

        return {

            "total_beats":
                len(predictions),

            "class_distribution":
                summary,

            "predictions":
                predictions,
        }