"""
Backend wrapper around ML service.
"""

from training.prediction.prediction_service import (
    PredictionService
)


class BackendPredictionService:

    def __init__(self):

        self.service = (
            PredictionService()
        )

    def analyze(
        self,
        directory,
    ):

        return self.service.analyze_record(
            directory
        )