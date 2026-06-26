"""
Prediction Service.
"""

from training.ecg_processing.ecg_service import (
    ECGProcessingService,
)

from training.prediction.batch_predictor import (
    BatchPredictor,
)

from training.prediction.statistics import (
    PredictionStatistics,
)

from training.prediction.risk_analyzer import (
    RiskAnalyzer,
)


class PredictionService:

    def __init__(self):

        self.processor = (
            ECGProcessingService()
        )

        self.predictor = (
            BatchPredictor()
        )

    def analyze_record(
        self,
        directory,
    ):

        processed = (
            self.processor
            .process_directory(
                directory
            )
        )

        beats = (
            processed["beats"]
        )

        (
            predictions,
            confidences,
            probabilities,
        ) = (

            self.predictor.predict(
                beats
            )

        )

        distribution = (

            PredictionStatistics
            .class_distribution(
                predictions
            )

        )

        percentages = (

            PredictionStatistics
            .class_percentages(
                predictions
            )

        )

        risk = (
            RiskAnalyzer.assess(
                percentages
            )
        )

        return {

            "metadata":
                processed[
                    "metadata"
                ],

            "summary": {

                "distribution":
                    distribution,

                "percentages":
                    percentages,

                "risk":
                    risk,
            },

            "beat_predictions":
                predictions,

            "confidences":
                confidences,

            "probabilities":
                probabilities,
        }