"""
Prediction statistics.
"""

from collections import Counter


class PredictionStatistics:

    @staticmethod
    def class_distribution(
        predictions
    ):

        counts = Counter(
            predictions
        )

        return dict(counts)

    @staticmethod
    def class_percentages(
        predictions
    ):

        total = len(predictions)

        counts = Counter(
            predictions
        )

        return {

            cls: round(
                count / total * 100,
                2
            )

            for cls, count
            in counts.items()
        }