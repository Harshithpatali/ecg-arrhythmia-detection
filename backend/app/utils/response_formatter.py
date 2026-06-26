"""
API response formatting.
"""


class ResponseFormatter:

    @staticmethod
    def format(
        prediction_result
    ):

        return {

            "metadata":
                prediction_result[
                    "metadata"
                ],

            "summary":
                prediction_result[
                    "summary"
                ],

            "beat_predictions":
                prediction_result[
                    "beat_predictions"
                ],

            "confidences":
                prediction_result[
                    "confidences"
                ],
        }