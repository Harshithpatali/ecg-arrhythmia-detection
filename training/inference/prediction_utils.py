"""
Prediction helper functions.
"""

import numpy as np


def probability_dict(
    probabilities,
    labels,
):

    return {

        label: float(prob)

        for label, prob in zip(
            labels,
            probabilities,
        )

    }


def confidence_score(
    probabilities,
):

    return float(
        np.max(probabilities)
    )