"""
Utility functions.
"""

import numpy as np


def average_confidence(
    confidences,
):

    return round(
        float(np.mean(confidences)),
        4,
    )


def max_confidence(
    confidences,
):

    return round(
        float(np.max(confidences)),
        4,
    )


def min_confidence(
    confidences,
):

    return round(
        float(np.min(confidences)),
        4,
    )