"""
Reshape heartbeat dataset for CNN input.
"""

from __future__ import annotations

import numpy as np


class DatasetReshaper:
    """
    Reshape heartbeat datasets for deep learning.
    """

    @staticmethod
    def reshape(X: np.ndarray) -> np.ndarray:
        """
        Convert:

        (N,180)

        into

        (N,180,1)
        """

        return X.reshape(
            X.shape[0],
            X.shape[1],
            1,
        ).astype(np.float32)