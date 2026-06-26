"""
Heartbeat normalization.
"""

from __future__ import annotations

import numpy as np


class HeartbeatNormalizer:
    """
    Normalize heartbeat signals.
    """

    @staticmethod
    def z_score(
        beats: np.ndarray,
    ) -> np.ndarray:
        """
        Z-score normalization.

        Each heartbeat is normalized independently.
        """

        mean = np.mean(
            beats,
            axis=1,
            keepdims=True,
        )

        std = np.std(
            beats,
            axis=1,
            keepdims=True,
        )

        std[std == 0] = 1.0

        return (beats - mean) / std

    @staticmethod
    def min_max(
        beats: np.ndarray,
    ) -> np.ndarray:
        """
        Min-Max normalization.
        """

        minimum = np.min(
            beats,
            axis=1,
            keepdims=True,
        )

        maximum = np.max(
            beats,
            axis=1,
            keepdims=True,
        )

        denominator = maximum - minimum
        denominator[denominator == 0] = 1.0

        return (beats - minimum) / denominator

    @staticmethod
    def normalize(
        beats: np.ndarray,
        method: str = "zscore",
    ) -> np.ndarray:
        """
        Normalize heartbeat dataset.

        Parameters
        ----------
        beats : np.ndarray

        method : str
            "zscore" or "minmax"
        """

        method = method.lower()

        if method == "zscore":
            return HeartbeatNormalizer.z_score(beats)

        if method == "minmax":
            return HeartbeatNormalizer.min_max(beats)

        raise ValueError(
            f"Unsupported normalization method: {method}"
        )