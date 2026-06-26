"""
Heartbeat segmentation.

Extract fixed-length heartbeat windows
centered on R-peaks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from configs.constants import (
    LEFT_WINDOW,
    RIGHT_WINDOW,
)
from configs.logging_config import logger


class HeartbeatSegmenter:
    """
    Segment ECG recordings into
    heartbeat windows.
    """

    def __init__(
        self,
        left_window: int = LEFT_WINDOW,
        right_window: int = RIGHT_WINDOW,
    ):
        self.left_window = left_window
        self.right_window = right_window

    def segment(
        self,
        signal: np.ndarray,
        annotations: pd.DataFrame,
        channel: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract heartbeat windows.

        Parameters
        ----------
        signal : np.ndarray
            ECG signal of shape (samples, channels).

        annotations : pd.DataFrame
            Parsed annotation table.

        channel : int
            ECG channel index.

        Returns
        -------
        beats : np.ndarray
            Shape (N, beat_length)

        labels : np.ndarray
            Shape (N,)
        """

        beats = []
        labels = []

        ecg = signal[:, channel]

        signal_length = len(ecg)

        skipped = 0

        for _, row in annotations.iterrows():

            peak = int(row["sample"])

            start = peak - self.left_window
            end = peak + self.right_window

            if start < 0 or end > signal_length:
                skipped += 1
                continue

            beat = ecg[start:end]

            if len(beat) != self.left_window + self.right_window:
                skipped += 1
                continue

            beats.append(beat.astype(np.float32))
            labels.append(row["label"])

        logger.info(
            "Segmented %d beats (%d skipped).",
            len(beats),
            skipped,
        )

        return (
            np.asarray(beats),
            np.asarray(labels),
        )