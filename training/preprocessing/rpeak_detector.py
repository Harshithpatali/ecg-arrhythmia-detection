"""
R-Peak Detection.
"""

from __future__ import annotations

import numpy as np

from scipy.signal import find_peaks


class RPeakDetector:
    """
    Detect R-peaks from ECG signal.
    """

    def detect(
        self,
        signal: np.ndarray,
        sampling_rate: int = 360,
    ) -> np.ndarray:

        if signal.ndim == 2:
            signal = signal[:, 0]

        peaks, _ = find_peaks(
            signal,
            distance=int(
                0.25 * sampling_rate
            ),
            prominence=0.3,
        )

        return peaks