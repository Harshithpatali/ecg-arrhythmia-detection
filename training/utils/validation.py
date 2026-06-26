"""
Validation utilities.
"""

import numpy as np

from configs.constants import BEAT_LENGTH


def validate_heartbeat(
    heartbeat: np.ndarray,
) -> bool:
    """
    Validate heartbeat shape and values.

    Parameters
    ----------
    heartbeat : np.ndarray

    Returns
    -------
    bool
    """
    if heartbeat is None:
        return False

    if not isinstance(heartbeat, np.ndarray):
        return False

    if heartbeat.shape[0] != BEAT_LENGTH:
        return False

    if np.isnan(heartbeat).any():
        return False

    if np.isinf(heartbeat).any():
        return False

    return True