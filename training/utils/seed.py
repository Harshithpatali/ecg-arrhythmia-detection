"""
Random seed utilities.
"""

import random
import numpy as np
import tensorflow as tf

from configs.constants import RANDOM_STATE


def set_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)