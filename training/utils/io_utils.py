"""
Input/output helper functions.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def save_numpy(array: np.ndarray, path: str | Path) -> None:
    """
    Save NumPy array.
    """
    np.save(path, array)


def load_numpy(path: str | Path) -> np.ndarray:
    """
    Load NumPy array.
    """
    return np.load(path, allow_pickle=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save DataFrame to CSV.
    """
    df.to_csv(path, index=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load CSV file.
    """
    return pd.read_csv(path)