"""
File utility functions.
"""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
    """
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory