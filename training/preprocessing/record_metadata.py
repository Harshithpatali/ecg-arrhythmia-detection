"""
Record metadata utilities.
"""

from typing import Any


def summarize_metadata(
    metadata: dict[str, Any],
) -> str:
    """
    Create a readable summary.

    Parameters
    ----------
    metadata : dict

    Returns
    -------
    str
    """

    return (
        f"Record : {metadata['record_name']}\n"
        f"Sampling Frequency : {metadata['sampling_frequency']} Hz\n"
        f"Channels : {', '.join(metadata['channels'])}\n"
        f"Length : {metadata['signal_length']} samples\n"
    )