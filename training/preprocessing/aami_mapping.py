"""
AAMI EC57 heartbeat mapping.

Maps MIT-BIH annotation symbols to the
five AAMI heartbeat classes.
"""

from typing import Optional

# MIT-BIH -> AAMI mapping
AAMI_MAPPING: dict[str, str] = {
    # -------------------------
    # Normal
    # -------------------------
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # -------------------------
    # Supraventricular
    # -------------------------
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # -------------------------
    # Ventricular
    # -------------------------
    "V": "V",
    "E": "V",

    # -------------------------
    # Fusion
    # -------------------------
    "F": "F",

    # -------------------------
    # Unknown / Paced / Others
    # -------------------------
    "/": "Q",
    "f": "Q",
    "Q": "Q",
}


VALID_SYMBOLS = set(AAMI_MAPPING.keys())


def map_symbol(symbol: str) -> Optional[str]:
    """
    Convert a MIT-BIH annotation symbol
    into an AAMI heartbeat class.

    Parameters
    ----------
    symbol : str

    Returns
    -------
    str | None
    """

    return AAMI_MAPPING.get(symbol)