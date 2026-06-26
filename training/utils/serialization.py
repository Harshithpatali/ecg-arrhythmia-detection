"""
Serialization utilities.
"""

import json
import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, path: str | Path) -> None:
    """
    Save object as pickle.
    """
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: str | Path) -> Any:
    """
    Load pickle object.
    """
    with open(path, "rb") as file:
        return pickle.load(file)


def save_json(data: dict, path: str | Path) -> None:
    """
    Save dictionary as JSON.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def load_json(path: str | Path) -> dict:
    """
    Load JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)