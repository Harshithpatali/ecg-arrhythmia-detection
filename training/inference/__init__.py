"""
Inference package.
"""

from .beat_predictor import BeatPredictor
from .full_record_predictor import FullRecordPredictor

__all__ = [
    "BeatPredictor",
    "FullRecordPredictor",
]