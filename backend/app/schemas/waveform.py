from typing import List
from pydantic import BaseModel


class WaveformResponse(
    BaseModel
):
    signal: List[float]

    time_axis: List[float]

    sampling_rate: int

    signal_length: int


class RPeakResponse(
    BaseModel
):
    peaks: List[int]

    total_peaks: int