from typing import Dict
from typing import List

from pydantic import BaseModel


class RiskResponse(
    BaseModel
):

    risk_level: str

    message: str


class SummaryResponse(
    BaseModel
):

    distribution: Dict

    percentages: Dict

    risk: RiskResponse


class PredictionResponse(
    BaseModel
):

    metadata: Dict

    summary: SummaryResponse

    beat_predictions: List[str]

    confidences: List[float]