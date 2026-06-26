from pathlib import Path

from training.prediction.prediction_service import (
    PredictionService
)

service = PredictionService()

result = service.analyze_record(
    Path("data/mitdb")
)

print()

print(
    result["summary"]
)

print()

print(
    result["metadata"]
)