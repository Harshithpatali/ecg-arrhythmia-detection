from pathlib import Path

from training.ecg_processing.ecg_service import (
    ECGProcessingService
)

MIT_RECORD_DIR = (
    Path("data/mitdb")
)

service = ECGProcessingService()

result = service.process_directory(
    MIT_RECORD_DIR
)

print()

print(
    result["metadata"]
)

print()

print(
    result["beats"].shape
)