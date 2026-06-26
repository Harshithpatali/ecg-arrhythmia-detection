from training.ecg_processing.ecg_service import (
    ECGProcessingService
)


class HeartbeatService:

    def __init__(self):

        self.service = (
            ECGProcessingService()
        )

    def get_beats(
        self,
        directory,
    ):

        result = (
            self.service
            .process_directory(
                directory
            )
        )

        return {

            "beats":
                result["beats"]
                .tolist(),

            "total_beats":
                len(
                    result["beats"]
                ),
        }