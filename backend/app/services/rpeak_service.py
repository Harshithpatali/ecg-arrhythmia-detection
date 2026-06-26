from training.ecg_processing.record_reader import (
    RecordReader,
)

from training.preprocessing.rpeak_detector import (
    RPeakDetector,
)


class RPeakService:

    def __init__(self):

        self.detector = (
            RPeakDetector()
        )

    def detect(
        self,
        directory,
    ):

        record = (
            RecordReader.load_record(
                directory
            )
        )

        signal = (
            record.p_signal[:, 0]
        )

        peaks = (
            self.detector.detect(
                signal
            )
        )

        return {

            "peaks":
                peaks.tolist(),

            "total_peaks":
                len(peaks),
        }