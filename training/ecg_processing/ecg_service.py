"""
ECG Processing Service.
"""

from pathlib import Path

from training.ecg_processing.file_validator import (
    ECGFileValidator,
)

from training.ecg_processing.record_reader import (
    RecordReader,
)

from training.ecg_processing.annotation_handler import (
    AnnotationHandler,
)

from training.ecg_processing.signal_processor import (
    SignalProcessor,
)


class ECGProcessingService:

    def __init__(self):

        self.annotation_handler = (
            AnnotationHandler()
        )

        self.signal_processor = (
            SignalProcessor()
        )

    def process_directory(
        self,
        directory: str,
    ):

        ECGFileValidator.validate(
            directory
        )

        record = (
            RecordReader.load_record(
                directory
            )
        )

        signal = record.p_signal

        annotations = (
            self.annotation_handler
            .get_annotations(
                directory,
                signal,
            )
        )

        beats, labels = (
            self.signal_processor.process(
                signal,
                annotations,
            )
        )

        metadata = {

            "record_name":
                RecordReader.find_record_name(
                    directory
                ),

            "sampling_rate":
                record.fs,

            "signal_length":
                len(signal),

            "channels":
                record.n_sig,

            "total_beats":
                len(beats),
        }

        return {

            "metadata":
                metadata,

            "beats":
                beats,

            "labels":
                labels,
        }