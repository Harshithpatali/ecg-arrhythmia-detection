from training.ecg_processing.record_reader import (
    RecordReader,
)


class WaveformService:

    def get_signal(
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

        fs = int(record.fs)

        time_axis = [

            i / fs

            for i in range(
                len(signal)
            )

        ]

        return {

            "signal":
                signal.tolist(),

            "time_axis":
                time_axis,

            "sampling_rate":
                fs,

            "signal_length":
                len(signal),
        }