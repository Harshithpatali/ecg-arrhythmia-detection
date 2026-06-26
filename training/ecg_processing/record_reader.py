"""
Read ECG records.
"""

from pathlib import Path

import wfdb


class RecordReader:

    @staticmethod
    def find_record_name(
        directory: str
    ) -> str:

        hea_file = list(
            Path(directory).glob("*.hea")
        )[0]

        return hea_file.stem

    @staticmethod
    def load_record(
        directory: str
    ):

        record_name = (
            RecordReader.find_record_name(
                directory
            )
        )

        record_path = str(
            Path(directory) / record_name
        )

        record = wfdb.rdrecord(
            record_path
        )

        return record