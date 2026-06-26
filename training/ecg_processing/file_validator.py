"""
ECG file validation.
"""

from pathlib import Path


class ECGFileValidator:

    REQUIRED = {
        ".dat",
        ".hea",
    }

    OPTIONAL = {
        ".atr",
    }

    @staticmethod
    def validate(directory: str):

        files = {
            p.suffix
            for p in Path(directory).glob("*")
        }

        missing = (
            ECGFileValidator.REQUIRED - files
        )

        if missing:

            raise ValueError(
                f"Missing ECG files: {missing}"
            )

        return True