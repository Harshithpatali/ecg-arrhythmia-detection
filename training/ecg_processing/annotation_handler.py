"""
Annotation management.
"""

from pathlib import Path

import pandas as pd
import wfdb

from training.preprocessing.rpeak_detector import (
    RPeakDetector,
)

from training.preprocessing.annotation_parser import (
    AnnotationParser,
)


class AnnotationHandler:

    def __init__(self):

        self.parser = (
            AnnotationParser()
        )

        self.detector = (
            RPeakDetector()
        )

    def get_annotations(
        self,
        directory: str,
        signal,
    ) -> pd.DataFrame:

        atr_files = list(
            Path(directory).glob("*.atr")
        )

        if len(atr_files) > 0:

            record_name = (
                atr_files[0].stem
            )

            annotation = wfdb.rdann(
                str(
                    Path(directory)
                    / record_name
                ),
                "atr",
            )

            return self.parser.parse(

                annotation.sample,

                annotation.symbol,

            )

        peaks = self.detector.detect(
            signal
        )

        return pd.DataFrame(
            {
                "sample": peaks,
                "symbol": ["N"] * len(peaks),
                "label": ["N"] * len(peaks),
            }
        )