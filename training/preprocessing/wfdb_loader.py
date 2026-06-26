"""
WFDB ECG loader.

Loads ECG records from the MIT-BIH Arrhythmia Database.
"""

from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from configs.logging_config import logger


class WFDBLoader:
    """
    Load MIT-BIH ECG records.
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    def _record_path(self, record_name: str) -> str:
        """
        Return the record path without extension.

        Example:
            data/mitdb/100
        """
        return str(self.dataset_path / record_name)

    def load_record(
        self,
        record_name: str,
    ) -> dict[str, Any]:
        """
        Load ECG signal and metadata.

        Returns
        -------
        dict
        """

        record_path = self._record_path(record_name)

        logger.info("Loading ECG record %s", record_name)

        try:

            record = wfdb.rdrecord(record_path)

            signal = record.p_signal

            metadata = {
                "record_name": record_name,
                "sampling_frequency": record.fs,
                "signal_length": len(signal),
                "channels": record.sig_name,
                "units": record.units,
            }

            return {
                "signal": signal,
                "metadata": metadata,
            }

        except Exception as exc:

            logger.exception(exc)

            raise RuntimeError(
                f"Unable to load record {record_name}"
            ) from exc

    def load_annotations(
        self,
        record_name: str,
        extension: str = "atr",
    ) -> dict[str, np.ndarray]:
        """
        Load annotation file.

        Returns
        -------
        dict
        """

        record_path = self._record_path(record_name)

        logger.info(
            "Loading annotations for %s",
            record_name,
        )

        try:

            annotation = wfdb.rdann(
                record_path,
                extension,
            )

            return {
                "samples": annotation.sample,
                "symbols": np.array(annotation.symbol),
            }

        except Exception:

            logger.warning(
                "Annotation file not found."
            )

            return {
                "samples": np.array([]),
                "symbols": np.array([]),
            }