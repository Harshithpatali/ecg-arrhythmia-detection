"""
Dataset Builder

Creates the complete heartbeat dataset from
all MIT-BIH records.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from configs.config import (
    MITDB_DIR,
    PROCESSED_DATA_DIR,
    LABEL_ENCODER_PATH,
    CLASS_MAPPING_PATH,
)

from configs.logging_config import logger

from training.preprocessing.dataset_scanner import DatasetScanner
from training.preprocessing.wfdb_loader import WFDBLoader
from training.preprocessing.annotation_parser import AnnotationParser
from training.preprocessing.heartbeat_segmenter import HeartbeatSegmenter
from training.preprocessing.normalizer import HeartbeatNormalizer

from training.utils.serialization import (
    save_pickle,
    save_json,
)
from training.utils.file_utils import ensure_directory


class DatasetBuilder:

    def __init__(self):

        ensure_directory(PROCESSED_DATA_DIR)

        self.loader = WFDBLoader(MITDB_DIR)
        self.scanner = DatasetScanner(MITDB_DIR)
        self.parser = AnnotationParser()
        self.segmenter = HeartbeatSegmenter()

    def build(self):

        all_beats = []
        all_labels = []

        records = self.scanner.list_records()

        logger.info("Found %d records.", len(records))

        for record_name in records:

            logger.info("Processing %s", record_name)

            record = self.loader.load_record(record_name)

            annotations = self.loader.load_annotations(record_name)

            df = self.parser.parse(
                annotations["samples"],
                annotations["symbols"],
            )

            beats, labels = self.segmenter.segment(
                record["signal"],
                df,
            )

            beats = HeartbeatNormalizer.normalize(beats)

            all_beats.append(beats)
            all_labels.extend(labels)

        X = np.vstack(all_beats)

        y = np.asarray(all_labels)

        encoder = LabelEncoder()

        y_encoded = encoder.fit_transform(y)

        np.save(
            Path(PROCESSED_DATA_DIR) / "X.npy",
            X,
        )

        np.save(
            Path(PROCESSED_DATA_DIR) / "y.npy",
            y_encoded,
        )

        save_pickle(
            encoder,
            LABEL_ENCODER_PATH,
        )

        save_json(
            {
                "classes": encoder.classes_.tolist(),
            },
            CLASS_MAPPING_PATH,
        )

        info = {
            "samples": int(len(X)),
            "beat_length": int(X.shape[1]),
            "classes": encoder.classes_.tolist(),
            "distribution": dict(
                Counter(y)
            ),
        }

        save_json(
            info,
            Path(PROCESSED_DATA_DIR) / "dataset_info.json",
        )

        logger.info("Dataset Created")

        logger.info("Shape: %s", X.shape)

        logger.info("Labels: %s", Counter(y))

        return X, y_encoded