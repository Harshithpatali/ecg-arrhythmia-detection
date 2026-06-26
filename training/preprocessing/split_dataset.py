"""
Dataset Splitter
"""

from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from configs.config import PROCESSED_DATA_DIR

from configs.constants import (
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def split_dataset():

    X = np.load(
        Path(PROCESSED_DATA_DIR) / "X.npy"
    )

    y = np.load(
        Path(PROCESSED_DATA_DIR) / "y.npy"
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    validation_ratio = (
        VALIDATION_SIZE /
        (TEST_SIZE + VALIDATION_SIZE)
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    np.save(
        Path(PROCESSED_DATA_DIR) / "X_train.npy",
        X_train,
    )

    np.save(
        Path(PROCESSED_DATA_DIR) / "X_val.npy",
        X_val,
    )

    np.save(
        Path(PROCESSED_DATA_DIR) / "X_test.npy",
        X_test,
    )

    np.save(
        Path(PROCESSED_DATA_DIR) / "y_train.npy",
        y_train,
    )

    np.save(
        Path(PROCESSED_DATA_DIR) / "y_val.npy",
        y_val,
    )

    np.save(
        Path(PROCESSED_DATA_DIR) / "y_test.npy",
        y_test,
    )

    print("Dataset Split Complete")

    print()

    print("Train :", X_train.shape)

    print("Validation :", X_val.shape)

    print("Test :", X_test.shape)