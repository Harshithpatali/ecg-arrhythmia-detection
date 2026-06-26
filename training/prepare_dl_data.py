"""
Prepare Deep Learning Dataset
"""

from pathlib import Path

import numpy as np

from configs.config import PROCESSED_DATA_DIR

from training.preprocessing.balance_dataset import (
    DatasetBalancer,
)

from training.preprocessing.reshape_dataset import (
    DatasetReshaper,
)


def main():

    data_dir = Path(PROCESSED_DATA_DIR)

    X_train = np.load(data_dir / "X_train.npy")
    y_train = np.load(data_dir / "y_train.npy")

    X_val = np.load(data_dir / "X_val.npy")
    y_val = np.load(data_dir / "y_val.npy")

    X_test = np.load(data_dir / "X_test.npy")
    y_test = np.load(data_dir / "y_test.npy")

    balancer = DatasetBalancer()

    X_train, y_train = balancer.balance(
        X_train,
        y_train,
    )

    X_train = DatasetReshaper.reshape(X_train)

    X_val = DatasetReshaper.reshape(X_val)

    X_test = DatasetReshaper.reshape(X_test)

    np.save(data_dir / "X_train_dl.npy", X_train)
    np.save(data_dir / "X_val_dl.npy", X_val)
    np.save(data_dir / "X_test_dl.npy", X_test)

    np.save(data_dir / "y_train_dl.npy", y_train)
    np.save(data_dir / "y_val_dl.npy", y_val)
    np.save(data_dir / "y_test_dl.npy", y_test)

    print()

    print("Final Shapes")

    print("Train :", X_train.shape)

    print("Validation :", X_val.shape)

    print("Test :", X_test.shape)


if __name__ == "__main__":

    main()