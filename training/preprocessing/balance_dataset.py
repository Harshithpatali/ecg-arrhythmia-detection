"""
Balance training dataset using SMOTE.
"""

from __future__ import annotations

from collections import Counter

from imblearn.over_sampling import SMOTE


class DatasetBalancer:

    def __init__(self):

        self.smote = SMOTE(
            random_state=42,
        )

    def balance(
        self,
        X,
        y,
    ):

        print("Before SMOTE")

        print(Counter(y))

        X_balanced, y_balanced = self.smote.fit_resample(
            X,
            y,
        )

        print()

        print("After SMOTE")

        print(Counter(y_balanced))

        return X_balanced, y_balanced