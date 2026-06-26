"""
Dataset scanner.
"""

from pathlib import Path


class DatasetScanner:
    """
    Scan MIT-BIH dataset directory.
    """

    def __init__(self, dataset_path: str | Path):
        self.dataset_path = Path(dataset_path)

    def list_records(self) -> list[str]:
        """
        Return all available records.
        """

        records = []

        for file in self.dataset_path.glob("*.hea"):
            records.append(file.stem)

        records.sort()

        return records

    def count(self) -> int:
        """
        Number of ECG records.
        """

        return len(self.list_records())