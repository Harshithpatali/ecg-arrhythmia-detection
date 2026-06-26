"""
Annotation parser.

Converts WFDB annotations into
AAMI heartbeat labels.
"""

from __future__ import annotations

import pandas as pd

from configs.logging_config import logger
from .aami_mapping import map_symbol


class AnnotationParser:
    """
    Parse MIT-BIH annotations.
    """

    def parse(
        self,
        samples,
        symbols,
    ) -> pd.DataFrame:
        """
        Convert annotations into a DataFrame.

        Parameters
        ----------
        samples : array-like

        symbols : array-like

        Returns
        -------
        pandas.DataFrame
        """

        rows = []

        skipped = 0

        for sample, symbol in zip(samples, symbols):

            label = map_symbol(symbol)

            if label is None:
                skipped += 1
                continue

            rows.append(
                {
                    "sample": int(sample),
                    "symbol": symbol,
                    "label": label,
                }
            )

        logger.info(
            "Parsed %d heartbeat annotations (%d skipped).",
            len(rows),
            skipped,
        )

        return pd.DataFrame(rows)