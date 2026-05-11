"""Chronological train/validation/test split for labelled time-series data.

The splitter keeps all rows in timestamp order and creates contiguous
train/validation/test partitions from the configured split fractions.  It does
not shuffle rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd

from src.config.model import TrainingConfig
from src.labels.target import TARGET_LABEL_COLUMN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChronologicalSplit:
    """Contiguous train/validation/test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    @property
    def row_counts(self) -> dict[str, int]:
        """Return partition row counts for logging and metadata."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }


def split_chronologically(
    labelled_df: pd.DataFrame,
    config: TrainingConfig,
    *,
    timestamp_column: str = "timestamp",
    target_column: str = TARGET_LABEL_COLUMN,
) -> ChronologicalSplit:
    """Split labelled rows into chronological train/validation/test frames.

    Parameters
    ----------
    labelled_df:
        DataFrame produced after feature engineering and target-label
        generation. The input is not mutated.
    config:
        Validated training config supplying split fractions.
    timestamp_column:
        Name of the timestamp column used for chronological ordering.
    target_column:
        Name of the supervised target column expected by later model steps.

    Returns
    -------
    ChronologicalSplit
        Three contiguous DataFrames with reset indexes.

    Raises
    ------
    ValueError
        If the input is empty, required columns are missing, timestamps contain
        nulls or duplicates, or the configured fractions would create an empty
        train/validation/test partition.
    """
    if labelled_df.empty:
        raise ValueError(
            "labelled_df is empty. Ensure labels were created before splitting."
        )

    missing = {timestamp_column, target_column} - set(labelled_df.columns)
    if missing:
        raise ValueError(
            f"labelled_df is missing required columns for splitting: {sorted(missing)}."
        )

    if labelled_df[timestamp_column].isna().any():
        raise ValueError(f"Column '{timestamp_column}' contains null timestamps.")

    df = labelled_df.sort_values(timestamp_column).reset_index(drop=True).copy()

    if df[timestamp_column].duplicated().any():
        raise ValueError(
            f"Column '{timestamp_column}' contains duplicate timestamps. "
            "Resolve duplicate bars before splitting."
        )

    row_count = len(df)
    train_end = int(row_count * config.split.train)
    validation_end = int(row_count * (config.split.train + config.split.validation))

    train = df.iloc[:train_end].reset_index(drop=True)
    validation = df.iloc[train_end:validation_end].reset_index(drop=True)
    test = df.iloc[validation_end:].reset_index(drop=True)

    split = ChronologicalSplit(train=train, validation=validation, test=test)
    empty_partitions = [
        name for name, count in split.row_counts.items() if count == 0
    ]
    if empty_partitions:
        raise ValueError(
            "Configured split fractions create empty partition(s) "
            f"{empty_partitions} for {row_count} row(s). "
            "Use more labelled history or adjust split fractions."
        )

    logger.info(
        "Chronological split complete: train=%d, validation=%d, test=%d rows.",
        len(train),
        len(validation),
        len(test),
    )

    return split

