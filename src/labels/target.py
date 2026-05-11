"""Target-label generation for the configured trading symbol.

The label at timestamp ``t`` is based on the future close return from ``t`` to
``t + prediction_horizon_bars``:

* ``1`` when future return is greater than ``return_threshold``.
* ``-1`` when future return is less than negative ``return_threshold``.
* ``0`` otherwise.

Rows without a full future horizon are dropped so downstream splitting and
training never receive labels with missing future data.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config.model import TrainingConfig

logger = logging.getLogger(__name__)

TARGET_RETURN_COLUMN = "target_future_return"
TARGET_LABEL_COLUMN = "target_label"


def add_target_labels(
    feature_df: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Add multiclass target labels for ``config.trading_symbol``.

    Parameters
    ----------
    feature_df:
        Feature DataFrame containing ``timestamp`` and
        ``{trading_symbol}_close``. The input is not mutated.
    config:
        Validated training config supplying the trading symbol, prediction
        horizon, and return threshold.

    Returns
    -------
    pandas.DataFrame
        Chronologically sorted DataFrame with ``target_future_return`` and
        ``target_label`` appended. The final ``prediction_horizon_bars`` rows
        are dropped because their future return is unknown.

    Raises
    ------
    ValueError
        If required columns are missing, if the input is empty, or if no rows
        remain after dropping rows without a full future horizon.
    """
    if feature_df.empty:
        raise ValueError(
            "feature_df is empty. Ensure feature engineering produced data before labeling."
        )

    if "timestamp" not in feature_df.columns:
        raise ValueError("feature_df is missing the required 'timestamp' column.")

    trading_symbol = config.trading_symbol
    close_col = f"{trading_symbol}_close"
    if close_col not in feature_df.columns:
        raise ValueError(
            f"Required close column '{close_col}' not found in feature_df. "
            f"The trading symbol must be present in the feature data before labels can be created."
        )

    horizon = config.prediction_horizon_bars
    threshold = config.return_threshold

    df = feature_df.sort_values("timestamp").reset_index(drop=True).copy()
    close = df[close_col]
    future_close = close.shift(-horizon)
    df[TARGET_RETURN_COLUMN] = (future_close - close) / close

    df[TARGET_LABEL_COLUMN] = 0
    df.loc[df[TARGET_RETURN_COLUMN] > threshold, TARGET_LABEL_COLUMN] = 1
    df.loc[df[TARGET_RETURN_COLUMN] < -threshold, TARGET_LABEL_COLUMN] = -1
    df[TARGET_LABEL_COLUMN] = df[TARGET_LABEL_COLUMN].astype("int64")

    rows_before = len(df)
    df = df.dropna(subset=[TARGET_RETURN_COLUMN]).reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    if rows_dropped:
        logger.info(
            "Dropped %d row(s) without full %d-bar future horizon after target labeling.",
            rows_dropped,
            horizon,
        )

    if df.empty:
        raise ValueError(
            "Labelled DataFrame is empty after dropping rows without a full future horizon. "
            "The input series may be too short for prediction_horizon_bars."
        )

    logger.info(
        "Target labeling complete for %s: %d rows, horizon=%d, threshold=%s, "
        "class_counts=%s",
        trading_symbol,
        len(df),
        horizon,
        threshold,
        df[TARGET_LABEL_COLUMN].value_counts().sort_index().to_dict(),
    )

    return df

