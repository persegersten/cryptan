"""Feature builder: orchestrate feature construction across all signal symbols.

This module provides :func:`build_features`, the single entry point for Step 4
of the training pipeline.  It iterates over all configured signal symbols,
applies the per-symbol transformations from :mod:`src.features.technical`,
optionally adds cross-asset features, and then drops rows whose lookback is
insufficient (i.e. rows that still carry ``NaN`` in any new feature column).

Usage
-----
::

    from src.features.builder import build_features

    feature_df = build_features(merged_df, config)
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config.model import TrainingConfig
from src.features.technical import (
    add_cross_asset_features,
    add_moving_average_features,
    add_return_features,
    add_volatility_features,
    add_volume_features,
)

logger = logging.getLogger(__name__)


def build_features(
    merged_df: pd.DataFrame,
    config: TrainingConfig,
) -> pd.DataFrame:
    """Build all features for every configured signal symbol.

    Steps performed in order:

    1. Validate that *merged_df* is non-empty and sorted ascending by
       ``timestamp``.
    2. For each symbol in ``config.signal_symbols``:

       * N-bar close returns (windows from ``config.feature_config.return_windows``).
       * Short and long moving averages and their ratio.
       * Rolling close-return volatility.
       * Rolling volume mean and z-score.

    3. Cross-asset features between ``config.trading_symbol`` and every
       *other* signal symbol (i.e. all signal symbols except the trading
       symbol itself):

       * Relative strength (price ratio).
       * Rolling return correlation.

       Cross-asset features are **only** computed when the trading symbol
       appears in ``config.signal_symbols`` (its close column must be present
       in the merged DataFrame).

    4. Drop rows where any new feature column is ``NaN`` — these are rows
       near the beginning of the series with insufficient lookback history.
       The count of dropped rows is logged.

    Parameters
    ----------
    merged_df:
        Wide DataFrame produced by
        :func:`src.preprocessing.merger.merge_symbol_frames`.  Must contain a
        ``timestamp`` column and symbol-prefixed OHLCV columns.
    config:
        Validated training configuration supplying ``signal_symbols``,
        ``trading_symbol``, and ``feature_config`` windows.

    Returns
    -------
    pandas.DataFrame
        DataFrame with all original columns **plus** new feature columns,
        rows with insufficient lookback dropped, sorted ascending by
        ``timestamp``.

    Raises
    ------
    ValueError
        If *merged_df* is empty, if ``timestamp`` is missing, or if any
        required OHLCV column for a configured symbol is absent.
    """
    if merged_df.empty:
        raise ValueError(
            "merged_df is empty. Ensure the preprocessing step produced data."
        )

    if "timestamp" not in merged_df.columns:
        raise ValueError(
            "merged_df is missing the required 'timestamp' column."
        )

    # Ensure chronological order before applying rolling windows.
    df = merged_df.sort_values("timestamp").reset_index(drop=True)

    # Remember which columns existed before feature generation so we can
    # identify new feature columns afterwards.
    original_columns = set(df.columns)

    feature_cfg = config.feature_config
    signal_symbols = config.signal_symbols
    trading_symbol = config.trading_symbol

    logger.info(
        "Building features for %d signal symbol(s): %s",
        len(signal_symbols),
        ", ".join(signal_symbols),
    )

    # ------------------------------------------------------------------
    # Per-symbol features
    # ------------------------------------------------------------------
    for symbol in signal_symbols:
        logger.debug("Adding return features for %s ...", symbol)
        df = add_return_features(df, symbol, feature_cfg.return_windows)

        logger.debug("Adding moving average features for %s ...", symbol)
        df = add_moving_average_features(
            df, symbol, feature_cfg.ma_short_window, feature_cfg.ma_long_window
        )

        logger.debug("Adding volatility features for %s ...", symbol)
        df = add_volatility_features(df, symbol, feature_cfg.volatility_window)

        logger.debug("Adding volume features for %s ...", symbol)
        df = add_volume_features(df, symbol, feature_cfg.volume_window)

    # ------------------------------------------------------------------
    # Cross-asset features (trading symbol vs each other signal symbol)
    # ------------------------------------------------------------------
    trading_close_col = f"{trading_symbol}_close"
    if trading_close_col in df.columns:
        other_symbols = [s for s in signal_symbols if s != trading_symbol]
        for other_symbol in other_symbols:
            other_close_col = f"{other_symbol}_close"
            if other_close_col not in df.columns:
                logger.warning(
                    "Skipping cross-asset features for %s vs %s: column '%s' not found.",
                    trading_symbol,
                    other_symbol,
                    other_close_col,
                )
                continue
            logger.debug(
                "Adding cross-asset features for %s vs %s ...",
                trading_symbol,
                other_symbol,
            )
            df = add_cross_asset_features(
                df, trading_symbol, other_symbol, feature_cfg.correlation_window
            )
    else:
        logger.info(
            "Trading symbol '%s' has no close column in merged_df; "
            "skipping cross-asset features.",
            trading_symbol,
        )

    # ------------------------------------------------------------------
    # Drop rows with insufficient lookback (NaN in any new feature column)
    # ------------------------------------------------------------------
    new_feature_columns = [c for c in df.columns if c not in original_columns]
    rows_before = len(df)
    df = df.dropna(subset=new_feature_columns).reset_index(drop=True)
    rows_dropped = rows_before - len(df)

    if rows_dropped:
        logger.info(
            "Dropped %d row(s) with insufficient lookback after feature engineering "
            "(%d row(s) remaining).",
            rows_dropped,
            len(df),
        )

    if df.empty:
        raise ValueError(
            "Feature DataFrame is empty after dropping lookback rows. "
            "The input series may be too short for the configured feature windows."
        )

    logger.info(
        "Feature engineering complete: %d rows × %d columns "
        "(%d new feature columns).",
        len(df),
        len(df.columns),
        len(new_feature_columns),
    )

    return df
