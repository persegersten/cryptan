"""OHLCV DataFrame cleaner for individual symbol datasets.

Responsibilities
----------------
- Validate that all required OHLCV columns are present.
- Ensure the ``timestamp`` column is UTC-aware.
- Sort rows ascending by timestamp.
- Remove duplicate timestamps, keeping the first occurrence.
- Drop rows with missing values in critical OHLCV columns.

The cleaner is intentionally conservative: it drops bad rows rather than
trying to repair them, and always logs what was removed so the caller can
decide whether the data quality is acceptable.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.ingestion.base import OHLCV_COLUMNS

logger = logging.getLogger(__name__)

# Numeric columns that must not be NaN for a row to be usable downstream.
_CRITICAL_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a normalized OHLCV DataFrame for a single symbol.

    The input is expected to conform to the schema defined in
    :data:`src.ingestion.base.OHLCV_COLUMNS`.  The function returns a new
    DataFrame; the original is never mutated.

    Steps applied in order:

    1. Validate required columns.
    2. Fail fast if the DataFrame is empty before cleaning.
    3. Ensure timestamps are UTC-aware.
    4. Sort ascending by timestamp.
    5. Remove duplicate timestamps (keep first occurrence).
    6. Drop rows with ``NaN`` in any critical OHLCV column.
    7. Fail fast if the DataFrame is empty after cleaning.

    Parameters
    ----------
    df:
        Normalized OHLCV DataFrame with at least the columns defined in
        :data:`~src.ingestion.base.OHLCV_COLUMNS`.

    Returns
    -------
    pandas.DataFrame
        Cleaned copy of *df*: UTC timestamps, ascending order, no duplicate
        timestamps, no rows with missing critical OHLCV values.

    Raises
    ------
    ValueError
        If required columns are missing, or if the DataFrame is empty before
        or after cleaning.
    """
    _validate_columns(df)

    if df.empty:
        raise ValueError(
            "Cannot clean an empty DataFrame. "
            "Ensure the ingestion step returned data."
        )

    df = df.copy()

    # Step 3: ensure UTC-aware timestamps
    df = _ensure_utc_timestamps(df)

    # Step 4: sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Step 5: remove duplicate timestamps
    duplicate_mask = df["timestamp"].duplicated(keep="first")
    n_dupes = int(duplicate_mask.sum())
    if n_dupes:
        symbol_label = str(df["symbol"].iloc[0]) if "symbol" in df.columns else "unknown"
        logger.warning(
            "Dropping %d duplicate timestamp(s) for symbol '%s' (keeping first occurrence).",
            n_dupes,
            symbol_label,
        )
        df = df[~duplicate_mask].reset_index(drop=True)

    # Step 6: drop rows with missing critical OHLCV values
    n_before = len(df)
    df = df.dropna(subset=list(_CRITICAL_COLUMNS)).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.warning(
            "Dropped %d row(s) with missing values in critical OHLCV columns.",
            n_dropped,
        )

    # Step 7: fail fast if nothing remains
    if df.empty:
        raise ValueError(
            "DataFrame is empty after cleaning. "
            "All rows were removed due to duplicates or missing OHLCV values."
        )

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any required OHLCV column is absent.

    Parameters
    ----------
    df:
        DataFrame to validate.

    Raises
    ------
    ValueError
        If one or more columns from :data:`~src.ingestion.base.OHLCV_COLUMNS`
        are missing.
    """
    missing = [col for col in OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {missing}. "
            f"Expected all of: {list(OHLCV_COLUMNS)}."
        )


def _ensure_utc_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with the ``timestamp`` column guaranteed to be UTC-aware.

    - Already UTC-aware: returned unchanged.
    - Timezone-naive: localized to UTC (with a warning).
    - Non-UTC timezone: converted to UTC.
    - Non-datetime dtype: parsed with ``utc=True``.

    Parameters
    ----------
    df:
        DataFrame whose ``timestamp`` column will be normalized.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with a UTC-aware ``timestamp`` column.
    """
    ts = df["timestamp"]

    if not pd.api.types.is_datetime64_any_dtype(ts):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(ts, utc=True)
        return df

    if ts.dt.tz is None:
        logger.warning(
            "Timestamp column is timezone-naive; localizing to UTC."
        )
        df = df.copy()
        df["timestamp"] = ts.dt.tz_localize("UTC")
    elif str(ts.dt.tz) != "UTC":
        df = df.copy()
        df["timestamp"] = ts.dt.tz_convert("UTC")

    return df
