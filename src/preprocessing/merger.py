"""Multi-symbol OHLCV merger.

Merges cleaned per-symbol OHLCV DataFrames into a single wide DataFrame with
unambiguous, symbol-prefixed column names.

Design
------
* Input: a mapping of ``{symbol: cleaned_ohlcv_dataframe}``.
* Each symbol's numeric columns are renamed ``{SYMBOL}_open``, ``{SYMBOL}_close``, etc.
* The ``symbol`` string column is dropped before merging (it is encoded in the prefix).
* Merge is performed on ``timestamp``.
* Default join is ``"inner"``: only timestamps present in **all** symbols are kept.
  This guarantees no NaN gaps propagate into feature engineering.
* The result is sorted ascending by timestamp.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Numeric OHLCV columns carried into the merged DataFrame.
_NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def merge_symbol_frames(
    frames: dict[str, pd.DataFrame],
    join: str = "inner",
) -> pd.DataFrame:
    """Merge cleaned per-symbol OHLCV DataFrames into one wide DataFrame.

    Each symbol's numeric columns are prefixed with ``{SYMBOL}_`` (e.g.
    ``ETH_close``, ``BNB_volume``).  The ``symbol`` string column is excluded
    from the output since the information is already in the column names.

    The merge is performed iteratively on ``timestamp``.  The default inner
    join retains only rows where **every** symbol has a bar, which is the
    safest baseline for ML feature generation.

    Parameters
    ----------
    frames:
        Mapping from uppercase symbol string to its cleaned OHLCV DataFrame.
        Each DataFrame must contain ``timestamp`` and all five numeric OHLCV
        columns (``open``, ``high``, ``low``, ``close``, ``volume``).
    join:
        Merge strategy passed to :func:`pandas.DataFrame.merge`.  ``"inner"``
        (default) keeps only timestamps present in all symbols.  ``"outer"``
        preserves all timestamps and may introduce ``NaN``.

    Returns
    -------
    pandas.DataFrame
        Wide DataFrame with ``timestamp`` as the first column, followed by
        symbol-prefixed OHLCV columns, sorted ascending by timestamp.

    Raises
    ------
    ValueError
        If *frames* is empty, if any frame is missing required columns or is
        empty, or if the merged result contains no rows.
    """
    if not frames:
        raise ValueError("frames must not be empty.")

    _validate_frames(frames)

    # Build per-symbol sub-DataFrames with prefixed column names.
    prefixed: list[pd.DataFrame] = []
    for symbol, df in frames.items():
        sub = df[["timestamp", *_NUMERIC_COLUMNS]].copy()
        rename_map = {col: f"{symbol}_{col}" for col in _NUMERIC_COLUMNS}
        sub = sub.rename(columns=rename_map)
        prefixed.append(sub)

    # Iteratively merge on timestamp.
    merged = prefixed[0]
    for right in prefixed[1:]:
        merged = merged.merge(right, on="timestamp", how=join)

    merged = merged.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Merged %d symbol(s) into %d rows × %d columns (join=%r).",
        len(frames),
        len(merged),
        len(merged.columns),
        join,
    )

    if merged.empty:
        raise ValueError(
            f"Merged DataFrame is empty after join={join!r}. "
            f"No overlapping timestamps found across symbols: {list(frames.keys())}."
        )

    return merged


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_frames(frames: dict[str, pd.DataFrame]) -> None:
    """Raise ``ValueError`` if any frame is empty or missing required columns.

    Parameters
    ----------
    frames:
        Mapping to validate.

    Raises
    ------
    ValueError
        On the first frame that fails validation.
    """
    required = {"timestamp", *_NUMERIC_COLUMNS}
    for symbol, df in frames.items():
        if df.empty:
            raise ValueError(
                f"Frame for symbol '{symbol}' is empty. "
                f"Clean the data before merging."
            )
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Frame for symbol '{symbol}' is missing required columns: "
                f"{sorted(missing)}. Expected: {sorted(required)}."
            )
