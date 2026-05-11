"""Pure, testable feature transformation functions for individual symbol data.

All transformations operate on a wide merged DataFrame where each symbol's
columns carry a ``{SYMBOL}_`` prefix (e.g. ``ETH_close``, ``BNB_volume``).

Design rules
------------
* Every function at row ``t`` uses **only** data from rows ≤ ``t``.
* No future information is introduced — rolling windows are anchored on the
  current row and look *backwards*.
* Functions return a **new** DataFrame; the input is never mutated.
* NaN rows caused by insufficient lookback are **not** dropped here; that
  responsibility belongs to :func:`src.features.builder.build_features`.

These functions are designed to be called in sequence by the builder, but they
are also directly importable for unit tests and ad-hoc analysis.
"""

from __future__ import annotations

import pandas as pd


def add_return_features(
    df: pd.DataFrame,
    symbol: str,
    return_windows: list[int],
) -> pd.DataFrame:
    """Add N-bar close-price return features for *symbol*.

    For each window ``n`` in *return_windows*, a column
    ``{symbol}_return_{n}`` is added with values::

        (close[t] - close[t-n]) / close[t-n]

    This is the ``pandas.Series.pct_change(n)`` definition.  The feature at
    row ``t`` uses close prices from row ``t-n`` to row ``t`` — no future
    information.  The first ``n`` rows will be ``NaN`` for the corresponding
    window.

    Parameters
    ----------
    df:
        Wide merged DataFrame containing a ``{symbol}_close`` column.
    symbol:
        Uppercase ticker string, e.g. ``"ETH"``.
    return_windows:
        List of lookback bar counts, e.g. ``[1, 5, 20]``.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with new ``{symbol}_return_{n}`` columns appended.

    Raises
    ------
    ValueError
        If the required close column is absent.
    """
    close_col = f"{symbol}_close"
    _require_column(df, close_col)

    df = df.copy()
    for n in return_windows:
        df[f"{symbol}_return_{n}"] = df[close_col].pct_change(periods=n)
    return df


def add_moving_average_features(
    df: pd.DataFrame,
    symbol: str,
    short_window: int,
    long_window: int,
) -> pd.DataFrame:
    """Add short MA, long MA, and their ratio for *symbol*.

    Added columns:

    * ``{symbol}_ma_short`` — rolling mean of close over *short_window* bars.
    * ``{symbol}_ma_long`` — rolling mean of close over *long_window* bars.
    * ``{symbol}_ma_ratio`` — ``ma_short / ma_long``.

    All rolling windows use ``min_periods`` equal to the window size, so the
    first ``window - 1`` rows are ``NaN``.

    Parameters
    ----------
    df:
        Wide merged DataFrame containing a ``{symbol}_close`` column.
    symbol:
        Uppercase ticker string.
    short_window:
        Number of bars for the short moving average.
    long_window:
        Number of bars for the long moving average.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with the three new MA columns appended.

    Raises
    ------
    ValueError
        If the required close column is absent, or if *short_window* ≥
        *long_window*.
    """
    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be less than long_window ({long_window})."
        )
    close_col = f"{symbol}_close"
    _require_column(df, close_col)

    df = df.copy()
    close = df[close_col]
    short_ma = close.rolling(window=short_window, min_periods=short_window).mean()
    long_ma = close.rolling(window=long_window, min_periods=long_window).mean()

    df[f"{symbol}_ma_short"] = short_ma
    df[f"{symbol}_ma_long"] = long_ma
    df[f"{symbol}_ma_ratio"] = short_ma / long_ma
    return df


def add_volatility_features(
    df: pd.DataFrame,
    symbol: str,
    volatility_window: int,
) -> pd.DataFrame:
    """Add rolling 1-bar return volatility for *symbol*.

    The volatility is the rolling standard deviation of 1-bar close returns
    over the last *volatility_window* return observations.  The 1-bar return
    at row ``t`` is ``(close[t] - close[t-1]) / close[t-1]``.  NaN rows
    arise for the first ``volatility_window`` rows.

    Added column: ``{symbol}_volatility``.

    Parameters
    ----------
    df:
        Wide merged DataFrame containing a ``{symbol}_close`` column.
    symbol:
        Uppercase ticker string.
    volatility_window:
        Rolling window size (bar count) for the standard deviation.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with the new volatility column appended.

    Raises
    ------
    ValueError
        If the required close column is absent.
    """
    close_col = f"{symbol}_close"
    _require_column(df, close_col)

    df = df.copy()
    returns_1bar = df[close_col].pct_change(periods=1)
    df[f"{symbol}_volatility"] = returns_1bar.rolling(
        window=volatility_window, min_periods=volatility_window
    ).std()
    return df


def add_volume_features(
    df: pd.DataFrame,
    symbol: str,
    volume_window: int,
) -> pd.DataFrame:
    """Add rolling volume mean and z-score for *symbol*.

    Added columns:

    * ``{symbol}_volume_mean`` — rolling mean of volume over *volume_window*.
    * ``{symbol}_volume_zscore`` — ``(volume - rolling_mean) / rolling_std``.

    When the rolling standard deviation is zero (constant volume), the
    z-score is set to ``0.0`` to avoid division-by-zero ``NaN`` or
    ``Inf`` values.

    Parameters
    ----------
    df:
        Wide merged DataFrame containing a ``{symbol}_volume`` column.
    symbol:
        Uppercase ticker string.
    volume_window:
        Rolling window size (bar count).

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with the two new volume columns appended.

    Raises
    ------
    ValueError
        If the required volume column is absent.
    """
    volume_col = f"{symbol}_volume"
    _require_column(df, volume_col)

    df = df.copy()
    volume = df[volume_col]
    rolling = volume.rolling(window=volume_window, min_periods=volume_window)
    vol_mean = rolling.mean()
    vol_std = rolling.std()

    zscore = (volume - vol_mean) / vol_std
    # Replace inf/NaN caused by zero std with 0.0
    zscore = zscore.where(vol_std.notna() & (vol_std != 0.0), other=0.0)

    df[f"{symbol}_volume_mean"] = vol_mean
    df[f"{symbol}_volume_zscore"] = zscore
    return df


def add_cross_asset_features(
    df: pd.DataFrame,
    target_symbol: str,
    other_symbol: str,
    correlation_window: int,
) -> pd.DataFrame:
    """Add cross-asset features between *target_symbol* and *other_symbol*.

    Added columns:

    * ``{target}_{other}_rel_strength`` — ratio of target close to other
      close: ``target_close / other_close``.  A value > 1 means the target
      is priced higher in absolute terms; trends in this ratio capture
      relative momentum.
    * ``{target}_{other}_corr`` — rolling Pearson correlation between the
      1-bar returns of *target_symbol* and *other_symbol* over
      *correlation_window* bars.

    Both columns at row ``t`` use only data up to row ``t``.

    Parameters
    ----------
    df:
        Wide merged DataFrame containing ``{target}_close`` and
        ``{other}_close`` columns.
    target_symbol:
        Uppercase ticker for the trading (target) asset.
    other_symbol:
        Uppercase ticker for the signal asset to compare against.
    correlation_window:
        Rolling window size (bar count) for the correlation.

    Returns
    -------
    pandas.DataFrame
        Copy of *df* with the two new cross-asset columns appended.

    Raises
    ------
    ValueError
        If either required close column is absent.
    """
    target_close_col = f"{target_symbol}_close"
    other_close_col = f"{other_symbol}_close"
    _require_column(df, target_close_col)
    _require_column(df, other_close_col)

    df = df.copy()
    target_close = df[target_close_col]
    other_close = df[other_close_col]

    df[f"{target_symbol}_{other_symbol}_rel_strength"] = target_close / other_close

    target_returns = target_close.pct_change(periods=1)
    other_returns = other_close.pct_change(periods=1)
    df[f"{target_symbol}_{other_symbol}_corr"] = target_returns.rolling(
        window=correlation_window, min_periods=correlation_window
    ).corr(other_returns)

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_column(df: pd.DataFrame, column: str) -> None:
    """Raise ``ValueError`` if *column* is absent from *df*.

    Parameters
    ----------
    df:
        DataFrame to check.
    column:
        Column name that must be present.

    Raises
    ------
    ValueError
        If the column is missing.
    """
    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' not found in DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}."
        )
