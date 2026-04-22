"""Abstract base class for market-data providers.

All concrete providers must implement :class:`MarketDataSource` so that the
rest of the pipeline can work with a stable, provider-agnostic interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


# ---------------------------------------------------------------------------
# Required columns for any normalized OHLCV DataFrame returned by a provider.
# ---------------------------------------------------------------------------
OHLCV_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
)


class MarketDataSource(ABC):
    """Protocol for historical OHLCV data providers.

    Subclasses must implement :meth:`fetch_ohlcv` and return a DataFrame that
    conforms to the normalized schema defined by :data:`OHLCV_COLUMNS`.

    The normalized schema guarantees:

    * ``timestamp`` — timezone-aware UTC :class:`~pandas.Timestamp`.
    * ``open``, ``high``, ``low``, ``close``, ``volume`` — ``float64``.
    * ``symbol`` — ``str`` (uppercase ticker, e.g. ``"ETHUSDT"``).
    * Rows are sorted ascending by ``timestamp``.
    * No duplicate timestamps.
    """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars for *symbol* over [*start*, *end*).

        Parameters
        ----------
        symbol:
            Ticker symbol accepted by the underlying provider, e.g. ``"ETHUSDT"``.
        start:
            Inclusive start of the requested time range.  Timezone-aware or
            naive (treated as UTC if naive).
        end:
            Exclusive end of the requested time range.  Same timezone rules as
            *start*.
        timeframe:
            Bar interval string, e.g. ``"1h"``, ``"4h"``, ``"1d"``.

        Returns
        -------
        pandas.DataFrame
            Normalized OHLCV DataFrame with columns as defined in
            :data:`OHLCV_COLUMNS`.  May be empty if no data exists for the
            given range.
        """
