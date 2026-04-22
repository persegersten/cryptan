"""Binance historical OHLCV ingestion adapter.

Fetches candlestick (klines) data from the Binance REST API and normalises
the response into the project's standard OHLCV schema.

Binance klines endpoint
-----------------------
``GET /api/v3/klines``

Each element of the response array maps to the following indices::

    [0]  Open time         (ms epoch)
    [1]  Open              (str float)
    [2]  High              (str float)
    [3]  Low               (str float)
    [4]  Close             (str float)
    [5]  Volume            (str float)
    [6]  Close time        (ms epoch)  — not used
    [7]  Quote asset vol   (str float) — not used
    ...  (further fields not used)

Pagination
----------
Binance caps each request at 1 000 bars.  This adapter loops over the full
requested date range, advancing the ``startTime`` cursor after each page, and
concatenates all pages before returning.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests

from src.ingestion.base import OHLCV_COLUMNS, MarketDataSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Binance klines index positions in each row
# ---------------------------------------------------------------------------
_IDX_OPEN_TIME = 0
_IDX_OPEN = 1
_IDX_HIGH = 2
_IDX_LOW = 3
_IDX_CLOSE = 4
_IDX_VOLUME = 5

_KLINES_ENDPOINT = "/api/v3/klines"


def _to_utc_ms(dt: datetime) -> int:
    """Convert a datetime to UTC milliseconds since epoch."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000)


def _parse_klines(raw_rows: list[list[Any]], symbol: str) -> pd.DataFrame:
    """Convert raw Binance klines rows to a normalised OHLCV DataFrame.

    Parameters
    ----------
    raw_rows:
        List of kline rows as returned by the Binance API.
    symbol:
        Ticker string to stamp on the ``symbol`` column.

    Returns
    -------
    pandas.DataFrame
        Normalised OHLCV DataFrame.  Empty if *raw_rows* is empty.
    """
    if not raw_rows:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))

    timestamps = pd.to_datetime(
        [row[_IDX_OPEN_TIME] for row in raw_rows],
        unit="ms",
        utc=True,
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": pd.array([float(row[_IDX_OPEN]) for row in raw_rows], dtype="float64"),
            "high": pd.array([float(row[_IDX_HIGH]) for row in raw_rows], dtype="float64"),
            "low": pd.array([float(row[_IDX_LOW]) for row in raw_rows], dtype="float64"),
            "close": pd.array([float(row[_IDX_CLOSE]) for row in raw_rows], dtype="float64"),
            "volume": pd.array([float(row[_IDX_VOLUME]) for row in raw_rows], dtype="float64"),
            "symbol": symbol,
        }
    )
    return df


class BinanceMarketDataSource(MarketDataSource):
    """Binance REST API adapter for historical OHLCV data.

    Parameters
    ----------
    api_key:
        Binance API key.  Used in request headers to unlock higher rate limits.
        Klines are a public endpoint and do not require authentication, but
        including the key is best practice.
    api_secret:
        Binance API secret.  Stored but not used for unsigned public endpoints.
    base_url:
        Binance REST base URL.  Defaults to ``https://api.binance.com``.
        Override to ``https://testnet.binance.vision`` for testing.
    bars_per_request:
        Maximum bars to request per API call.  Binance hard-caps at 1 000.
    request_delay_seconds:
        Seconds to sleep between paginated requests to respect rate limits.
    """

    _BINANCE_TIMEFRAME_MAP: dict[str, str] = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d",
        "3d": "3d",
        "1w": "1w",
        "1M": "1M",
    }

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.binance.com",
        bars_per_request: int = 1000,
        request_delay_seconds: float = 0.2,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._base_url = base_url.rstrip("/")
        self._bars_per_request = min(bars_per_request, 1000)
        self._request_delay_seconds = request_delay_seconds

        self._session = requests.Session()
        self._session.headers.update({"X-MBX-APIKEY": self._api_key})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV bars from Binance for *symbol* over [*start*, *end*).

        Paginates automatically when the requested range exceeds
        *bars_per_request* bars.

        Parameters
        ----------
        symbol:
            Binance trading pair, e.g. ``"ETHUSDT"``.
        start:
            Inclusive start datetime (UTC or timezone-aware).
        end:
            Exclusive end datetime (UTC or timezone-aware).
        timeframe:
            Bar interval string recognised by Binance, e.g. ``"1h"``.

        Returns
        -------
        pandas.DataFrame
            Normalised OHLCV DataFrame with columns ``timestamp``, ``open``,
            ``high``, ``low``, ``close``, ``volume``, ``symbol``.

        Raises
        ------
        ValueError
            If *timeframe* is not supported.
        requests.HTTPError
            If the Binance API returns an error response.
        """
        interval = self._resolve_interval(timeframe)
        start_ms = _to_utc_ms(start)
        end_ms = _to_utc_ms(end)

        logger.info(
            "Fetching %s %s bars from Binance: %s → %s",
            symbol,
            timeframe,
            start,
            end,
        )

        pages: list[pd.DataFrame] = []
        cursor_ms = start_ms

        while cursor_ms < end_ms:
            page = self._fetch_page(
                symbol=symbol,
                interval=interval,
                start_ms=cursor_ms,
                end_ms=end_ms,
            )
            if page.empty:
                break

            pages.append(page)

            # Advance cursor to the millisecond after the last returned bar.
            last_ts_ms = int(page["timestamp"].iloc[-1].timestamp() * 1_000)
            next_cursor = last_ts_ms + 1
            if next_cursor <= cursor_ms:
                # Safeguard against an infinite loop if the API misbehaves.
                break
            cursor_ms = next_cursor

            if cursor_ms < end_ms:
                time.sleep(self._request_delay_seconds)

        if not pages:
            logger.warning("No data returned for %s %s [%s, %s)", symbol, timeframe, start, end)
            return pd.DataFrame(columns=list(OHLCV_COLUMNS))

        result = pd.concat(pages, ignore_index=True)
        result = self._clean(result, end_ms)

        logger.info(
            "Fetched %d bars for %s %s",
            len(result),
            symbol,
            timeframe,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_interval(self, timeframe: str) -> str:
        """Map a generic timeframe string to a Binance interval string.

        Raises
        ------
        ValueError
            If *timeframe* is not in the supported map.
        """
        interval = self._BINANCE_TIMEFRAME_MAP.get(timeframe)
        if interval is None:
            supported = sorted(self._BINANCE_TIMEFRAME_MAP.keys())
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}. "
                f"Supported values: {supported}"
            )
        return interval

    def _fetch_page(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch a single page of klines from Binance.

        Parameters
        ----------
        symbol:
            Binance trading pair.
        interval:
            Binance interval string (e.g. ``"1h"``).
        start_ms:
            Page start in UTC milliseconds (inclusive).
        end_ms:
            Overall end in UTC milliseconds (exclusive).

        Returns
        -------
        pandas.DataFrame
            Normalised page, possibly empty.
        """
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms - 1,  # Binance endTime is inclusive; subtract 1 ms
            "limit": self._bars_per_request,
        }

        url = f"{self._base_url}{_KLINES_ENDPOINT}"
        logger.debug("GET %s params=%s", url, params)

        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()

        raw_rows: list[list[Any]] = response.json()
        return _parse_klines(raw_rows, symbol)

    @staticmethod
    def _clean(df: pd.DataFrame, end_ms: int) -> pd.DataFrame:
        """Sort, deduplicate, and trim to the exclusive end boundary.

        Parameters
        ----------
        df:
            Concatenated multi-page DataFrame.
        end_ms:
            Exclusive end boundary in UTC milliseconds.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame.
        """
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicate timestamps — keep the first occurrence.
        duplicate_mask = df["timestamp"].duplicated(keep="first")
        n_dupes = int(duplicate_mask.sum())
        if n_dupes:
            logger.warning("Dropping %d duplicate timestamps", n_dupes)
            df = df[~duplicate_mask].reset_index(drop=True)

        # Exclude bars at or beyond the exclusive end boundary.
        end_ts = pd.Timestamp(end_ms, unit="ms", tz="UTC")
        df = df[df["timestamp"] < end_ts].reset_index(drop=True)

        return df
