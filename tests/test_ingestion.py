"""Tests for the ingestion layer (Step 2 of the MVP).

All tests use synthetic data and mock HTTP responses — no network access.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.ingestion.base import OHLCV_COLUMNS, MarketDataSource
from src.ingestion.market_data import BinanceMarketDataSource, _parse_klines, _to_utc_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kline_row(
    open_time_ms: int,
    open_: float = 1.0,
    high: float = 2.0,
    low: float = 0.5,
    close: float = 1.5,
    volume: float = 100.0,
) -> list[Any]:
    """Build a minimal Binance kline row with the fields the adapter uses."""
    return [
        open_time_ms,
        str(open_),
        str(high),
        str(low),
        str(close),
        str(volume),
        open_time_ms + 3_599_999,  # close time (unused)
        "0",                        # quote asset volume (unused)
    ]


def _make_source(
    base_url: str = "https://api.binance.com",
    bars_per_request: int = 1000,
    request_delay_seconds: float = 0.0,
) -> BinanceMarketDataSource:
    return BinanceMarketDataSource(
        api_key="test-key",
        api_secret="test-secret",
        base_url=base_url,
        bars_per_request=bars_per_request,
        request_delay_seconds=request_delay_seconds,
    )


def _mock_response(rows: list[list[Any]], status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response that returns *rows* as JSON."""
    mock = MagicMock(spec=requests.Response)
    mock.status_code = status_code
    mock.json.return_value = rows
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.HTTPError(
            response=mock
        )
    return mock


# ---------------------------------------------------------------------------
# _to_utc_ms
# ---------------------------------------------------------------------------

class TestToUtcMs:
    def test_naive_datetime_treated_as_utc(self) -> None:
        dt = datetime(2022, 1, 1, 0, 0, 0)
        expected = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1_000)
        assert _to_utc_ms(dt) == expected

    def test_aware_datetime_converts_correctly(self) -> None:
        dt = datetime(2022, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        assert _to_utc_ms(dt) == int(dt.timestamp() * 1_000)

    def test_returns_integer(self) -> None:
        dt = datetime(2023, 3, 1, tzinfo=timezone.utc)
        result = _to_utc_ms(dt)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# _parse_klines
# ---------------------------------------------------------------------------

class TestParseKlines:
    def test_empty_rows_returns_empty_dataframe(self) -> None:
        df = _parse_klines([], "ETHUSDT")
        assert df.empty
        assert list(df.columns) == list(OHLCV_COLUMNS)

    def test_columns_match_schema(self) -> None:
        rows = [_make_kline_row(1_640_995_200_000)]
        df = _parse_klines(rows, "ETHUSDT")
        assert list(df.columns) == list(OHLCV_COLUMNS)

    def test_symbol_column_is_set(self) -> None:
        rows = [_make_kline_row(1_640_995_200_000)]
        df = _parse_klines(rows, "BNBUSDT")
        assert df["symbol"].iloc[0] == "BNBUSDT"

    def test_timestamp_is_utc_aware(self) -> None:
        rows = [_make_kline_row(1_640_995_200_000)]
        df = _parse_klines(rows, "ETHUSDT")
        ts = df["timestamp"].iloc[0]
        assert ts.tzinfo is not None
        assert str(ts.tzinfo) == "UTC"

    def test_timestamp_value_correct(self) -> None:
        # 2022-01-01 00:00:00 UTC
        ms = 1_640_995_200_000
        rows = [_make_kline_row(ms)]
        df = _parse_klines(rows, "ETHUSDT")
        expected = pd.Timestamp(ms, unit="ms", tz="UTC")
        assert df["timestamp"].iloc[0] == expected

    def test_ohlcv_values_parsed_as_float(self) -> None:
        rows = [_make_kline_row(1_640_995_200_000, open_=3_000.5, high=3_100.0, low=2_950.0, close=3_050.0, volume=500.25)]
        df = _parse_klines(rows, "ETHUSDT")
        assert df["open"].iloc[0] == pytest.approx(3_000.5)
        assert df["high"].iloc[0] == pytest.approx(3_100.0)
        assert df["low"].iloc[0] == pytest.approx(2_950.0)
        assert df["close"].iloc[0] == pytest.approx(3_050.0)
        assert df["volume"].iloc[0] == pytest.approx(500.25)

    def test_dtypes_are_float64(self) -> None:
        rows = [_make_kline_row(1_640_995_200_000)]
        df = _parse_klines(rows, "ETHUSDT")
        for col in ("open", "high", "low", "close", "volume"):
            assert df[col].dtype == "float64", f"{col} should be float64"

    def test_multiple_rows_preserves_order(self) -> None:
        ms_base = 1_640_995_200_000
        hour_ms = 3_600_000
        rows = [_make_kline_row(ms_base + i * hour_ms) for i in range(5)]
        df = _parse_klines(rows, "ETHUSDT")
        assert len(df) == 5
        assert df["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# BinanceMarketDataSource — resolve_interval
# ---------------------------------------------------------------------------

class TestResolveInterval:
    def test_known_timeframes_are_accepted(self) -> None:
        source = _make_source()
        for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"):
            assert source._resolve_interval(tf) == tf

    def test_unknown_timeframe_raises_value_error(self) -> None:
        source = _make_source()
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            source._resolve_interval("2d")


# ---------------------------------------------------------------------------
# BinanceMarketDataSource — fetch_ohlcv (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchOhlcv:
    """Tests for fetch_ohlcv using mocked HTTP responses."""

    _START = datetime(2022, 1, 1, tzinfo=timezone.utc)
    _END = datetime(2022, 1, 2, tzinfo=timezone.utc)

    def _patch_get(self, rows: list[list[Any]]) -> MagicMock:
        return _mock_response(rows)

    def test_single_page_returns_correct_dataframe(self) -> None:
        ms_base = _to_utc_ms(self._START)
        hour_ms = 3_600_000
        rows = [_make_kline_row(ms_base + i * hour_ms) for i in range(24)]

        source = _make_source()
        with patch.object(source._session, "get", return_value=self._patch_get(rows)):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        assert not df.empty
        assert list(df.columns) == list(OHLCV_COLUMNS)
        assert (df["symbol"] == "ETHUSDT").all()

    def test_empty_response_returns_empty_dataframe(self) -> None:
        source = _make_source()
        with patch.object(source._session, "get", return_value=self._patch_get([])):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        assert df.empty
        assert list(df.columns) == list(OHLCV_COLUMNS)

    def test_result_timestamps_are_utc_aware(self) -> None:
        ms_base = _to_utc_ms(self._START)
        rows = [_make_kline_row(ms_base)]
        source = _make_source()
        with patch.object(source._session, "get", return_value=self._patch_get(rows)):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        assert df["timestamp"].dt.tz is not None

    def test_result_is_sorted_ascending(self) -> None:
        ms_base = _to_utc_ms(self._START)
        hour_ms = 3_600_000
        # Return rows in reverse order to verify sorting
        rows = [_make_kline_row(ms_base + i * hour_ms) for i in range(5, -1, -1)]
        source = _make_source()
        with patch.object(source._session, "get", return_value=self._patch_get(rows)):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        assert df["timestamp"].is_monotonic_increasing

    def test_bars_at_or_after_end_are_excluded(self) -> None:
        """Bars with timestamp >= end must be excluded (exclusive end boundary)."""
        end_ms = _to_utc_ms(self._END)
        # One bar exactly at end, one bar 1 hour before
        rows = [
            _make_kline_row(end_ms - 3_600_000),
            _make_kline_row(end_ms),          # must be excluded
            _make_kline_row(end_ms + 3_600_000),  # must be excluded
        ]
        source = _make_source()
        with patch.object(source._session, "get", return_value=self._patch_get(rows)):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        # Only the bar 1 hour before end should remain
        assert len(df) == 1
        assert df["timestamp"].iloc[0] < pd.Timestamp(self._END)

    def test_duplicate_timestamps_are_dropped(self) -> None:
        ms_base = _to_utc_ms(self._START)
        rows = [
            _make_kline_row(ms_base),
            _make_kline_row(ms_base),  # duplicate
            _make_kline_row(ms_base + 3_600_000),
        ]
        source = _make_source()
        with patch.object(source._session, "get", return_value=self._patch_get(rows)):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        assert len(df) == 2
        assert not df["timestamp"].duplicated().any()

    def test_http_error_propagates(self) -> None:
        source = _make_source()
        error_response = _mock_response([], status_code=429)
        with patch.object(source._session, "get", return_value=error_response):
            with pytest.raises(requests.HTTPError):
                source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

    def test_pagination_fetches_multiple_pages(self) -> None:
        """When bars_per_request is small, multiple HTTP calls must be made."""
        ms_base = _to_utc_ms(self._START)
        hour_ms = 3_600_000

        # 24 bars total, limit = 10 → should trigger 3 pages
        all_rows = [_make_kline_row(ms_base + i * hour_ms) for i in range(24)]

        call_count = 0
        page_size = 10

        def _get_side_effect(url: str, params: dict, timeout: int) -> MagicMock:
            nonlocal call_count
            start_ms = params["startTime"]
            # Return up to page_size rows starting from start_ms
            page_rows = [
                r for r in all_rows
                if r[0] >= start_ms
            ][:page_size]
            call_count += 1
            return _mock_response(page_rows)

        source = _make_source(bars_per_request=page_size, request_delay_seconds=0.0)
        with patch.object(source._session, "get", side_effect=_get_side_effect):
            df = source.fetch_ohlcv("ETHUSDT", self._START, self._END, "1h")

        assert len(df) == 24
        assert call_count > 1

    def test_unsupported_timeframe_raises_before_http(self) -> None:
        source = _make_source()
        with patch.object(source._session, "get") as mock_get:
            with pytest.raises(ValueError, match="Unsupported timeframe"):
                source.fetch_ohlcv("ETHUSDT", self._START, self._END, "2d")
            mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# MarketDataSource — abstract base enforcement
# ---------------------------------------------------------------------------

class TestMarketDataSourceAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            MarketDataSource()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_fetch_ohlcv(self) -> None:
        class Incomplete(MarketDataSource):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        class Complete(MarketDataSource):
            def fetch_ohlcv(self, symbol, start, end, timeframe):
                return pd.DataFrame(columns=list(OHLCV_COLUMNS))

        instance = Complete()
        assert isinstance(instance, MarketDataSource)
