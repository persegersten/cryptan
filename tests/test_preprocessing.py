"""Tests for the preprocessing layer (Step 3 of the MVP).

Covers:
- clean_market_data: column validation, UTC timestamps, sorting, deduplication,
  missing-value handling, and empty-input guards.
- merge_symbol_frames: symbol-prefixed columns, inner/outer join alignment,
  duplicate timestamp handling in merged output, empty-input guards.

All tests use deterministic synthetic DataFrames — no network access.
"""

from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from src.ingestion.base import OHLCV_COLUMNS
from src.preprocessing.cleaner import clean_market_data
from src.preprocessing.merger import merge_symbol_frames


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    timestamps: list[pd.Timestamp],
    symbol: str = "ETHUSDT",
    close_values: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal valid OHLCV DataFrame from a list of UTC timestamps."""
    n = len(timestamps)
    closes = close_values if close_values is not None else [float(i + 1) for i in range(n)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [float(i + 1) for i in range(n)],
            "high": [float(i + 2) for i in range(n)],
            "low": [float(i) for i in range(n)],
            "close": closes,
            "volume": [float(i + 100) for i in range(n)],
            "symbol": symbol,
        }
    )


def _hourly_timestamps(
    start: str = "2022-01-01",
    periods: int = 5,
) -> list[pd.Timestamp]:
    """Return a list of UTC-aware hourly timestamps."""
    return list(pd.date_range(start=start, periods=periods, freq="1h", tz="UTC"))


# ---------------------------------------------------------------------------
# clean_market_data — column validation
# ---------------------------------------------------------------------------


class TestCleanMarketDataColumnValidation:
    def test_raises_if_required_column_missing(self) -> None:
        ts = _hourly_timestamps()
        df = _make_ohlcv(ts)
        df_bad = df.drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required columns"):
            clean_market_data(df_bad)

    def test_raises_for_each_missing_column(self) -> None:
        for col in OHLCV_COLUMNS:
            ts = _hourly_timestamps()
            df = _make_ohlcv(ts)
            df_bad = df.drop(columns=[col])
            with pytest.raises(ValueError, match="missing required columns"):
                clean_market_data(df_bad)

    def test_valid_schema_does_not_raise(self) -> None:
        df = _make_ohlcv(_hourly_timestamps())
        result = clean_market_data(df)
        assert not result.empty


# ---------------------------------------------------------------------------
# clean_market_data — UTC timestamp handling
# ---------------------------------------------------------------------------


class TestCleanMarketDataUTCTimestamps:
    def test_utc_aware_timestamps_unchanged(self) -> None:
        ts = _hourly_timestamps()
        df = _make_ohlcv(ts)
        result = clean_market_data(df)
        assert result["timestamp"].dt.tz is not None
        assert str(result["timestamp"].dt.tz) == "UTC"

    def test_naive_timestamps_localized_to_utc(self) -> None:
        naive_ts = list(
            pd.date_range(start="2022-01-01", periods=5, freq="1h")
        )  # no tz
        df = _make_ohlcv(naive_ts)
        result = clean_market_data(df)
        assert str(result["timestamp"].dt.tz) == "UTC"

    def test_non_utc_timezone_converted(self) -> None:
        eastern_ts = list(
            pd.date_range(start="2022-01-01", periods=5, freq="1h", tz="US/Eastern")
        )
        df = _make_ohlcv(eastern_ts)
        result = clean_market_data(df)
        assert str(result["timestamp"].dt.tz) == "UTC"

    def test_timestamp_values_preserved_after_utc_conversion(self) -> None:
        ts_utc = pd.Timestamp("2022-01-01 12:00:00", tz="UTC")
        ts_eastern = ts_utc.tz_convert("US/Eastern")
        df = _make_ohlcv([ts_eastern])
        result = clean_market_data(df)
        assert result["timestamp"].iloc[0] == ts_utc


# ---------------------------------------------------------------------------
# clean_market_data — sorting
# ---------------------------------------------------------------------------


class TestCleanMarketDataSorting:
    def test_unsorted_input_is_sorted_ascending(self) -> None:
        ts = _hourly_timestamps(periods=5)
        ts_reversed = list(reversed(ts))
        df = _make_ohlcv(ts_reversed)
        result = clean_market_data(df)
        assert result["timestamp"].is_monotonic_increasing

    def test_already_sorted_input_remains_sorted(self) -> None:
        df = _make_ohlcv(_hourly_timestamps(periods=5))
        result = clean_market_data(df)
        assert result["timestamp"].is_monotonic_increasing

    def test_sort_preserves_correct_ohlcv_row_alignment(self) -> None:
        """After sorting, close values must correspond to their original timestamps."""
        ts = _hourly_timestamps(periods=3)
        # Build in reverse order
        df = pd.DataFrame(
            {
                "timestamp": [ts[2], ts[0], ts[1]],
                "open": [30.0, 10.0, 20.0],
                "high": [31.0, 11.0, 21.0],
                "low": [29.0, 9.0, 19.0],
                "close": [30.5, 10.5, 20.5],
                "volume": [300.0, 100.0, 200.0],
                "symbol": "ETHUSDT",
            }
        )
        result = clean_market_data(df)
        assert list(result["close"]) == [10.5, 20.5, 30.5]


# ---------------------------------------------------------------------------
# clean_market_data — duplicate handling
# ---------------------------------------------------------------------------


class TestCleanMarketDataDuplicates:
    def test_duplicate_timestamps_are_removed(self) -> None:
        ts = _hourly_timestamps(periods=3)
        df = _make_ohlcv([ts[0], ts[0], ts[1], ts[2]])  # ts[0] duplicated
        result = clean_market_data(df)
        assert len(result) == 3
        assert not result["timestamp"].duplicated().any()

    def test_first_occurrence_of_duplicate_is_kept(self) -> None:
        ts = _hourly_timestamps(periods=2)
        df = pd.DataFrame(
            {
                "timestamp": [ts[0], ts[0]],
                "open": [1.0, 99.0],  # different close values for each duplicate
                "high": [2.0, 100.0],
                "low": [0.5, 98.0],
                "close": [1.5, 99.5],
                "volume": [10.0, 20.0],
                "symbol": "ETHUSDT",
            }
        )
        result = clean_market_data(df)
        assert len(result) == 1
        assert result["close"].iloc[0] == pytest.approx(1.5)

    def test_no_duplicates_in_input_returns_same_count(self) -> None:
        ts = _hourly_timestamps(periods=5)
        df = _make_ohlcv(ts)
        result = clean_market_data(df)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# clean_market_data — missing value handling
# ---------------------------------------------------------------------------


class TestCleanMarketDataMissingValues:
    def test_row_with_nan_close_is_dropped(self) -> None:
        ts = _hourly_timestamps(periods=3)
        df = _make_ohlcv(ts, close_values=[1.0, float("nan"), 3.0])
        result = clean_market_data(df)
        assert len(result) == 2
        assert not result["close"].isna().any()

    def test_row_with_nan_volume_is_dropped(self) -> None:
        ts = _hourly_timestamps(periods=3)
        df = _make_ohlcv(ts)
        df.loc[1, "volume"] = float("nan")
        result = clean_market_data(df)
        assert len(result) == 2

    def test_nan_in_symbol_column_does_not_drop_row(self) -> None:
        """Only critical OHLCV columns trigger row removal."""
        ts = _hourly_timestamps(periods=3)
        df = _make_ohlcv(ts)
        df.loc[1, "symbol"] = None  # type: ignore[call-overload]
        result = clean_market_data(df)
        # Row with NaN symbol must NOT be dropped
        assert len(result) == 3

    def test_all_rows_nan_raises_after_cleaning(self) -> None:
        ts = _hourly_timestamps(periods=2)
        df = _make_ohlcv(ts, close_values=[float("nan"), float("nan")])
        with pytest.raises(ValueError, match="empty after cleaning"):
            clean_market_data(df)


# ---------------------------------------------------------------------------
# clean_market_data — empty input guard
# ---------------------------------------------------------------------------


class TestCleanMarketDataEmptyGuard:
    def test_empty_dataframe_raises_before_processing(self) -> None:
        df = pd.DataFrame(columns=list(OHLCV_COLUMNS))
        with pytest.raises(ValueError, match="empty DataFrame"):
            clean_market_data(df)


# ---------------------------------------------------------------------------
# clean_market_data — original is not mutated
# ---------------------------------------------------------------------------


class TestCleanMarketDataImmutability:
    def test_original_dataframe_not_mutated(self) -> None:
        ts = _hourly_timestamps(periods=5)
        df = _make_ohlcv(ts)
        original_len = len(df)
        original_cols = list(df.columns)
        clean_market_data(df)
        assert len(df) == original_len
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# merge_symbol_frames — column naming
# ---------------------------------------------------------------------------


class TestMergeSymbolFramesColumnNaming:
    def test_merged_columns_are_symbol_prefixed(self) -> None:
        ts = _hourly_timestamps(periods=5)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames)
        assert "ETH_close" in result.columns
        assert "ETH_volume" in result.columns
        assert "BNB_close" in result.columns
        assert "BNB_volume" in result.columns

    def test_symbol_column_not_present_in_merged_output(self) -> None:
        ts = _hourly_timestamps(periods=3)
        frames = {"ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT"))}
        result = merge_symbol_frames(frames)
        assert "symbol" not in result.columns

    def test_timestamp_column_present_and_unique(self) -> None:
        ts = _hourly_timestamps(periods=3)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "SOL": clean_market_data(_make_ohlcv(ts, symbol="SOLUSDT")),
        }
        result = merge_symbol_frames(frames)
        assert "timestamp" in result.columns
        assert result.columns.tolist().count("timestamp") == 1

    def test_no_ambiguous_duplicate_columns(self) -> None:
        ts = _hourly_timestamps(periods=5)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT")),
            "SOL": clean_market_data(_make_ohlcv(ts, symbol="SOLUSDT")),
        }
        result = merge_symbol_frames(frames)
        assert len(result.columns) == len(set(result.columns))

    def test_three_symbols_produce_expected_column_count(self) -> None:
        ts = _hourly_timestamps(periods=5)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT")),
            "SOL": clean_market_data(_make_ohlcv(ts, symbol="SOLUSDT")),
        }
        result = merge_symbol_frames(frames)
        # 1 timestamp + 3 symbols × 5 OHLCV columns
        assert len(result.columns) == 1 + 3 * 5


# ---------------------------------------------------------------------------
# merge_symbol_frames — inner join alignment
# ---------------------------------------------------------------------------


class TestMergeSymbolFramesInnerJoin:
    def test_inner_join_keeps_only_common_timestamps(self) -> None:
        ts_eth = _hourly_timestamps(start="2022-01-01", periods=5)
        ts_bnb = _hourly_timestamps(start="2022-01-01 02:00", periods=5)
        # Overlap: hours 02, 03, 04 (3 rows)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts_eth, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts_bnb, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames, join="inner")
        assert len(result) == 3

    def test_inner_join_result_is_sorted_ascending(self) -> None:
        ts = _hourly_timestamps(periods=5)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames, join="inner")
        assert result["timestamp"].is_monotonic_increasing

    def test_inner_join_no_nan_in_numeric_columns(self) -> None:
        ts = _hourly_timestamps(periods=5)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames, join="inner")
        numeric_cols = [c for c in result.columns if c != "timestamp"]
        assert not result[numeric_cols].isna().any().any()

    def test_no_overlap_raises_value_error(self) -> None:
        ts_eth = _hourly_timestamps(start="2022-01-01", periods=3)
        ts_bnb = _hourly_timestamps(start="2022-02-01", periods=3)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts_eth, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts_bnb, symbol="BNBUSDT")),
        }
        with pytest.raises(ValueError, match="empty"):
            merge_symbol_frames(frames, join="inner")


# ---------------------------------------------------------------------------
# merge_symbol_frames — outer join
# ---------------------------------------------------------------------------


class TestMergeSymbolFramesOuterJoin:
    def test_outer_join_preserves_all_timestamps(self) -> None:
        ts_eth = _hourly_timestamps(start="2022-01-01", periods=3)
        ts_bnb = _hourly_timestamps(start="2022-01-01 02:00", periods=3)
        # ETH: 00, 01, 02 | BNB: 02, 03, 04 → union has 5 rows
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts_eth, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts_bnb, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames, join="outer")
        assert len(result) == 5

    def test_outer_join_introduces_nan_for_missing_bars(self) -> None:
        ts_eth = _hourly_timestamps(start="2022-01-01", periods=3)
        ts_bnb = _hourly_timestamps(start="2022-01-01 02:00", periods=3)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts_eth, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts_bnb, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames, join="outer")
        # ETH has no bar at hours 03 and 04 → NaN in ETH_close
        assert result["ETH_close"].isna().sum() == 2


# ---------------------------------------------------------------------------
# merge_symbol_frames — timestamp alignment exactness
# ---------------------------------------------------------------------------


class TestMergeSymbolFramesTimestampAlignment:
    def test_merged_timestamps_match_exact_input_values(self) -> None:
        ts = _hourly_timestamps(start="2022-06-15", periods=4)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT")),
        }
        result = merge_symbol_frames(frames)
        for expected, actual in zip(ts, result["timestamp"]):
            assert actual == expected

    def test_correct_close_values_aligned_to_timestamps(self) -> None:
        """Close values must stay correctly aligned after merge."""
        ts = _hourly_timestamps(periods=3)
        eth_closes = [100.0, 200.0, 300.0]
        bnb_closes = [10.0, 20.0, 30.0]
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT", close_values=eth_closes)),
            "BNB": clean_market_data(_make_ohlcv(ts, symbol="BNBUSDT", close_values=bnb_closes)),
        }
        result = merge_symbol_frames(frames)
        assert list(result["ETH_close"]) == pytest.approx(eth_closes)
        assert list(result["BNB_close"]) == pytest.approx(bnb_closes)


# ---------------------------------------------------------------------------
# merge_symbol_frames — single symbol
# ---------------------------------------------------------------------------


class TestMergeSymbolFramesSingleSymbol:
    def test_single_symbol_frame_returns_prefixed_columns(self) -> None:
        ts = _hourly_timestamps(periods=5)
        frames = {"ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT"))}
        result = merge_symbol_frames(frames)
        assert "ETH_close" in result.columns
        assert "timestamp" in result.columns
        assert len(result) == 5


# ---------------------------------------------------------------------------
# merge_symbol_frames — error guards
# ---------------------------------------------------------------------------


class TestMergeSymbolFramesErrorGuards:
    def test_empty_frames_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            merge_symbol_frames({})

    def test_empty_frame_raises(self) -> None:
        ts = _hourly_timestamps(periods=3)
        frames = {
            "ETH": clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT")),
            "BNB": pd.DataFrame(columns=list(OHLCV_COLUMNS)),
        }
        with pytest.raises(ValueError, match="empty"):
            merge_symbol_frames(frames)

    def test_frame_missing_column_raises(self) -> None:
        ts = _hourly_timestamps(periods=3)
        df_ok = clean_market_data(_make_ohlcv(ts, symbol="ETHUSDT"))
        df_bad = _make_ohlcv(ts, symbol="BNBUSDT").drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            merge_symbol_frames({"ETH": df_ok, "BNB": df_bad})
