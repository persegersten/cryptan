"""Tests for feature engineering (Step 4 of the MVP).

Covers:
- Symbol-prefixed column naming after feature construction.
- No future leakage: features at row t only use data up to row t.
- Insufficient lookback rows are dropped.
- Return feature correctness (pct_change formula).
- Moving average and ratio correctness.
- Volatility feature correctness.
- Volume mean and z-score correctness.
- Cross-asset relative strength and correlation features.
- Single-symbol configuration (no cross-asset features generated).
- Error guards: empty input, missing timestamp column.
- Feature DataFrame preserves temporal order after construction.

All tests use deterministic synthetic DataFrames — no network access.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config.model import FeatureConfig, TrainingConfig
from src.features.builder import build_features
from src.features.technical import (
    add_cross_asset_features,
    add_moving_average_features,
    add_return_features,
    add_volatility_features,
    add_volume_features,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


def _make_config(
    signal_symbols: list[str] | None = None,
    trading_symbol: str = "ETH",
    feature_config: FeatureConfig | None = None,
) -> TrainingConfig:
    """Build a minimal TrainingConfig for testing."""
    return TrainingConfig(
        trading_symbol=trading_symbol,
        signal_symbols=signal_symbols or ["ETH", "BNB", "SOL"],
        timeframe="1h",
        start_date="2022-01-01",
        end_date="2024-01-01",
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
        feature_config=feature_config or FeatureConfig(),
    )


def _hourly_timestamps(start: str = "2022-01-01", periods: int = 50) -> list[pd.Timestamp]:
    """Return a list of UTC-aware hourly timestamps."""
    return list(pd.date_range(start=start, periods=periods, freq="1h", tz="UTC"))


def _make_merged_df(
    periods: int = 50,
    symbols: list[str] | None = None,
    close_values: dict[str, list[float]] | None = None,
    volume_values: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Build a minimal wide merged DataFrame (output of merge_symbol_frames)."""
    symbols = symbols or ["ETH", "BNB", "SOL"]
    timestamps = _hourly_timestamps(periods=periods)
    data: dict[str, object] = {"timestamp": timestamps}
    for sym in symbols:
        closes = (
            close_values.get(sym)
            if close_values
            else [float(i + 1) * (1 + 0.001 * hash(sym) % 0.1) for i in range(periods)]
        )
        vols = (
            volume_values.get(sym)
            if volume_values
            else [float(i + 100) for i in range(periods)]
        )
        data[f"{sym}_open"] = [float(i + 1) for i in range(periods)]
        data[f"{sym}_high"] = [float(i + 2) for i in range(periods)]
        data[f"{sym}_low"] = [float(i) for i in range(periods)]
        data[f"{sym}_close"] = closes
        data[f"{sym}_volume"] = vols
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# build_features — column naming
# ---------------------------------------------------------------------------


class TestBuildFeaturesColumnNaming:
    def test_return_columns_are_symbol_prefixed(self) -> None:
        df = _make_merged_df()
        config = _make_config(signal_symbols=["ETH"])
        result = build_features(df, config)
        for window in config.feature_config.return_windows:
            assert f"ETH_return_{window}" in result.columns

    def test_ma_columns_are_symbol_prefixed(self) -> None:
        df = _make_merged_df()
        config = _make_config(signal_symbols=["ETH"])
        result = build_features(df, config)
        assert "ETH_ma_short" in result.columns
        assert "ETH_ma_long" in result.columns
        assert "ETH_ma_ratio" in result.columns

    def test_volatility_column_is_symbol_prefixed(self) -> None:
        df = _make_merged_df()
        config = _make_config(signal_symbols=["ETH"])
        result = build_features(df, config)
        assert "ETH_volatility" in result.columns

    def test_volume_columns_are_symbol_prefixed(self) -> None:
        df = _make_merged_df()
        config = _make_config(signal_symbols=["ETH"])
        result = build_features(df, config)
        assert "ETH_volume_mean" in result.columns
        assert "ETH_volume_zscore" in result.columns

    def test_cross_asset_columns_use_target_other_prefix(self) -> None:
        df = _make_merged_df(symbols=["ETH", "BNB"])
        config = _make_config(signal_symbols=["ETH", "BNB"], trading_symbol="ETH")
        result = build_features(df, config)
        assert "ETH_BNB_rel_strength" in result.columns
        assert "ETH_BNB_corr" in result.columns

    def test_all_three_symbols_produce_expected_feature_columns(self) -> None:
        df = _make_merged_df()
        config = _make_config()
        result = build_features(df, config)
        for sym in ["ETH", "BNB", "SOL"]:
            for window in config.feature_config.return_windows:
                assert f"{sym}_return_{window}" in result.columns
            assert f"{sym}_ma_short" in result.columns
            assert f"{sym}_volatility" in result.columns
            assert f"{sym}_volume_mean" in result.columns
        # Cross-asset: ETH vs BNB and ETH vs SOL
        assert "ETH_BNB_rel_strength" in result.columns
        assert "ETH_SOL_rel_strength" in result.columns

    def test_no_duplicate_columns_after_feature_generation(self) -> None:
        df = _make_merged_df()
        config = _make_config()
        result = build_features(df, config)
        assert len(result.columns) == len(set(result.columns))

    def test_timestamp_column_preserved(self) -> None:
        df = _make_merged_df()
        config = _make_config()
        result = build_features(df, config)
        assert "timestamp" in result.columns


# ---------------------------------------------------------------------------
# build_features — no future leakage
# ---------------------------------------------------------------------------


class TestNoFutureLeakage:
    """Features at row t must not change when rows after t are added or removed."""

    def test_truncating_future_rows_does_not_change_past_features(self) -> None:
        """Build features on N rows and N+10 rows; compare at the last common row."""
        periods_full = 50
        periods_short = 40

        # Use strictly increasing close so return values are well-defined.
        eth_close = [float(i + 10) for i in range(periods_full)]
        bnb_close = [float(i + 5) for i in range(periods_full)]

        df_full = _make_merged_df(
            periods=periods_full,
            symbols=["ETH", "BNB"],
            close_values={"ETH": eth_close, "BNB": bnb_close},
        )
        df_short = df_full.iloc[:periods_short].copy().reset_index(drop=True)

        # Use small windows to keep enough rows after lookback drop.
        feature_cfg = FeatureConfig(
            return_windows=[1, 5],
            ma_short_window=3,
            ma_long_window=7,
            volatility_window=7,
            volume_window=7,
            correlation_window=7,
        )
        config = _make_config(
            signal_symbols=["ETH", "BNB"],
            trading_symbol="ETH",
            feature_config=feature_cfg,
        )

        result_full = build_features(df_full, config)
        result_short = build_features(df_short, config)

        # Find last timestamp in the shorter result.
        last_ts = result_short["timestamp"].max()

        row_full = result_full[result_full["timestamp"] == last_ts]
        row_short = result_short[result_short["timestamp"] == last_ts]

        # Both must contain exactly one row at that timestamp.
        assert len(row_full) == 1
        assert len(row_short) == 1

        feature_cols = [c for c in result_short.columns if c != "timestamp"]
        for col in feature_cols:
            assert row_full[col].values[0] == pytest.approx(
                row_short[col].values[0], rel=1e-9
            ), f"Leakage detected in column '{col}'"

    def test_rolling_mean_does_not_use_future_bars(self) -> None:
        """Rolling mean at row t must equal mean of rows [t-window+1 .. t]."""
        closes = [float(i + 1) for i in range(30)]
        df = _make_merged_df(
            periods=30, symbols=["ETH"],
            close_values={"ETH": closes},
        )
        feature_cfg = FeatureConfig(
            return_windows=[1],
            ma_short_window=3,
            ma_long_window=5,
            volatility_window=5,
            volume_window=5,
            correlation_window=5,
        )
        config = _make_config(signal_symbols=["ETH"], feature_config=feature_cfg)
        result = build_features(df, config)

        # Verify ma_short at a specific row against manual computation.
        row_idx = 0  # first valid row after lookback drop
        ts = result["timestamp"].iloc[row_idx]
        original_pos = df[df["timestamp"] == ts].index[0]

        expected_ma3 = sum(closes[original_pos - 2 : original_pos + 1]) / 3
        assert result["ETH_ma_short"].iloc[row_idx] == pytest.approx(expected_ma3, rel=1e-9)


# ---------------------------------------------------------------------------
# build_features — lookback row dropping
# ---------------------------------------------------------------------------


class TestLookbackRowDropping:
    def test_rows_with_insufficient_lookback_are_dropped(self) -> None:
        """With window=20 the first 20 rows must be dropped."""
        periods = 50
        df = _make_merged_df(periods=periods, symbols=["ETH"])
        config = _make_config(
            signal_symbols=["ETH"],
            feature_config=FeatureConfig(
                return_windows=[1, 5, 20],
                ma_short_window=7,
                ma_long_window=20,
                volatility_window=20,
                volume_window=20,
                correlation_window=20,
            ),
        )
        result = build_features(df, config)
        # return_20 needs 20 rows ahead; volatility/vol/MA_long also need 20.
        # The last NaN is at index 19 (0-based), so 20 rows dropped.
        assert len(result) == periods - 20

    def test_result_contains_no_nan_in_feature_columns(self) -> None:
        df = _make_merged_df(periods=50, symbols=["ETH"])
        config = _make_config(signal_symbols=["ETH"])
        result = build_features(df, config)
        feature_cols = [c for c in result.columns if c != "timestamp"]
        assert not result[feature_cols].isna().any().any()

    def test_too_short_series_raises(self) -> None:
        """Dataset shorter than the maximum window size should raise."""
        periods = 5  # far too short for default windows of 20
        df = _make_merged_df(periods=periods, symbols=["ETH"])
        config = _make_config(signal_symbols=["ETH"])
        with pytest.raises(ValueError, match="empty"):
            build_features(df, config)

    def test_dropped_rows_have_earlier_timestamps(self) -> None:
        """Dropped rows must be at the beginning (earliest timestamps)."""
        periods = 50
        df = _make_merged_df(periods=periods, symbols=["ETH"])
        config = _make_config(
            signal_symbols=["ETH"],
            feature_config=FeatureConfig(
                return_windows=[1],
                ma_short_window=3,
                ma_long_window=5,
                volatility_window=5,
                volume_window=5,
                correlation_window=5,
            ),
        )
        result = build_features(df, config)
        first_kept_ts = result["timestamp"].min()
        original_first_ts = df["timestamp"].min()
        assert first_kept_ts > original_first_ts


# ---------------------------------------------------------------------------
# build_features — temporal order preserved
# ---------------------------------------------------------------------------


class TestTemporalOrder:
    def test_output_is_sorted_ascending_by_timestamp(self) -> None:
        df = _make_merged_df(periods=50, symbols=["ETH"])
        config = _make_config(signal_symbols=["ETH"])
        result = build_features(df, config)
        assert result["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# build_features — error guards
# ---------------------------------------------------------------------------


class TestBuildFeaturesErrorGuards:
    def test_empty_merged_df_raises(self) -> None:
        df = pd.DataFrame(columns=["timestamp", "ETH_close", "ETH_volume",
                                   "ETH_open", "ETH_high", "ETH_low"])
        config = _make_config(signal_symbols=["ETH"])
        with pytest.raises(ValueError, match="empty"):
            build_features(df, config)

    def test_missing_timestamp_column_raises(self) -> None:
        df = _make_merged_df(periods=50, symbols=["ETH"])
        df = df.drop(columns=["timestamp"])
        config = _make_config(signal_symbols=["ETH"])
        with pytest.raises(ValueError, match="timestamp"):
            build_features(df, config)


# ---------------------------------------------------------------------------
# build_features — single signal symbol (no cross-asset features)
# ---------------------------------------------------------------------------


class TestSingleSignalSymbol:
    def test_no_cross_asset_features_when_single_symbol(self) -> None:
        df = _make_merged_df(periods=50, symbols=["ETH"])
        config = _make_config(signal_symbols=["ETH"], trading_symbol="ETH")
        result = build_features(df, config)
        cross_cols = [c for c in result.columns if "_ETH_" in c and c.startswith("ETH_ETH")]
        # No cross-asset features when trading symbol is the only signal symbol.
        assert cross_cols == []


# ---------------------------------------------------------------------------
# technical.add_return_features — correctness
# ---------------------------------------------------------------------------


class TestAddReturnFeatures:
    def test_return_1_bar_equals_pct_change(self) -> None:
        closes = [10.0, 11.0, 12.1, 11.0, 13.0]
        df = pd.DataFrame({"ETH_close": closes})
        result = add_return_features(df, "ETH", return_windows=[1])
        expected = pd.Series(closes).pct_change(1).tolist()
        assert result["ETH_return_1"].tolist() == pytest.approx(expected, nan_ok=True)

    def test_return_5_bar_first_5_rows_are_nan(self) -> None:
        df = pd.DataFrame({"ETH_close": [float(i + 1) for i in range(10)]})
        result = add_return_features(df, "ETH", return_windows=[5])
        assert result["ETH_return_5"].isna().sum() == 5

    def test_missing_close_column_raises(self) -> None:
        df = pd.DataFrame({"ETH_volume": [1.0, 2.0]})
        with pytest.raises(ValueError, match="ETH_close"):
            add_return_features(df, "ETH", return_windows=[1])

    def test_original_df_not_mutated(self) -> None:
        df = pd.DataFrame({"ETH_close": [1.0, 2.0, 3.0]})
        original_cols = list(df.columns)
        add_return_features(df, "ETH", return_windows=[1])
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# technical.add_moving_average_features — correctness
# ---------------------------------------------------------------------------


class TestAddMovingAverageFeatures:
    def test_ma_short_at_valid_row_is_correct(self) -> None:
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"ETH_close": closes})
        result = add_moving_average_features(df, "ETH", short_window=3, long_window=5)
        # Row index 2 (3rd row): mean of [1, 2, 3] = 2.0
        assert result["ETH_ma_short"].iloc[2] == pytest.approx(2.0)

    def test_ma_long_at_valid_row_is_correct(self) -> None:
        closes = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"ETH_close": closes})
        result = add_moving_average_features(df, "ETH", short_window=2, long_window=5)
        # Row index 4: mean of [1, 2, 3, 4, 5] = 3.0
        assert result["ETH_ma_long"].iloc[4] == pytest.approx(3.0)

    def test_ma_ratio_is_short_divided_by_long(self) -> None:
        closes = [2.0, 2.0, 2.0, 2.0, 2.0]  # constant → both MAs = 2.0
        df = pd.DataFrame({"ETH_close": closes})
        result = add_moving_average_features(df, "ETH", short_window=2, long_window=5)
        # First row where both MAs are valid: index 4
        assert result["ETH_ma_ratio"].iloc[4] == pytest.approx(1.0)

    def test_short_window_ge_long_window_raises(self) -> None:
        df = pd.DataFrame({"ETH_close": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="short_window"):
            add_moving_average_features(df, "ETH", short_window=5, long_window=5)

    def test_missing_close_column_raises(self) -> None:
        df = pd.DataFrame({"ETH_volume": [1.0]})
        with pytest.raises(ValueError, match="ETH_close"):
            add_moving_average_features(df, "ETH", short_window=3, long_window=7)


# ---------------------------------------------------------------------------
# technical.add_volatility_features — correctness
# ---------------------------------------------------------------------------


class TestAddVolatilityFeatures:
    def test_volatility_is_nan_for_first_rows(self) -> None:
        df = pd.DataFrame({"ETH_close": [float(i + 1) for i in range(25)]})
        result = add_volatility_features(df, "ETH", volatility_window=20)
        # 1-bar returns start at index 1; rolling std(20) valid from index 20.
        assert result["ETH_volatility"].isna().sum() == 20

    def test_volatility_is_positive_for_varying_prices(self) -> None:
        # Alternating prices → non-zero returns → positive volatility.
        prices = [10.0, 11.0] * 15
        df = pd.DataFrame({"ETH_close": prices})
        result = add_volatility_features(df, "ETH", volatility_window=5)
        valid = result["ETH_volatility"].dropna()
        assert (valid > 0).all()

    def test_volatility_is_zero_for_constant_prices(self) -> None:
        # Constant prices → zero returns → zero volatility.
        df = pd.DataFrame({"ETH_close": [5.0] * 30})
        result = add_volatility_features(df, "ETH", volatility_window=5)
        valid = result["ETH_volatility"].dropna()
        assert (valid.abs() < 1e-9).all()

    def test_missing_close_column_raises(self) -> None:
        df = pd.DataFrame({"ETH_volume": [1.0]})
        with pytest.raises(ValueError, match="ETH_close"):
            add_volatility_features(df, "ETH", volatility_window=5)


# ---------------------------------------------------------------------------
# technical.add_volume_features — correctness
# ---------------------------------------------------------------------------


class TestAddVolumeFeatures:
    def test_volume_mean_at_valid_row_is_correct(self) -> None:
        volumes = [1.0, 2.0, 3.0, 4.0, 5.0]
        df = pd.DataFrame({"ETH_volume": volumes})
        result = add_volume_features(df, "ETH", volume_window=5)
        # Row 4: mean of [1,2,3,4,5] = 3.0
        assert result["ETH_volume_mean"].iloc[4] == pytest.approx(3.0)

    def test_volume_zscore_is_zero_for_constant_volume(self) -> None:
        df = pd.DataFrame({"ETH_volume": [10.0] * 30})
        result = add_volume_features(df, "ETH", volume_window=5)
        valid = result["ETH_volume_zscore"].dropna()
        assert valid.tolist() == pytest.approx([0.0] * len(valid))

    def test_volume_zscore_formula(self) -> None:
        """z-score at row t = (vol[t] - mean) / std over the window."""
        vols = list(range(1, 21))  # 1..20
        df = pd.DataFrame({"ETH_volume": vols})
        result = add_volume_features(df, "ETH", volume_window=5)
        # Row 19 (last): window = [16, 17, 18, 19, 20]
        window_vals = vols[15:20]
        mean = sum(window_vals) / 5
        std = pd.Series(window_vals).std()  # ddof=1 by default
        expected_z = (vols[19] - mean) / std
        assert result["ETH_volume_zscore"].iloc[19] == pytest.approx(expected_z, rel=1e-9)

    def test_missing_volume_column_raises(self) -> None:
        df = pd.DataFrame({"ETH_close": [1.0]})
        with pytest.raises(ValueError, match="ETH_volume"):
            add_volume_features(df, "ETH", volume_window=5)


# ---------------------------------------------------------------------------
# technical.add_cross_asset_features — correctness
# ---------------------------------------------------------------------------


class TestAddCrossAssetFeatures:
    def test_rel_strength_is_target_close_divided_by_other_close(self) -> None:
        df = pd.DataFrame({"ETH_close": [100.0, 200.0], "BNB_close": [50.0, 100.0]})
        result = add_cross_asset_features(df, "ETH", "BNB", correlation_window=2)
        assert list(result["ETH_BNB_rel_strength"]) == pytest.approx([2.0, 2.0])

    def test_correlation_is_nan_for_insufficient_rows(self) -> None:
        df = pd.DataFrame(
            {
                "ETH_close": [float(i + 1) for i in range(5)],
                "BNB_close": [float(i + 2) for i in range(5)],
            }
        )
        result = add_cross_asset_features(df, "ETH", "BNB", correlation_window=5)
        # Correlation over window=5 of 1-bar returns (returns start at row 1)
        # needs 5 return values: rows 1-5, so first valid corr at row 5.
        assert result["ETH_BNB_corr"].isna().sum() >= 1

    def test_perfectly_correlated_series_gives_correlation_one(self) -> None:
        # ETH and BNB move identically → Pearson corr = 1.0
        prices = [float(i + 10) for i in range(25)]
        df = pd.DataFrame({"ETH_close": prices, "BNB_close": prices})
        result = add_cross_asset_features(df, "ETH", "BNB", correlation_window=10)
        valid_corr = result["ETH_BNB_corr"].dropna()
        assert valid_corr.tolist() == pytest.approx([1.0] * len(valid_corr), abs=1e-9)

    def test_missing_target_close_column_raises(self) -> None:
        df = pd.DataFrame({"BNB_close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="ETH_close"):
            add_cross_asset_features(df, "ETH", "BNB", correlation_window=2)

    def test_missing_other_close_column_raises(self) -> None:
        df = pd.DataFrame({"ETH_close": [1.0, 2.0]})
        with pytest.raises(ValueError, match="BNB_close"):
            add_cross_asset_features(df, "ETH", "BNB", correlation_window=2)

    def test_original_df_not_mutated(self) -> None:
        df = pd.DataFrame({"ETH_close": [1.0, 2.0, 3.0], "BNB_close": [1.5, 2.5, 3.5]})
        original_cols = list(df.columns)
        add_cross_asset_features(df, "ETH", "BNB", correlation_window=2)
        assert list(df.columns) == original_cols


# ---------------------------------------------------------------------------
# FeatureConfig — validation
# ---------------------------------------------------------------------------


class TestFeatureConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = FeatureConfig()
        assert cfg.return_windows == [1, 5, 20]
        assert cfg.ma_short_window == 7
        assert cfg.ma_long_window == 20

    def test_custom_return_windows(self) -> None:
        cfg = FeatureConfig(return_windows=[1, 10])
        assert cfg.return_windows == [1, 10]

    def test_zero_ma_short_window_raises(self) -> None:
        with pytest.raises(Exception):
            FeatureConfig(ma_short_window=0)

    def test_zero_volatility_window_raises(self) -> None:
        with pytest.raises(Exception):
            FeatureConfig(volatility_window=0)
