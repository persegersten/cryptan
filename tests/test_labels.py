"""Tests for target-label generation (Step 5 of the MVP).

Covers:
- Explicit future shifting for the configured trading symbol.
- Multiclass target assignment (-1, 0, 1).
- Threshold boundary behaviour.
- Dropping rows without a full future horizon.
- Chronological ordering and input immutability.
- Error guards for missing inputs.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.model import TrainingConfig
from src.labels.target import (
    TARGET_LABEL_COLUMN,
    TARGET_RETURN_COLUMN,
    add_target_labels,
)


_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


def _make_config(
    trading_symbol: str = "ETH",
    signal_symbols: list[str] | None = None,
    horizon: int = 2,
    threshold: float = 0.05,
) -> TrainingConfig:
    """Build a minimal TrainingConfig for target-label tests."""
    return TrainingConfig(
        trading_symbol=trading_symbol,
        signal_symbols=signal_symbols or [trading_symbol],
        timeframe="1h",
        start_date="2022-01-01",
        end_date="2024-01-01",
        prediction_horizon_bars=horizon,
        return_threshold=threshold,
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
    )


def _make_feature_df(
    closes: list[float],
    symbol: str = "ETH",
    start: str = "2022-01-01",
) -> pd.DataFrame:
    """Build a minimal feature-like DataFrame containing timestamp and close."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start=start, periods=len(closes), freq="1h", tz="UTC"),
            f"{symbol}_close": closes,
            f"{symbol}_return_1": [0.0] * len(closes),
        }
    )


# ---------------------------------------------------------------------------
# add_target_labels — future return and class correctness
# ---------------------------------------------------------------------------


class TestAddTargetLabelsCorrectness:
    def test_future_return_uses_configured_horizon(self) -> None:
        df = _make_feature_df([100.0, 110.0, 121.0, 133.1])
        config = _make_config(horizon=2, threshold=0.05)

        result = add_target_labels(df, config)

        # Row 0 return: (121 - 100) / 100 = 0.21.
        # Row 1 return: (133.1 - 110) / 110 = 0.21.
        assert result[TARGET_RETURN_COLUMN].tolist() == pytest.approx([0.21, 0.21])

    def test_positive_neutral_and_negative_labels_are_created(self) -> None:
        df = _make_feature_df([100.0, 110.0, 100.0, 95.0, 95.0])
        config = _make_config(horizon=1, threshold=0.05)

        result = add_target_labels(df, config)

        assert result[TARGET_LABEL_COLUMN].tolist() == [1, -1, 0, 0]

    def test_threshold_boundaries_are_neutral(self) -> None:
        df = _make_feature_df([100.0, 105.0, 99.75])
        config = _make_config(horizon=1, threshold=0.05)

        result = add_target_labels(df, config)

        # +5% and -5% are both neutral because directional classes require
        # strictly greater than threshold magnitude.
        assert result[TARGET_LABEL_COLUMN].tolist() == [0, 0]

    def test_drops_final_rows_without_full_future_horizon(self) -> None:
        df = _make_feature_df([100.0, 101.0, 102.0, 103.0, 104.0])
        config = _make_config(horizon=2, threshold=0.01)

        result = add_target_labels(df, config)

        assert len(result) == len(df) - 2
        assert result["timestamp"].max() == df["timestamp"].iloc[-3]

    def test_labels_target_symbol_when_multiple_symbol_closes_exist(self) -> None:
        df = _make_feature_df([100.0, 110.0, 120.0], symbol="ETH")
        df["BNB_close"] = [100.0, 50.0, 25.0]
        config = _make_config(trading_symbol="BNB", signal_symbols=["ETH", "BNB"], horizon=1)

        result = add_target_labels(df, config)

        assert result[TARGET_LABEL_COLUMN].tolist() == [-1, -1]


# ---------------------------------------------------------------------------
# add_target_labels — time-series safety and shape
# ---------------------------------------------------------------------------


class TestAddTargetLabelsTimeSeriesSafety:
    def test_unsorted_input_is_sorted_before_labeling(self) -> None:
        df = _make_feature_df([100.0, 110.0, 121.0, 133.1])
        df = df.iloc[[2, 0, 3, 1]].reset_index(drop=True)
        config = _make_config(horizon=1, threshold=0.05)

        result = add_target_labels(df, config)

        assert result["timestamp"].is_monotonic_increasing
        assert result[TARGET_RETURN_COLUMN].tolist() == pytest.approx([0.10, 0.10, 0.10])

    def test_output_label_dtype_is_integer(self) -> None:
        df = _make_feature_df([100.0, 110.0, 100.0])
        config = _make_config(horizon=1, threshold=0.05)

        result = add_target_labels(df, config)

        assert pd.api.types.is_integer_dtype(result[TARGET_LABEL_COLUMN])

    def test_original_dataframe_not_mutated(self) -> None:
        df = _make_feature_df([100.0, 110.0, 120.0])
        original_cols = list(df.columns)
        config = _make_config(horizon=1, threshold=0.05)

        add_target_labels(df, config)

        assert list(df.columns) == original_cols
        assert TARGET_LABEL_COLUMN not in df.columns
        assert TARGET_RETURN_COLUMN not in df.columns


# ---------------------------------------------------------------------------
# add_target_labels — error guards
# ---------------------------------------------------------------------------


class TestAddTargetLabelsErrorGuards:
    def test_empty_feature_df_raises(self) -> None:
        df = pd.DataFrame(columns=["timestamp", "ETH_close"])
        config = _make_config()

        with pytest.raises(ValueError, match="empty"):
            add_target_labels(df, config)

    def test_missing_timestamp_column_raises(self) -> None:
        df = pd.DataFrame({"ETH_close": [100.0, 110.0, 120.0]})
        config = _make_config()

        with pytest.raises(ValueError, match="timestamp"):
            add_target_labels(df, config)

    def test_missing_trading_symbol_close_column_raises(self) -> None:
        df = _make_feature_df([100.0, 110.0, 120.0], symbol="ETH")
        config = _make_config(trading_symbol="BTC", signal_symbols=["ETH", "BTC"])

        with pytest.raises(ValueError, match="BTC_close"):
            add_target_labels(df, config)

    def test_too_short_series_for_horizon_raises(self) -> None:
        df = _make_feature_df([100.0, 101.0])
        config = _make_config(horizon=2)

        with pytest.raises(ValueError, match="empty"):
            add_target_labels(df, config)

