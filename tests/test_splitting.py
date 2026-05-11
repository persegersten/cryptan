"""Tests for chronological train/validation/test splitting (Step 5 of the MVP)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config.model import TrainingConfig
from src.labels.target import TARGET_LABEL_COLUMN
from src.splitting.chronological import ChronologicalSplit, split_chronologically


_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


def _make_config(
    split: dict[str, float] | None = None,
) -> TrainingConfig:
    """Build a minimal TrainingConfig for split tests."""
    return TrainingConfig(
        trading_symbol="ETH",
        signal_symbols=["ETH", "BNB", "SOL"],
        timeframe="1h",
        start_date="2022-01-01",
        end_date="2024-01-01",
        split=split or {"train": 0.70, "validation": 0.15, "test": 0.15},
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
    )


def _make_labelled_df(
    rows: int = 20,
    start: str = "2022-01-01",
) -> pd.DataFrame:
    """Build a minimal labelled DataFrame with deterministic hourly timestamps."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start=start, periods=rows, freq="1h", tz="UTC"),
            "ETH_close": [100.0 + i for i in range(rows)],
            "ETH_return_1": [0.01] * rows,
            TARGET_LABEL_COLUMN: [(-1, 0, 1)[i % 3] for i in range(rows)],
        }
    )


class TestSplitChronologicallyCorrectness:
    def test_splits_rows_by_configured_fractions(self) -> None:
        df = _make_labelled_df(rows=20)
        config = _make_config(split={"train": 0.60, "validation": 0.20, "test": 0.20})

        result = split_chronologically(df, config)

        assert isinstance(result, ChronologicalSplit)
        assert result.row_counts == {"train": 12, "validation": 4, "test": 4}

    def test_default_split_uses_config_model_defaults(self) -> None:
        df = _make_labelled_df(rows=20)
        config = _make_config()

        result = split_chronologically(df, config)

        assert result.row_counts == {"train": 14, "validation": 3, "test": 3}

    def test_partitions_are_contiguous_without_overlap(self) -> None:
        df = _make_labelled_df(rows=10)
        config = _make_config(split={"train": 0.60, "validation": 0.20, "test": 0.20})

        result = split_chronologically(df, config)

        assert result.train["timestamp"].max() < result.validation["timestamp"].min()
        assert result.validation["timestamp"].max() < result.test["timestamp"].min()
        combined = pd.concat([result.train, result.validation, result.test])
        assert combined["timestamp"].tolist() == df["timestamp"].tolist()

    def test_floor_boundaries_assign_remainder_to_test(self) -> None:
        df = _make_labelled_df(rows=11)
        config = _make_config(split={"train": 0.60, "validation": 0.20, "test": 0.20})

        result = split_chronologically(df, config)

        assert result.row_counts == {"train": 6, "validation": 2, "test": 3}


class TestSplitChronologicallyTimeSeriesSafety:
    def test_unsorted_input_is_sorted_before_split(self) -> None:
        df = _make_labelled_df(rows=10)
        unsorted = df.iloc[[5, 0, 9, 1, 2, 3, 4, 6, 7, 8]].reset_index(drop=True)
        config = _make_config(split={"train": 0.60, "validation": 0.20, "test": 0.20})

        result = split_chronologically(unsorted, config)

        assert result.train["timestamp"].is_monotonic_increasing
        assert result.validation["timestamp"].is_monotonic_increasing
        assert result.test["timestamp"].is_monotonic_increasing
        assert result.train["timestamp"].iloc[0] == df["timestamp"].iloc[0]

    def test_original_dataframe_not_mutated(self) -> None:
        df = _make_labelled_df(rows=10)
        original = df.copy(deep=True)
        config = _make_config(split={"train": 0.60, "validation": 0.20, "test": 0.20})

        split_chronologically(df, config)

        pd.testing.assert_frame_equal(df, original)

    def test_duplicate_timestamps_raise(self) -> None:
        df = _make_labelled_df(rows=10)
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
        config = _make_config(split={"train": 0.60, "validation": 0.20, "test": 0.20})

        with pytest.raises(ValueError, match="duplicate timestamps"):
            split_chronologically(df, config)


class TestSplitChronologicallyErrorGuards:
    def test_empty_dataframe_raises(self) -> None:
        df = pd.DataFrame(columns=["timestamp", TARGET_LABEL_COLUMN])
        config = _make_config()

        with pytest.raises(ValueError, match="empty"):
            split_chronologically(df, config)

    def test_missing_timestamp_column_raises(self) -> None:
        df = _make_labelled_df(rows=10).drop(columns=["timestamp"])
        config = _make_config()

        with pytest.raises(ValueError, match="timestamp"):
            split_chronologically(df, config)

    def test_missing_target_column_raises(self) -> None:
        df = _make_labelled_df(rows=10).drop(columns=[TARGET_LABEL_COLUMN])
        config = _make_config()

        with pytest.raises(ValueError, match=TARGET_LABEL_COLUMN):
            split_chronologically(df, config)

    def test_null_timestamp_raises(self) -> None:
        df = _make_labelled_df(rows=10)
        df.loc[0, "timestamp"] = pd.NaT
        config = _make_config()

        with pytest.raises(ValueError, match="null timestamps"):
            split_chronologically(df, config)

    def test_too_few_rows_for_non_empty_partitions_raises(self) -> None:
        df = _make_labelled_df(rows=3)
        config = _make_config()

        with pytest.raises(ValueError, match="empty partition"):
            split_chronologically(df, config)

