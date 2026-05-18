"""Tests for model training and validation-based selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

from src.config.model import TrainingConfig
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import train_and_select_model
from src.splitting.chronological import ChronologicalSplit


_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


class SignClassifier(BaseEstimator, ClassifierMixin):
    """Predict class 1 when the first feature is positive, else -1."""

    def fit(self, X: object, y: object) -> "SignClassifier":
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: object) -> np.ndarray:
        values = np.asarray(X)
        return np.where(values[:, 0] > 0, 1, -1)


def _make_config(
    *,
    metric: str = "accuracy",
    candidates: list[dict] | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        trading_symbol="ETH",
        signal_symbols=["ETH"],
        timeframe="1h",
        start_date="2022-01-01",
        end_date="2024-01-01",
        model_selection_metric=metric,
        model_candidates=candidates
        or [
            {
                "name": "bad_constant",
                "model_type": "BadConstant",
                "model_params": {},
            },
            {
                "name": "sign",
                "model_type": "SignClassifier",
                "model_params": {},
            },
        ],
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
    )


def _make_split() -> ChronologicalSplit:
    train = _frame([-2.0, -1.0, 1.0, 2.0])
    validation = _frame([-3.0, -0.5, 0.25, 4.0])
    test = _frame([-10.0, 10.0])
    return ChronologicalSplit(train=train, validation=validation, test=test)


def _frame(signal_values: list[float]) -> pd.DataFrame:
    labels = [-1 if value <= 0 else 1 for value in signal_values]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2022-01-01", periods=len(signal_values), freq="1h", tz="UTC"
            ),
            "signal": signal_values,
            "ETH_close": [100.0 + index for index in range(len(signal_values))],
            TARGET_RETURN_COLUMN: [-label * 0.10 for label in labels],
            TARGET_LABEL_COLUMN: labels,
        }
    )


@pytest.fixture
def _patch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build_estimator(model_type: str, model_params: dict) -> object:
        if model_type == "BadConstant":
            return DummyClassifier(strategy="constant", constant=-1)
        if model_type == "SignClassifier":
            return SignClassifier()
        raise AssertionError(f"Unexpected model_type: {model_type}")

    monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)


class TestTrainAndSelectModel:
    def test_trains_all_candidates_and_selects_best_validation_metric(
        self, _patch_registry: None
    ) -> None:
        config = _make_config(metric="accuracy")

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate.name == "sign"
        assert result.best_candidate.validation_metrics["accuracy"] == 1.0
        assert [candidate.name for candidate in result.candidates] == [
            "bad_constant",
            "sign",
        ]

    def test_feature_columns_exclude_label_and_future_return(
        self, _patch_registry: None
    ) -> None:
        config = _make_config(metric="accuracy")

        result = train_and_select_model(_make_split(), config)

        assert TARGET_LABEL_COLUMN not in result.feature_columns
        assert TARGET_RETURN_COLUMN not in result.feature_columns
        assert "timestamp" not in result.feature_columns
        assert "signal" in result.feature_columns

    def test_single_legacy_model_config_is_supported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            assert model_type == "SignClassifier"
            assert model_params == {"unused": 1}
            return SignClassifier()

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        config = TrainingConfig(
            trading_symbol="ETH",
            signal_symbols=["ETH"],
            timeframe="1h",
            start_date="2022-01-01",
            end_date="2024-01-01",
            model_type="SignClassifier",
            model_params={"unused": 1},
            model_selection_metric="accuracy",
            data_api_key=_API_KEY,
            data_api_secret=_API_SECRET,
        )

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate.name == "SignClassifier"
        assert len(result.candidates) == 1

    def test_no_numeric_feature_columns_raises(self, _patch_registry: None) -> None:
        split = _make_split()
        without_features = ChronologicalSplit(
            train=split.train.drop(columns=["signal", "ETH_close"]),
            validation=split.validation.drop(columns=["signal", "ETH_close"]),
            test=split.test.drop(columns=["signal", "ETH_close"]),
        )

        with pytest.raises(ValueError, match="No numeric feature columns"):
            train_and_select_model(without_features, _make_config())
