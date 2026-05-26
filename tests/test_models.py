"""Tests for model training and validation-based selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config.model import TrainingConfig
from src.evaluation.metrics import RETURN_OVER_DRAWDOWN_METRIC
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


class FixedPredictionClassifier(BaseEstimator, ClassifierMixin):
    """Return a fixed prediction vector for deterministic selection tests."""

    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def fit(self, X: object, y: object) -> "FixedPredictionClassifier":
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(self.predictions[: len(X)])


class CapturingClassifier(BaseEstimator, ClassifierMixin):
    """Capture transformed training features passed by the sklearn pipeline."""

    def __init__(self, captured: dict[str, np.ndarray]) -> None:
        self.captured = captured

    def fit(self, X: object, y: object) -> "CapturingClassifier":
        self.captured["fit_X"] = np.asarray(X, dtype=float)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X: object) -> np.ndarray:
        return np.zeros(len(np.asarray(X)), dtype=int)


def _make_config(
    *,
    metric: str = "accuracy",
    candidates: list[dict] | None = None,
) -> TrainingConfig:
    return TrainingConfig(
        trading_symbol="ETH",
        signal_symbols=["ETH"],
        timeframe="1h",
        start_date=-365,
        end_date=-1,
        model_selection_metric=metric,
        model_candidates=candidates
        or [
            {
                "name": "f1_winner",
                "model_type": "F1Winner",
                "model_params": {},
            },
            {
                "name": "return_winner",
                "model_type": "ReturnWinner",
                "model_params": {},
            },
        ],
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
    )


def _make_split() -> ChronologicalSplit:
    train = _frame([-2.0, -1.0, 1.0, 2.0])
    validation = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2022-01-01", periods=4, freq="1h", tz="UTC"
            ),
            "signal": [-3.0, -0.5, 0.25, 4.0],
            "ETH_close": [100.0, 101.0, 102.0, 103.0],
            TARGET_RETURN_COLUMN: [0.10, 0.10, -0.10, 0.10],
            TARGET_LABEL_COLUMN: [-1, 1, -1, 1],
        }
    )
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
        if model_type == "F1Winner":
            return FixedPredictionClassifier([-1, 1, -1, 1])
        if model_type == "ReturnWinner":
            return FixedPredictionClassifier([1, -1, 1, -1])
        if model_type == "SignClassifier":
            return SignClassifier()
        raise AssertionError(f"Unexpected model_type: {model_type}")

    monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)


class TestTrainAndSelectModel:
    def test_trains_all_candidates_and_selects_best_validation_backtest_score(
        self, _patch_registry: None
    ) -> None:
        config = _make_config(metric="f1_macro")

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate.name == "return_winner"
        assert result.selection_metric == RETURN_OVER_DRAWDOWN_METRIC
        assert result.best_candidate.validation_score > 0
        assert result.candidates[0].validation_metrics["f1_macro"] == 1.0
        assert result.candidates[0].validation_score < 0
        assert [candidate.name for candidate in result.candidates] == [
            "f1_winner",
            "return_winner",
        ]

    def test_zero_cumulative_return_gets_negative_validation_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            return FixedPredictionClassifier([0, 0, 0, 0])

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        config = _make_config(
            candidates=[
                {
                    "name": "flat",
                    "model_type": "Flat",
                    "model_params": {},
                }
            ]
        )

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate.validation_backtest_metrics[
            "cumulative_return"
        ] == 0.0
        assert result.best_candidate.validation_score < 0

    def test_invalid_validation_backtest_gets_negative_infinity_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            return FixedPredictionClassifier([1, -1, 1, -1])

        def fake_backtest_metrics(**_: object) -> dict[str, float]:
            return {
                "cumulative_return": float("nan"),
                "max_drawdown": 0.0,
                "turnover": 1.0,
            }

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        monkeypatch.setattr("src.models.trainer.backtest_metrics", fake_backtest_metrics)
        config = _make_config(
            candidates=[
                {
                    "name": "bad_backtest",
                    "model_type": "BadBacktest",
                    "model_params": {},
                }
            ]
        )

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate is None
        assert result.model_selection_status == "no_eligible_model"
        assert result.eligible_candidate_count == 0
        assert result.rejected_candidate_count == 1
        assert result.candidates[0].validation_score == float("-inf")
        assert result.candidates[0].rejection_reason == "validation_backtest_non_finite"

    @pytest.mark.parametrize(
        ("metrics", "expected_reason"),
        [
            (
                {
                    "bars": 100,
                    "traded_bars": 10,
                    "cumulative_return": 0.50,
                    "max_drawdown": -0.86,
                    "turnover": 10.0,
                },
                "validation_max_drawdown_below_-0.85",
            ),
            (
                {
                    "bars": 100,
                    "traded_bars": 10,
                    "cumulative_return": 0.50,
                    "max_drawdown": -0.10,
                    "turnover": 251.0,
                },
                "validation_turnover_above_250",
            ),
        ],
    )
    def test_validation_risk_filters_reject_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        metrics: dict[str, float],
        expected_reason: str,
    ) -> None:
        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            return FixedPredictionClassifier([1, -1, 1, -1])

        def fake_backtest_metrics(**_: object) -> dict[str, float]:
            return metrics

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        monkeypatch.setattr("src.models.trainer.backtest_metrics", fake_backtest_metrics)
        config = _make_config(
            candidates=[
                {
                    "name": "rejected",
                    "model_type": "Rejected",
                    "model_params": {},
                }
            ]
        )

        result = train_and_select_model(_make_split(), config)

        assert result.selection_metric == RETURN_OVER_DRAWDOWN_METRIC
        assert result.best_candidate is None
        assert result.model_selection_status == "no_eligible_model"
        assert result.eligible_candidate_count == 0
        assert result.rejected_candidate_count == 1
        assert result.candidates[0].validation_score == float("-inf")
        assert result.candidates[0].rejection_reason == expected_reason
        assert result.candidates[0].validation_backtest_metrics is metrics

    def test_rejected_candidates_are_never_selected_when_finite_candidate_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            if model_type == "Rejected":
                return FixedPredictionClassifier([1, 1, 1, 1])
            if model_type == "Eligible":
                return FixedPredictionClassifier([0, 1, 0, 1])
            raise AssertionError(f"Unexpected model_type: {model_type}")

        backtests = {
            "rejected": {
                "bars": 100,
                "traded_bars": 10,
                "cumulative_return": 10.0,
                "max_drawdown": -0.86,
                "turnover": 10.0,
            },
            "eligible": {
                "bars": 100,
                "traded_bars": 10,
                "cumulative_return": 0.20,
                "max_drawdown": -0.10,
                "turnover": 10.0,
            },
        }
        calls = iter([backtests["rejected"], backtests["eligible"]])

        def fake_backtest_metrics(**_: object) -> dict[str, float]:
            return next(calls)

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        monkeypatch.setattr("src.models.trainer.backtest_metrics", fake_backtest_metrics)
        config = _make_config(
            candidates=[
                {"name": "rejected", "model_type": "Rejected", "model_params": {}},
                {"name": "eligible", "model_type": "Eligible", "model_params": {}},
            ]
        )

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate is not None
        assert result.best_candidate.name == "eligible"
        assert result.best_candidate.rejection_reason is None
        assert result.best_candidate.validation_score == pytest.approx(2.0)
        assert result.eligible_candidate_count == 1
        assert result.rejected_candidate_count == 1
        assert result.candidates[0].rejection_reason == "validation_max_drawdown_below_-0.85"

    def test_all_rejected_candidates_produce_no_selected_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            return FixedPredictionClassifier([1, 1, 1, 1])

        def fake_backtest_metrics(**_: object) -> dict[str, float]:
            return {
                "bars": 100,
                "traded_bars": 10,
                "cumulative_return": 0.50,
                "max_drawdown": -0.86,
                "turnover": 10.0,
            }

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        monkeypatch.setattr("src.models.trainer.backtest_metrics", fake_backtest_metrics)
        config = _make_config(
            candidates=[
                {"name": "rejected_a", "model_type": "RejectedA", "model_params": {}},
                {"name": "rejected_b", "model_type": "RejectedB", "model_params": {}},
            ]
        )

        result = train_and_select_model(_make_split(), config)

        assert result.best_candidate is None
        assert result.model_selection_status == "no_eligible_model"
        assert result.eligible_candidate_count == 0
        assert result.rejected_candidate_count == 2
        assert all(candidate.rejection_reason is not None for candidate in result.candidates)

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
            start_date=-365,
            end_date=-1,
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

    def test_model_selection_does_not_change_when_test_returns_change(
        self, _patch_registry: None
    ) -> None:
        split = _make_split()
        changed_test = split.test.copy()
        changed_test[TARGET_RETURN_COLUMN] = [999.0, -999.0]
        changed_test["ETH_close"] = [1_000_000.0, 1.0]
        split_with_changed_test = ChronologicalSplit(
            train=split.train,
            validation=split.validation,
            test=changed_test,
        )
        config = _make_config(metric="f1_macro")

        result = train_and_select_model(split, config)
        changed_result = train_and_select_model(split_with_changed_test, config)

        assert changed_result.best_candidate.name == result.best_candidate.name
        assert changed_result.best_candidate.validation_score == pytest.approx(
            result.best_candidate.validation_score
        )

    def test_feature_selection_is_determined_from_train_only(
        self, _patch_registry: None
    ) -> None:
        split = _make_split()
        validation = split.validation.copy()
        test = split.test.copy()
        validation["validation_only_feature"] = [1.0, 2.0, 3.0, 4.0]
        test["validation_only_feature"] = [5.0, 6.0]

        result = train_and_select_model(
            ChronologicalSplit(train=split.train, validation=validation, test=test),
            _make_config(metric="f1_macro"),
        )

        assert "validation_only_feature" not in result.feature_columns

    def test_imputer_and_scaler_are_fit_on_train_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, np.ndarray] = {}

        def fake_build_estimator(model_type: str, model_params: dict) -> object:
            assert model_type == "LogisticRegression"
            return CapturingClassifier(captured)

        monkeypatch.setattr("src.models.trainer.build_estimator", fake_build_estimator)
        train = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2022-01-01", periods=4, freq="1h", tz="UTC"
                ),
                "signal": [0.0, 2.0, np.nan, 4.0],
                TARGET_RETURN_COLUMN: [0.01, 0.01, -0.01, -0.01],
                TARGET_LABEL_COLUMN: [1, 1, -1, -1],
            }
        )
        validation = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2022-01-02", periods=2, freq="1h", tz="UTC"
                ),
                "signal": [1_000.0, np.nan],
                TARGET_RETURN_COLUMN: [0.01, -0.01],
                TARGET_LABEL_COLUMN: [1, -1],
            }
        )
        test = validation.copy()
        config = _make_config(
            candidates=[
                {
                    "name": "capture",
                    "model_type": "LogisticRegression",
                    "model_params": {},
                }
            ]
        )

        train_and_select_model(
            ChronologicalSplit(train=train, validation=validation, test=test),
            config,
        )

        assert captured["fit_X"].ravel().tolist() == pytest.approx(
            [-1.41421356, 0.0, 0.0, 1.41421356]
        )
