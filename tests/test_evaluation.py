"""Tests for model evaluation reports and simple backtest metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config.model import TrainingConfig
from src.evaluation import evaluate_and_save_report, evaluate_model
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import CandidateTrainingResult, ModelSelectionResult
from src.splitting.chronological import ChronologicalSplit


_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


class FixedPredictionClassifier(BaseEstimator, ClassifierMixin):
    """Estimator that returns a fixed prediction vector."""

    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(self.predictions[: len(X)])


def _make_config(tmp_path: Path | None = None) -> TrainingConfig:
    return TrainingConfig(
        trading_symbol="ETH",
        signal_symbols=["ETH", "BNB"],
        timeframe="1h",
        start_date=-365,
        end_date=-1,
        backtest={"transaction_fee": 0.01},
        artifacts_dir=tmp_path or Path("artifacts"),
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
    )


def _make_split() -> ChronologicalSplit:
    train = _frame([1.0, 2.0], [1, 1], [0.05, 0.02])
    validation = _frame([3.0, 4.0], [1, -1], [0.03, -0.04])
    test = _frame(
        [5.0, 6.0, 7.0, 8.0],
        [1, 0, 1, -1],
        [0.10, -0.05, 0.20, -0.10],
    )
    return ChronologicalSplit(train=train, validation=validation, test=test)


def _frame(
    signal_values: list[float],
    labels: list[int],
    future_returns: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2022-01-01", periods=len(signal_values), freq="1h", tz="UTC"
            ),
            "signal": signal_values,
            "ETH_close": [100.0 + index for index in range(len(signal_values))],
            TARGET_RETURN_COLUMN: future_returns,
            TARGET_LABEL_COLUMN: labels,
        }
    )


def _make_selection(predictions: list[int]) -> ModelSelectionResult:
    estimator = FixedPredictionClassifier(predictions)
    candidate = CandidateTrainingResult(
        name="fixed",
        model_type="FixedPredictionClassifier",
        model_params={},
        estimator=estimator,
        validation_metrics={
            "accuracy": 0.5,
            "precision_macro": 0.5,
            "recall_macro": 0.5,
            "f1_macro": 0.5,
        },
    )
    return ModelSelectionResult(
        best_candidate=candidate,
        candidates=[candidate],
        feature_columns=["signal", "ETH_close"],
        selection_metric="f1_macro",
    )


class TestEvaluateModel:
    def test_report_includes_ml_metrics_and_simple_backtest(self) -> None:
        selection = _make_selection([1, 0, -1, 1])

        report = evaluate_model(selection, _make_split(), _make_config())

        assert report["ml_metrics"]["accuracy"] == 0.5
        assert report["ml_metrics"]["confusion_matrix"]["labels"] == [-1, 0, 1]
        assert report["ml_metrics"]["confusion_matrix"]["matrix"] == [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]
        assert report["backtest_metrics"]["transaction_fee"] == 0.01
        assert report["backtest_metrics"]["traded_bars"] == 3
        assert report["backtest_metrics"]["hit_rate"] == pytest.approx(1 / 3)
        assert report["backtest_metrics"]["turnover"] == 5.0
        assert report["backtest_metrics"]["strategy_return_sum"] == pytest.approx(-0.25)
        assert report["backtest_metrics"]["cumulative_return"] == pytest.approx(
            (1.09 * 0.99 * 0.79 * 0.88) - 1.0
        )
        assert report["run_metadata"]["selected_model"]["name"] == "fixed"

    def test_missing_feature_column_raises_clear_error(self) -> None:
        split = _make_split()
        selection = _make_selection([1, 0, -1, 1])
        selection = ModelSelectionResult(
            best_candidate=selection.best_candidate,
            candidates=selection.candidates,
            feature_columns=["missing_feature"],
            selection_metric=selection.selection_metric,
        )

        with pytest.raises(ValueError, match="missing feature columns"):
            evaluate_model(selection, split, _make_config())


class TestEvaluateAndSaveReport:
    def test_saves_json_report_under_timestamped_artifact_directory(
        self, tmp_path: Path
    ) -> None:
        artifact = evaluate_and_save_report(
            _make_selection([1, 0, -1, 1]),
            _make_split(),
            _make_config(tmp_path),
        )

        assert artifact.report_path.name == "evaluation_report.json"
        assert artifact.report_path.exists()
        assert artifact.report_path.parent.parent == tmp_path

        saved = json.loads(artifact.report_path.read_text(encoding="utf-8"))
        assert saved["run_metadata"]["trading_symbol"] == "ETH"
        assert saved["backtest_metrics"]["bars"] == 4
