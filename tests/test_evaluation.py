"""Tests for model evaluation reports and simple backtest metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src.config.model import TrainingConfig
from src.evaluation.metrics import RETURN_OVER_DRAWDOWN_METRIC, backtest_metrics
from src.evaluation.metrics import probability_policy_backtest
from src.evaluation.metrics import signals_to_target_positions
from src.evaluation import evaluate_and_save_report, evaluate_model
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import CandidateTrainingResult, ModelSelectionResult
from src.splitting.chronological import ChronologicalSplit


_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


class FixedPredictionClassifier(BaseEstimator, ClassifierMixin):
    """Estimator that returns fixed binary predictions/probabilities."""

    def __init__(self, predictions: list[int]) -> None:
        self.predictions = predictions
        self.classes_ = np.asarray([0, 1])

    def predict(self, X: object) -> np.ndarray:
        return np.asarray(self.predictions[: len(X)])

    def predict_proba(self, X: object) -> np.ndarray:
        positive = np.asarray([1.0 if value == 1 else 0.0 for value in self.predict(X)])
        return np.column_stack([1.0 - positive, positive])


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
    return ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        raw_row_counts={"train": 5, "validation": 5, "test": 7},
    )


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
        validation_backtest_metrics={
            "cumulative_return": 0.12,
            "max_drawdown": -0.03,
            "turnover": 2.0,
            "traded_bars": 2,
            "exposure_ratio": 0.40,
        },
        entry_threshold=0.5,
        exit_threshold=0.4,
        min_hold_bars=0,
        return_buffer=0.005,
        validation_score=4.0,
        return_over_drawdown=4.0,
        rejection_reasons=[],
    )
    return ModelSelectionResult(
        best_candidate=candidate,
        candidates=[candidate],
        feature_columns=["signal", "ETH_close"],
        selection_metric=RETURN_OVER_DRAWDOWN_METRIC,
    )


class TestEvaluateModel:
    def test_report_includes_ml_metrics_and_simple_backtest(self) -> None:
        selection = _make_selection([1, 0, -1, 1])

        report = evaluate_model(selection, _make_split(), _make_config())

        assert report["ml_metrics"]["accuracy"] == 0.5
        assert report["ml_metrics"]["confusion_matrix"]["labels"] == [0, 1]
        assert report["ml_metrics"]["confusion_matrix"]["matrix"] == [
            [1, 1],
            [1, 1],
        ]
        assert report["backtest_metrics"]["transaction_fee"] == 0.01
        assert report["backtest_metrics"]["traded_bars"] == 1
        assert report["backtest_metrics"]["exposure_ratio"] == pytest.approx(0.25)
        assert report["backtest_metrics"]["hit_rate"] == pytest.approx(0.0)
        assert report["backtest_metrics"]["turnover"] == 2.0
        assert report["backtest_metrics"]["strategy_return_sum"] == pytest.approx(-0.07)
        assert report["backtest_metrics"]["cumulative_return"] == pytest.approx(
            (1.0 * 0.94 * 0.99 * 1.0) - 1.0
        )
        assert report["backtest_metrics"]["execution_lag_bars"] == 1
        assert report["backtest_metrics"]["executed_position_min"] == 0.0
        assert report["backtest_metrics"]["executed_position_max"] == 1.0
        assert report["backtest_metrics"]["executed_positions_are_long_cash"] is True
        selected_model = report["run_metadata"]["selected_model"]
        assert selected_model["name"] == "fixed"
        assert selected_model["selection_metric"] == RETURN_OVER_DRAWDOWN_METRIC
        assert selected_model["validation_metric_value"] == 4.0
        assert selected_model["validation_cumulative_return"] == 0.12
        assert selected_model["validation_max_drawdown"] == -0.03
        assert selected_model["validation_turnover"] == 2.0
        assert report["run_metadata"]["execution_lag_bars"] == 1
        assert report["run_metadata"]["portfolio_mode"] == "all_in_long_cash"
        assert report["run_metadata"]["sell_mode"] == "cash"
        assert report["run_metadata"]["shorting_enabled"] is False
        assert report["run_metadata"]["leverage"] == 1
        assert report["run_metadata"]["hold_behavior"] == "keep_previous_position"
        assert report["run_metadata"]["initial_position"] == 0
        assert report["run_metadata"]["max_validation_drawdown_filter"] == -0.85
        assert report["run_metadata"]["max_validation_turnover_filter"] == 250.0
        assert report["run_metadata"]["min_validation_cumulative_return"] == 0.10
        assert report["run_metadata"]["min_validation_exposure_ratio"] == 0.10
        assert report["run_metadata"]["min_validation_traded_bars"] == 100
        assert report["run_metadata"]["label_generation"] == "split_local"
        assert report["run_metadata"]["train_rows_raw"] == 5
        assert report["run_metadata"]["train_rows_labelled"] == 2
        assert report["run_metadata"]["validation_rows_raw"] == 5
        assert report["run_metadata"]["validation_rows_labelled"] == 2
        assert report["run_metadata"]["test_rows_raw"] == 7
        assert report["run_metadata"]["test_rows_labelled"] == 4

        candidate = report["validation_candidates"][0]
        assert candidate["validation_score"] == 4.0
        assert candidate["validation_cumulative_return"] == 0.12
        assert candidate["validation_max_drawdown"] == -0.03
        assert candidate["validation_turnover"] == 2.0
        assert candidate["validation_traded_bars"] == 2
        assert candidate["validation_backtest_metrics"]["traded_bars"] == 2
        assert candidate["rejection_reasons"] == []
        assert report["candidate_summary"][0]["eligible"] is True
        assert report["candidate_summary"][0]["validation_exposure_ratio"] == 0.40
        assert report["candidate_summary"][0]["rejection_reasons"] == []

    def test_report_includes_rejection_reason_for_rejected_candidate(self) -> None:
        selection = _make_selection([1, 0, -1, 1])
        rejected = CandidateTrainingResult(
            name="rejected",
            model_type="FixedPredictionClassifier",
            model_params={},
            estimator=FixedPredictionClassifier([1, 1, 1, 1]),
            validation_metrics={"accuracy": 0.0},
            validation_backtest_metrics={
                "bars": 100,
                "traded_bars": 10,
                "exposure_ratio": 0.10,
                "cumulative_return": 0.50,
                "max_drawdown": -0.86,
                "turnover": 10.0,
            },
            entry_threshold=0.5,
            exit_threshold=0.4,
            min_hold_bars=0,
            return_buffer=0.005,
            validation_score=float("-inf"),
            return_over_drawdown=5.0,
            rejection_reasons=["validation_max_drawdown_below_limit"],
        )
        selection = ModelSelectionResult(
            best_candidate=selection.best_candidate,
            candidates=[selection.best_candidate, rejected],
            feature_columns=selection.feature_columns,
            selection_metric=selection.selection_metric,
        )

        report = evaluate_model(selection, _make_split(), _make_config())

        rejected_payload = report["validation_candidates"][1]
        assert rejected_payload["validation_score"] == float("-inf")
        assert rejected_payload["validation_cumulative_return"] == 0.50
        assert rejected_payload["validation_max_drawdown"] == -0.86
        assert rejected_payload["validation_turnover"] == 10.0
        assert (
            rejected_payload["rejection_reasons"]
            == ["validation_max_drawdown_below_limit"]
        )
        assert report["candidate_summary"][1]["eligible"] is False

    def test_no_eligible_model_uses_cash_baseline_and_no_selected_model(self) -> None:
        rejected = CandidateTrainingResult(
            name="rejected",
            model_type="FixedPredictionClassifier",
            model_params={},
            estimator=FixedPredictionClassifier([1, 1, 1, 1]),
            validation_metrics={"accuracy": 0.0},
            validation_backtest_metrics={
                "bars": 100,
                "traded_bars": 10,
                "exposure_ratio": 0.10,
                "cumulative_return": 0.50,
                "max_drawdown": -0.86,
                "turnover": 10.0,
            },
            entry_threshold=0.5,
            exit_threshold=0.4,
            min_hold_bars=0,
            return_buffer=0.005,
            validation_score=float("-inf"),
            return_over_drawdown=5.0,
            rejection_reasons=["validation_max_drawdown_below_limit"],
        )
        selection = ModelSelectionResult(
            best_candidate=None,
            candidates=[rejected],
            feature_columns=["signal", "ETH_close"],
            selection_metric=RETURN_OVER_DRAWDOWN_METRIC,
        )

        report = evaluate_model(selection, _make_split(), _make_config())

        metadata = report["run_metadata"]
        assert metadata["selected_model"] is None
        assert metadata["model_selection_status"] == "no_eligible_model"
        assert metadata["trading_enabled"] is False
        assert (
            metadata["no_trade_reason"]
            == "No candidate passed binary long/cash validation filters"
        )
        assert metadata["eligible_candidate_count"] == 0
        assert metadata["rejected_candidate_count"] == 1
        assert metadata["risk_filters_applied"] is True
        assert report["ml_metrics"] is None
        assert report["backtest_metrics"]["baseline"] == "cash"
        assert report["backtest_metrics"]["cumulative_return"] == 0.0
        assert report["backtest_metrics"]["max_drawdown"] == 0.0
        assert report["backtest_metrics"]["traded_bars"] == 0
        assert report["backtest_metrics"]["turnover"] == 0.0

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


class TestBacktestExecutionTiming:
    def test_signal_executes_on_next_bar_not_same_bar(self) -> None:
        metrics = backtest_metrics(
            predictions=[1, 0, 0],
            future_returns=pd.Series([1.0, -1.0, 0.0]),
            transaction_fee=0.0,
        )

        assert metrics["execution_lag_bars"] == 1
        assert metrics["traded_bars"] == 2
        assert metrics["cumulative_return"] == pytest.approx(-1.0)

    def test_fees_apply_to_executed_position_changes(self) -> None:
        metrics = backtest_metrics(
            predictions=[1, 1, -1, -1],
            future_returns=pd.Series([0.0, 0.0, 0.0, 0.0]),
            transaction_fee=0.01,
        )

        assert metrics["turnover"] == 2.0
        assert metrics["strategy_return_sum"] == pytest.approx(-0.02)
        assert metrics["cumulative_return"] == pytest.approx((0.99 * 1.0 * 0.99) - 1.0)

    def test_buy_buy_buy_enters_once_then_holds(self) -> None:
        metrics = backtest_metrics(
            predictions=[1, 1, 1, 1],
            future_returns=pd.Series([0.0, 0.0, 0.0, 0.0]),
            transaction_fee=0.01,
        )

        assert metrics["turnover"] == 1.0
        assert metrics["traded_bars"] == 3
        assert metrics["executed_positions_are_long_cash"] is True

    def test_sell_sell_sell_exits_once_then_stays_cash(self) -> None:
        metrics = backtest_metrics(
            predictions=[-1, -1, -1, -1],
            future_returns=pd.Series([0.0, 0.0, 0.0, 0.0]),
            transaction_fee=0.01,
            initial_position=1,
        )

        assert metrics["turnover"] == 1.0
        assert metrics["traded_bars"] == 1
        assert metrics["executed_position_min"] == 0.0
        assert metrics["executed_position_max"] == 1.0

    def test_hold_preserves_previous_target_position(self) -> None:
        target_positions = signals_to_target_positions(
            predictions=[1, 0, 0, -1, 0, 1],
            index=pd.RangeIndex(6),
            initial_position=0,
        )

        assert target_positions.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 1.0]

    def test_no_negative_positions_are_possible(self) -> None:
        metrics = backtest_metrics(
            predictions=[1, -1, 0, 1, -1],
            future_returns=pd.Series([0.0] * 5),
            transaction_fee=0.0,
        )

        assert metrics["executed_position_min"] == 0.0
        assert metrics["executed_position_max"] == 1.0
        assert metrics["executed_positions_are_long_cash"] is True

    def test_probability_policy_uses_hysteresis_and_min_hold(self) -> None:
        metrics = probability_policy_backtest(
            probabilities=[0.80, 0.30, 0.30, 0.30, 0.30, 0.30],
            future_returns=pd.Series([0.0] * 6),
            transaction_fee=0.01,
            entry_threshold=0.60,
            exit_threshold=0.40,
            min_hold_bars=3,
        )

        assert metrics["entry_signals"] == 1
        assert metrics["exit_signals"] == 1
        assert metrics["turnover"] == 2.0
        assert metrics["executed_position_min"] == 0.0
        assert metrics["executed_position_max"] == 1.0
