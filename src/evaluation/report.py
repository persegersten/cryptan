"""Evaluate a selected model and persist a JSON report."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.model import TrainingConfig
from src.evaluation.metrics import EXECUTION_LAG_BARS
from src.evaluation.metrics import backtest_metrics, classification_metrics
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import ModelSelectionResult
from src.splitting.chronological import ChronologicalSplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationArtifact:
    """Saved evaluation report details."""

    report: dict[str, Any]
    run_dir: Path
    report_path: Path


def evaluate_and_save_report(
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> EvaluationArtifact:
    """Evaluate the selected model on test data and save a JSON report."""
    report = evaluate_model(model_selection, data_split, config)
    run_dir = _create_run_dir(config.artifacts_dir)
    report_path = run_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Saved evaluation report: %s", report_path)
    return EvaluationArtifact(report=report, run_dir=run_dir, report_path=report_path)


def evaluate_model(
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable ML and backtest report for the test partition."""
    if data_split.test.empty:
        raise ValueError("Test partition is empty; cannot evaluate selected model.")

    if model_selection.best_candidate is None:
        return {
            "run_metadata": _run_metadata(data_split, config, model_selection),
            "ml_metrics": None,
            "backtest_metrics": _cash_baseline_metrics(len(data_split.test)),
            "validation_candidates": [
                _candidate_report(candidate)
                for candidate in model_selection.candidates
            ],
        }

    missing_features = [
        column
        for column in model_selection.feature_columns
        if column not in data_split.test.columns
    ]
    if missing_features:
        raise ValueError(
            f"Test partition is missing feature columns: {missing_features}."
        )

    required_columns = {TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN, "timestamp"}
    missing_required = required_columns - set(data_split.test.columns)
    if missing_required:
        raise ValueError(
            f"Test partition is missing required evaluation columns: "
            f"{sorted(missing_required)}."
        )

    predictions = model_selection.estimator.predict(
        data_split.test[model_selection.feature_columns]
    )
    y_true = data_split.test[TARGET_LABEL_COLUMN]
    future_returns = data_split.test[TARGET_RETURN_COLUMN]

    return {
        "run_metadata": _run_metadata(data_split, config, model_selection),
        "ml_metrics": classification_metrics(y_true, predictions),
        "backtest_metrics": backtest_metrics(
            predictions=predictions,
            future_returns=future_returns,
            transaction_fee=config.backtest.transaction_fee,
        ),
        "validation_candidates": [
            _candidate_report(candidate)
            for candidate in model_selection.candidates
        ],
    }


def _candidate_report(candidate: object) -> dict[str, Any]:
    payload = {
        "name": candidate.name,
        "model_type": candidate.model_type,
        "validation_metrics": candidate.validation_metrics,
        "validation_backtest_metrics": candidate.validation_backtest_metrics,
        "validation_cumulative_return": candidate.validation_backtest_metrics[
            "cumulative_return"
        ],
        "validation_max_drawdown": candidate.validation_backtest_metrics[
            "max_drawdown"
        ],
        "validation_turnover": candidate.validation_backtest_metrics["turnover"],
        "validation_traded_bars": candidate.validation_backtest_metrics["traded_bars"],
        "validation_score": candidate.validation_score,
    }
    if candidate.rejection_reason is not None:
        payload["rejection_reason"] = candidate.rejection_reason
    return payload


def _run_metadata(
    data_split: ChronologicalSplit,
    config: TrainingConfig,
    model_selection: ModelSelectionResult,
) -> dict[str, Any]:
    test = data_split.test
    raw_counts = data_split.raw_row_counts or data_split.row_counts
    selected_model = None
    if model_selection.best_candidate is not None:
        selected_model = {
            "name": model_selection.best_candidate.name,
            "model_type": model_selection.best_candidate.model_type,
            "selection_metric": model_selection.selection_metric,
            "validation_metric_value": model_selection.best_candidate.validation_score,
            "validation_cumulative_return": (
                model_selection.best_candidate.validation_backtest_metrics[
                    "cumulative_return"
                ]
            ),
            "validation_max_drawdown": (
                model_selection.best_candidate.validation_backtest_metrics[
                    "max_drawdown"
                ]
            ),
            "validation_turnover": (
                model_selection.best_candidate.validation_backtest_metrics[
                    "turnover"
                ]
            ),
        }
    return {
        "created_at_utc": _utc_now().isoformat(),
        "trading_symbol": config.trading_symbol,
        "signal_symbols": config.signal_symbols,
        "timeframe": config.timeframe,
        "prediction_horizon_bars": config.prediction_horizon_bars,
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "label_generation": "split_local",
        "model_selection_status": model_selection.model_selection_status,
        "risk_filters_applied": True,
        "rejected_candidate_count": model_selection.rejected_candidate_count,
        "eligible_candidate_count": model_selection.eligible_candidate_count,
        "trading_enabled": model_selection.best_candidate is not None,
        "no_trade_reason": (
            None
            if model_selection.best_candidate is not None
            else "No candidate passed validation risk filters"
        ),
        "return_threshold": config.return_threshold,
        "selected_model": selected_model,
        "feature_count": len(model_selection.feature_columns),
        "split_row_counts": data_split.row_counts,
        "train_rows_raw": raw_counts["train"],
        "train_rows_labelled": len(data_split.train),
        "validation_rows_raw": raw_counts["validation"],
        "validation_rows_labelled": len(data_split.validation),
        "test_rows_raw": raw_counts["test"],
        "test_rows_labelled": len(data_split.test),
        "test_period": {
            "start": _timestamp_to_iso(test["timestamp"].iloc[0]),
            "end": _timestamp_to_iso(test["timestamp"].iloc[-1]),
        },
    }


def _cash_baseline_metrics(bars: int) -> dict[str, Any]:
    return {
        "transaction_fee": 0.0,
        "bars": int(bars),
        "traded_bars": 0,
        "mean_strategy_return": 0.0,
        "strategy_return_sum": 0.0,
        "cumulative_return": 0.0,
        "benchmark_cumulative_return": None,
        "hit_rate": 0.0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "baseline": "cash",
    }


def _create_run_dir(artifacts_dir: Path) -> Path:
    run_id = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = artifacts_dir / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)


def _timestamp_to_iso(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.isoformat()
