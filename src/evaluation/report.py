"""Evaluate a selected model and persist a JSON report."""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score

from src.config.model import TrainingConfig
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import ModelSelectionResult
from src.splitting.chronological import ChronologicalSplit

logger = logging.getLogger(__name__)

CLASS_LABELS = [-1, 0, 1]


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
        "ml_metrics": _classification_metrics(y_true, predictions),
        "backtest_metrics": _backtest_metrics(
            predictions=predictions,
            future_returns=future_returns,
            transaction_fee=config.backtest.transaction_fee,
        ),
        "validation_candidates": [
            {
                "name": candidate.name,
                "model_type": candidate.model_type,
                "validation_metrics": candidate.validation_metrics,
            }
            for candidate in model_selection.candidates
        ],
    }


def _classification_metrics(
    y_true: pd.Series,
    predictions: object,
) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, predictions, labels=CLASS_LABELS)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_macro": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": matrix.astype(int).tolist(),
        },
        "support": {
            str(label): int((y_true == label).sum())
            for label in CLASS_LABELS
        },
    }


def _backtest_metrics(
    *,
    predictions: object,
    future_returns: pd.Series,
    transaction_fee: float,
) -> dict[str, Any]:
    positions = pd.Series(
        np.asarray(predictions, dtype=float),
        index=future_returns.index,
    )
    realized_returns = future_returns.astype(float)
    gross_strategy_returns = positions * realized_returns
    turnover = positions.diff().abs().fillna(positions.abs())
    net_strategy_returns = gross_strategy_returns - (turnover * transaction_fee)
    equity_curve = (1.0 + net_strategy_returns).cumprod()

    traded = positions != 0
    hit_rate = 0.0
    if traded.any():
        hit_rate = float((gross_strategy_returns[traded] > 0.0).mean())

    return {
        "transaction_fee": float(transaction_fee),
        "bars": int(len(net_strategy_returns)),
        "traded_bars": int(traded.sum()),
        "mean_strategy_return": float(net_strategy_returns.mean()),
        "strategy_return_sum": float(net_strategy_returns.sum()),
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "benchmark_cumulative_return": float((1.0 + realized_returns).prod() - 1.0),
        "hit_rate": hit_rate,
        "max_drawdown": _max_drawdown(equity_curve),
        "turnover": float(turnover.sum()),
    }


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return float(drawdown.min())


def _run_metadata(
    data_split: ChronologicalSplit,
    config: TrainingConfig,
    model_selection: ModelSelectionResult,
) -> dict[str, Any]:
    test = data_split.test
    return {
        "created_at_utc": _utc_now().isoformat(),
        "trading_symbol": config.trading_symbol,
        "signal_symbols": config.signal_symbols,
        "timeframe": config.timeframe,
        "prediction_horizon_bars": config.prediction_horizon_bars,
        "return_threshold": config.return_threshold,
        "selected_model": {
            "name": model_selection.best_candidate.name,
            "model_type": model_selection.best_candidate.model_type,
            "selection_metric": model_selection.selection_metric,
            "validation_metric_value": model_selection.best_candidate.validation_metrics[
                model_selection.selection_metric
            ],
        },
        "feature_count": len(model_selection.feature_columns),
        "split_row_counts": data_split.row_counts,
        "test_period": {
            "start": _timestamp_to_iso(test["timestamp"].iloc[0]),
            "end": _timestamp_to_iso(test["timestamp"].iloc[-1]),
        },
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
