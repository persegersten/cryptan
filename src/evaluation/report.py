"""Evaluate a selected model and persist a JSON report."""

from __future__ import annotations

import datetime
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.model import TrainingConfig
from src.evaluation.metrics import EXECUTION_LAG_BARS, PORTFOLIO_MODE, SELL_MODE
from src.evaluation.metrics import backtest_metrics, classification_metrics
from src.evaluation.metrics import probability_policy_backtest
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import ModelSelectionResult, predict_long_probabilities
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
            "baselines": _baseline_reports(data_split, config),
            "validation_candidates": [
                _candidate_report(candidate)
                for candidate in model_selection.candidates
            ],
            "candidate_summary": [
                _candidate_summary(candidate)
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

    best = model_selection.best_candidate
    if best is None:
        raise ValueError("Expected a selected candidate after no-candidate branch.")

    probabilities = predict_long_probabilities(
        best.estimator,
        data_split.test[model_selection.feature_columns],
    )
    predictions = (probabilities >= 0.5).astype("int64")
    y_true = data_split.test[TARGET_LABEL_COLUMN]
    future_returns = data_split.test[TARGET_RETURN_COLUMN]

    return {
        "run_metadata": _run_metadata(data_split, config, model_selection),
        "ml_metrics": classification_metrics(y_true, predictions),
        "backtest_metrics": probability_policy_backtest(
            probabilities=probabilities,
            future_returns=future_returns,
            transaction_fee=config.backtest.transaction_fee,
            entry_threshold=best.entry_threshold,
            exit_threshold=best.exit_threshold,
            min_hold_bars=best.min_hold_bars,
            initial_position=config.backtest.initial_position,
            portfolio_mode=config.backtest.portfolio_mode,
        ),
        "baselines": _baseline_reports(data_split, config),
        "validation_candidates": [
            _candidate_report(candidate)
            for candidate in model_selection.candidates
        ],
        "candidate_summary": [
            _candidate_summary(candidate)
            for candidate in model_selection.candidates
        ],
    }


def _candidate_report(candidate: object) -> dict[str, Any]:
    payload = {
        "name": candidate.name,
        "model_type": candidate.model_type,
        "validation_metrics": candidate.validation_metrics,
        "validation_backtest_metrics": candidate.validation_backtest_metrics,
        "return_buffer": candidate.return_buffer,
        "entry_threshold": candidate.entry_threshold,
        "exit_threshold": candidate.exit_threshold,
        "min_hold_bars": candidate.min_hold_bars,
        "validation_cumulative_return": candidate.validation_backtest_metrics[
            "cumulative_return"
        ],
        "validation_max_drawdown": candidate.validation_backtest_metrics[
            "max_drawdown"
        ],
        "validation_turnover": candidate.validation_backtest_metrics["turnover"],
        "validation_traded_bars": candidate.validation_backtest_metrics["traded_bars"],
        "validation_exposure_ratio": candidate.validation_backtest_metrics[
            "exposure_ratio"
        ],
        "validation_score": candidate.validation_score,
        "return_over_drawdown": candidate.return_over_drawdown,
        "eligible": (
            not candidate.rejection_reasons
            and math.isfinite(candidate.validation_score)
        ),
        "rejection_reasons": candidate.rejection_reasons,
    }
    return payload


def _candidate_summary(candidate: object) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "return_buffer": candidate.return_buffer,
        "entry_threshold": candidate.entry_threshold,
        "exit_threshold": candidate.exit_threshold,
        "min_hold_bars": candidate.min_hold_bars,
        "validation_cumulative_return": candidate.validation_backtest_metrics[
            "cumulative_return"
        ],
        "validation_max_drawdown": candidate.validation_backtest_metrics[
            "max_drawdown"
        ],
        "validation_exposure_ratio": candidate.validation_backtest_metrics[
            "exposure_ratio"
        ],
        "validation_traded_bars": candidate.validation_backtest_metrics["traded_bars"],
        "validation_turnover": candidate.validation_backtest_metrics["turnover"],
        "validation_score": candidate.validation_score,
        "return_over_drawdown": candidate.return_over_drawdown,
        "eligible": (
            not candidate.rejection_reasons
            and math.isfinite(candidate.validation_score)
        ),
        "rejection_reasons": candidate.rejection_reasons,
    }


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
            "return_over_drawdown": model_selection.best_candidate.return_over_drawdown,
            "entry_threshold": model_selection.best_candidate.entry_threshold,
            "exit_threshold": model_selection.best_candidate.exit_threshold,
            "min_hold_bars": model_selection.best_candidate.min_hold_bars,
            "return_buffer": model_selection.best_candidate.return_buffer,
        }
    best_before_filters = _best_candidate_before_filters(model_selection)
    return {
        "created_at_utc": _utc_now().isoformat(),
        "trading_symbol": config.trading_symbol,
        "signal_symbols": config.signal_symbols,
        "timeframe": config.timeframe,
        "prediction_horizon_bars": config.prediction_horizon_bars,
        "model_task": config.model_task,
        "min_required_future_return": config.min_required_future_return,
        "return_buffer": config.backtest.return_buffer,
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "portfolio_mode": config.backtest.portfolio_mode,
        "sell_mode": SELL_MODE,
        "shorting_enabled": False,
        "leverage": 1,
        "hold_behavior": "keep_previous_position",
        "initial_position": config.backtest.initial_position,
        "min_validation_cumulative_return": (
            config.backtest.min_validation_cumulative_return
        ),
        "min_validation_exposure_ratio": config.backtest.min_validation_exposure_ratio,
        "min_validation_traded_bars": config.backtest.min_validation_traded_bars,
        "max_validation_drawdown_filter": config.backtest.max_validation_drawdown,
        "max_validation_turnover_filter": config.backtest.max_validation_turnover,
        "label_generation": "split_local",
        "model_selection_status": model_selection.model_selection_status,
        "risk_filters_applied": True,
        "rejected_candidate_count": model_selection.rejected_candidate_count,
        "eligible_candidate_count": model_selection.eligible_candidate_count,
        "trading_enabled": model_selection.best_candidate is not None,
        "no_trade_reason": (
            None
            if model_selection.best_candidate is not None
            else "No candidate passed binary long/cash validation filters"
        ),
        "best_candidate_before_filters": (
            None if best_before_filters is None else _candidate_summary(best_before_filters)
        ),
        "best_candidate_rejection_reasons": (
            None if best_before_filters is None else best_before_filters.rejection_reasons
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
        "exposure_ratio": 0.0,
        "mean_strategy_return": 0.0,
        "strategy_return_sum": 0.0,
        "cumulative_return": 0.0,
        "benchmark_cumulative_return": None,
        "hit_rate": 0.0,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "portfolio_mode": PORTFOLIO_MODE,
        "sell_mode": SELL_MODE,
        "shorting_enabled": False,
        "leverage": 1,
        "hold_behavior": "keep_previous_position",
        "initial_position": 0,
        "executed_position_min": 0.0,
        "executed_position_max": 0.0,
        "executed_positions_are_long_cash": True,
        "baseline": "cash",
        }


def _baseline_reports(
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Return validation/test baseline backtests for long/cash comparison."""
    return {
        "validation": _baseline_partition_reports(data_split.validation, config),
        "test": _baseline_partition_reports(data_split.test, config),
    }


def _baseline_partition_reports(
    frame: pd.DataFrame,
    config: TrainingConfig,
) -> dict[str, Any]:
    if frame.empty:
        return {}

    future_returns = frame[TARGET_RETURN_COLUMN]
    close_column = f"{config.trading_symbol}_close"
    cash = _cash_baseline_metrics(len(frame))
    cash["benchmark_cumulative_return"] = _benchmark_cumulative_return(future_returns)

    reports = {
        "cash": cash,
        "buy_and_hold": _baseline_backtest(
            predictions=pd.Series(1, index=frame.index),
            future_returns=future_returns,
            config=config,
        ),
    }
    reports["buy_and_hold"]["baseline"] = "buy_and_hold"

    if close_column in frame.columns:
        close = frame[close_column]
        sma_signal = close > close.rolling(72, min_periods=72).mean()
        momentum_signal = close.pct_change(12) > 0.0
        reports["sma_trend"] = _baseline_backtest(
            predictions=_binary_position_signal(sma_signal),
            future_returns=future_returns,
            config=config,
        )
        reports["sma_trend"]["baseline"] = "sma_trend"
        reports["momentum_12"] = _baseline_backtest(
            predictions=_binary_position_signal(momentum_signal),
            future_returns=future_returns,
            config=config,
        )
        reports["momentum_12"]["baseline"] = "momentum_12"
    return reports


def _baseline_backtest(
    *,
    predictions: pd.Series,
    future_returns: pd.Series,
    config: TrainingConfig,
) -> dict[str, Any]:
    return backtest_metrics(
        predictions=predictions,
        future_returns=future_returns,
        transaction_fee=config.backtest.transaction_fee,
        initial_position=config.backtest.initial_position,
        portfolio_mode=config.backtest.portfolio_mode,
    )


def _binary_position_signal(mask: pd.Series) -> pd.Series:
    """Convert a boolean long/cash condition to explicit long/cash signals."""
    return pd.Series(mask.fillna(False).map({True: 1, False: -1}), index=mask.index)


def _benchmark_cumulative_return(future_returns: pd.Series) -> float:
    return float((1.0 + future_returns.fillna(0.0)).prod() - 1.0)


def _best_candidate_before_filters(
    model_selection: ModelSelectionResult,
) -> object | None:
    candidates = model_selection.candidates
    if not candidates:
        return None
    finite = [
        candidate
        for candidate in candidates
        if math.isfinite(candidate.return_over_drawdown)
    ]
    if finite:
        return max(finite, key=lambda candidate: candidate.return_over_drawdown)
    return candidates[0]


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
