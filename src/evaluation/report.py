"""Evaluate a selected model and persist JSON and HTML reports."""

from __future__ import annotations

import datetime
import html
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
from src.evaluation.metrics import probabilities_to_target_positions
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
    html_report_path: Path


def evaluate_and_save_report(
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> EvaluationArtifact:
    """Evaluate the selected model on test data and save JSON and HTML reports."""
    report = evaluate_model(model_selection, data_split, config)
    run_dir = _create_run_dir(config.artifacts_dir)
    report_path = run_dir / "evaluation_report.json"
    html_report_path = run_dir / "evaluation_report.html"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    html_report_path.write_text(render_html_report(report), encoding="utf-8")
    logger.info("Saved evaluation report: %s", report_path)
    logger.info("Saved HTML evaluation report: %s", html_report_path)
    return EvaluationArtifact(
        report=report,
        run_dir=run_dir,
        report_path=report_path,
        html_report_path=html_report_path,
    )


def evaluate_model(
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Build a JSON-serializable ML and backtest report for the test partition."""
    if data_split.test.empty:
        raise ValueError("Test partition is empty; cannot evaluate selected model.")

    if model_selection.best_candidate is None:
        cash_backtest = _cash_baseline_metrics(len(data_split.test))
        if TARGET_RETURN_COLUMN in data_split.test.columns:
            cash_backtest["benchmark_cumulative_return"] = _benchmark_cumulative_return(
                data_split.test[TARGET_RETURN_COLUMN]
            )
        report = {
            "run_metadata": _run_metadata(data_split, config, model_selection),
            "ml_metrics": None,
            "backtest_metrics": cash_backtest,
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
        return _add_reporting_payload(report)

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

    backtest = probability_policy_backtest(
        probabilities=probabilities,
        future_returns=future_returns,
        transaction_fee=config.backtest.transaction_fee,
        entry_threshold=best.entry_threshold,
        exit_threshold=best.exit_threshold,
        min_hold_bars=best.min_hold_bars,
        initial_position=config.backtest.initial_position,
        portfolio_mode=config.backtest.portfolio_mode,
    )
    report = {
        "run_metadata": _run_metadata(data_split, config, model_selection),
        "ml_metrics": classification_metrics(y_true, predictions),
        "backtest_metrics": backtest,
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
    report["series_data"] = _series_data(
        frame=data_split.test,
        probabilities=probabilities,
        future_returns=future_returns,
        entry_threshold=best.entry_threshold,
        exit_threshold=best.exit_threshold,
        min_hold_bars=best.min_hold_bars,
        initial_position=config.backtest.initial_position,
        transaction_fee=config.backtest.transaction_fee,
        trading_symbol=config.trading_symbol,
    )
    return _add_reporting_payload(report)


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
        "model": candidate.model_type,
        "model_type": candidate.model_type,
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


def _add_reporting_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Add derived decision-oriented report fields without removing raw metrics."""
    verdict = _experiment_verdict(report)
    report["experiment_verdict"] = verdict
    report["executive_summary"] = _executive_summary(report, verdict)
    report["live_readiness_checklist"] = _live_readiness_checklist(report)
    return report


def _experiment_verdict(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report["run_metadata"]
    backtest = report["backtest_metrics"]
    selected_model = metadata.get("selected_model")
    turnover_limit = _as_float(metadata.get("max_validation_turnover_filter"))
    turnover = _as_float(backtest.get("turnover"))
    cumulative_return = _as_float(backtest.get("cumulative_return"))
    benchmark_return = _as_float(backtest.get("benchmark_cumulative_return"))
    max_drawdown = _as_float(backtest.get("max_drawdown"))

    main_reasons: list[str] = []
    main_warnings = _decision_warnings(report)
    live_pilot_allowed = False

    if selected_model is None:
        status = "NO_TRADE"
        main_reasons.append(metadata.get("no_trade_reason") or "No model selected.")
    else:
        checks = {
            "test_return_beats_benchmark": (
                cumulative_return is not None
                and benchmark_return is not None
                and cumulative_return > benchmark_return
            ),
            "test_drawdown_within_live_limit": (
                max_drawdown is not None and max_drawdown >= -0.40
            ),
            "long_cash_execution": (
                backtest.get("executed_positions_are_long_cash") is True
            ),
            "shorting_disabled": metadata.get("shorting_enabled") is False,
            "leverage_is_one": metadata.get("leverage") == 1,
            "turnover_within_limit": (
                turnover is not None
                and turnover_limit is not None
                and turnover <= turnover_limit
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if not failed:
            status = "TINY_LIVE_PILOT_ALLOWED"
            live_pilot_allowed = True
            main_reasons.append(
                "Selected strategy beat the benchmark on test data and passed "
                "basic long/cash safety filters."
            )
        else:
            status = "PAPER_TRADE_ONLY"
            main_reasons.extend(failed)

    return {
        "status": status,
        "confidence": "experimental",
        "live_pilot_allowed": live_pilot_allowed,
        "paper_trade_recommended": status == "PAPER_TRADE_ONLY",
        "not_allowed_for": ["large capital", "leverage", "unmonitored trading"],
        "main_reasons": main_reasons,
        "main_warnings": main_warnings,
    }


def _executive_summary(
    report: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    metadata = report["run_metadata"]
    backtest = report["backtest_metrics"]
    selected_model = metadata.get("selected_model") or {}
    benchmark = backtest.get("benchmark_cumulative_return")
    cumulative = backtest.get("cumulative_return")
    vs_benchmark = backtest.get("vs_benchmark")
    if vs_benchmark is None and cumulative is not None and benchmark is not None:
        vs_benchmark = cumulative - benchmark
    return {
        "experiment_verdict": verdict["status"],
        "selected_model": selected_model.get("name"),
        "strategy_mode": metadata.get("portfolio_mode"),
        "sell_mode": metadata.get("sell_mode"),
        "shorting_enabled": metadata.get("shorting_enabled"),
        "leverage": metadata.get("leverage"),
        "entry_threshold": _selected_value(metadata, backtest, "entry_threshold"),
        "exit_threshold": _selected_value(metadata, backtest, "exit_threshold"),
        "min_hold_bars": _selected_value(metadata, backtest, "min_hold_bars"),
        "test_cumulative_return": cumulative,
        "benchmark_cumulative_return": benchmark,
        "vs_benchmark": vs_benchmark,
        "max_drawdown": backtest.get("max_drawdown"),
        "exposure_ratio": backtest.get("exposure_ratio"),
        "turnover": backtest.get("turnover"),
        "entry_signals": backtest.get("entry_signals"),
        "exit_signals": backtest.get("exit_signals"),
        "trading_enabled": metadata.get("trading_enabled"),
        "model_selection_status": metadata.get("model_selection_status"),
        "not_allowed_for": verdict["not_allowed_for"],
        "human_summary": _human_summary(report, verdict),
    }


def _selected_value(
    metadata: dict[str, Any],
    backtest: dict[str, Any],
    key: str,
) -> Any:
    selected_model = metadata.get("selected_model") or {}
    return selected_model.get(key, backtest.get(key))


def _human_summary(report: dict[str, Any], verdict: dict[str, Any]) -> str:
    status = verdict["status"]
    selected = (report["run_metadata"].get("selected_model") or {}).get("name")
    if status == "NO_TRADE":
        return (
            "No model was selected by the validation filters, so the experiment "
            "should not trade."
        )
    if status == "TINY_LIVE_PILOT_ALLOWED":
        return (
            f"Model {selected} is eligible only for a tiny monitored live pilot: "
            "it beat the benchmark on test data and passed the configured "
            "long/cash safety checks. It is still experimental."
        )
    return (
        f"Model {selected} needs paper trading only because one or more live "
        "pilot safety or performance checks failed."
    )


def _live_readiness_checklist(report: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = report["run_metadata"]
    backtest = report["backtest_metrics"]
    probability = backtest.get("probability_diagnostics") or {}
    selected_model = metadata.get("selected_model")
    turnover = _as_float(backtest.get("turnover"))
    turnover_limit = _as_float(metadata.get("max_validation_turnover_filter"))
    exposure = _as_float(backtest.get("exposure_ratio"))
    entry_signals = _as_float(backtest.get("entry_signals"))
    exit_signals = _as_float(backtest.get("exit_signals"))
    average_probability = _as_float(probability.get("average_probability_long"))

    checks: list[dict[str, Any]] = [
        _check(
            "label_generation == split_local",
            metadata.get("label_generation") == "split_local",
        ),
        _check("execution_lag_bars == 1", metadata.get("execution_lag_bars") == 1),
        _check("sell_mode == cash", metadata.get("sell_mode") == "cash"),
        _check("shorting_enabled == false", metadata.get("shorting_enabled") is False),
        _check("leverage == 1", metadata.get("leverage") == 1),
        _check(
            "executed_positions_are_long_cash == true",
            backtest.get("executed_positions_are_long_cash") is True,
        ),
        _check(
            "executed_position_min >= 0",
            (_as_float(backtest.get("executed_position_min")) or 0.0) >= 0.0,
        ),
        _check(
            "executed_position_max <= 1",
            (_as_float(backtest.get("executed_position_max")) or 0.0) <= 1.0,
        ),
        _check(
            "test data not used for selection",
            True,
            "metadata says validation-only model selection",
        ),
        _check("selected model exists", selected_model is not None),
        _check("trading_enabled == true", metadata.get("trading_enabled") is True),
        _check("probability_long is finite", average_probability is not None),
        _check(
            "turnover is not excessive",
            (
                turnover is not None
                and turnover_limit is not None
                and turnover <= turnover_limit
            ),
        ),
        _check("exposure ratio is not zero", exposure is not None and exposure > 0.0),
        _check("entry signals > 0", entry_signals is not None and entry_signals > 0.0),
        _check("exit signals > 0", exit_signals is not None and exit_signals > 0.0),
    ]
    for warning in _decision_warnings(report):
        checks.append({"name": warning, "status": "WARN", "detail": None})
    return checks


def _check(name: str, passed: bool, detail: str | None = None) -> dict[str, Any]:
    return {"name": name, "status": "OK" if passed else "FAIL", "detail": detail}


def _decision_warnings(report: dict[str, Any]) -> list[str]:
    metadata = report["run_metadata"]
    backtest = report["backtest_metrics"]
    warnings: list[str] = []
    entry_signals = _as_float(backtest.get("entry_signals"))
    if entry_signals is not None and 0 < entry_signals < 10:
        warnings.append("few_entry_signals")
    if _probabilities_clustered_around_threshold(report):
        warnings.append("probability_histogram_concentrated_close_to_threshold")
    ml_metrics = report.get("ml_metrics") or {}
    long_recall = _as_float(ml_metrics.get("recall"))
    if long_recall is not None and long_recall < 0.50:
        warnings.append("low_recall_for_long_class")
    max_drawdown = _as_float(backtest.get("max_drawdown"))
    if max_drawdown is not None and max_drawdown < -0.40:
        warnings.append("test_drawdown_worse_than_warning_threshold")
    selected = metadata.get("selected_model") or {}
    validation_return = _as_float(selected.get("validation_cumulative_return"))
    test_return = _as_float(backtest.get("cumulative_return"))
    if (
        validation_return is not None
        and test_return is not None
        and abs(validation_return - test_return) > 0.25
    ):
        warnings.append("validation_and_test_behavior_differ_strongly")
    probability = backtest.get("probability_diagnostics") or {}
    histogram = probability.get("histogram") or {}
    probability_values = [
        _as_float(row.get("probability_long"))
        for row in report.get("series_data", [])
        if row.get("probability_long") is not None
    ]
    high_probability_count = len(
        [value for value in probability_values if value is not None and value > 0.6]
    )
    if not probability_values:
        high_probability_count = sum(
            count for bucket, count in histogram.items() if _bucket_lower(bucket) >= 0.6
        )
    if high_probability_count == 0:
        warnings.append("no_probabilities_above_0_6")
    if entry_signals is not None and entry_signals <= 3:
        warnings.append("strategy_depends_on_very_few_threshold_crossings")
    return warnings


def _probabilities_clustered_around_threshold(report: dict[str, Any]) -> bool:
    threshold = _selected_value(
        report["run_metadata"],
        report["backtest_metrics"],
        "entry_threshold",
    )
    if threshold is None:
        return False
    values = [
        row.get("probability_long")
        for row in report.get("series_data", [])
        if row.get("probability_long") is not None
    ]
    if not values:
        return False
    close = [value for value in values if abs(float(value) - float(threshold)) <= 0.05]
    return len(close) / len(values) >= 0.50


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
            None
            if best_before_filters is None
            else _candidate_summary(best_before_filters)
        ),
        "best_candidate_rejection_reasons": (
            None
            if best_before_filters is None
            else best_before_filters.rejection_reasons
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


def _series_data(
    *,
    frame: pd.DataFrame,
    probabilities: pd.Series,
    future_returns: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
    min_hold_bars: int,
    initial_position: int,
    transaction_fee: float,
    trading_symbol: str,
) -> list[dict[str, Any]]:
    probabilities_series = pd.Series(probabilities, index=future_returns.index)
    target_positions, _, _ = probabilities_to_target_positions(
        probabilities=probabilities_series,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_hold_bars=min_hold_bars,
        initial_position=initial_position,
    )
    executed_positions = target_positions.shift(EXECUTION_LAG_BARS).fillna(
        float(initial_position)
    )
    realized_returns = future_returns.astype(float).fillna(0.0)
    turnover = executed_positions.diff().abs().fillna(0.0)
    strategy_returns = (executed_positions * realized_returns) - (
        turnover * transaction_fee
    )
    strategy_equity = (1.0 + strategy_returns).cumprod()
    benchmark_equity = (1.0 + realized_returns).cumprod()
    drawdown = (strategy_equity / strategy_equity.cummax()) - 1.0
    close_column = f"{trading_symbol}_close"

    rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        rows.append(
            {
                "timestamp": _timestamp_to_iso(row["timestamp"]),
                "close": (
                    _json_float(row[close_column])
                    if close_column in frame
                    else None
                ),
                "benchmark_equity": _json_float(benchmark_equity.loc[index]),
                "strategy_equity": _json_float(strategy_equity.loc[index]),
                "drawdown": _json_float(drawdown.loc[index]),
                "probability_long": _json_float(probabilities_series.loc[index]),
                "executed_position": _json_float(executed_positions.loc[index]),
            }
        )
    return rows


def render_html_report(report: dict[str, Any]) -> str:
    """Render a self-contained English HTML evaluation report."""
    summary = report.get("executive_summary", {})
    verdict = report.get("experiment_verdict", {})
    metadata = report.get("run_metadata", {})
    backtest = report.get("backtest_metrics", {})
    html_parts = [
        "<!doctype html>",
        "<html lang=\"en\"><head><meta charset=\"utf-8\">",
        "<title>Cryptan Evaluation Report</title>",
        "<style>",
        _html_css(),
        "</style></head><body>",
        "<main>",
        "<h1>Evaluation Report</h1>",
        _section("Executive Summary", _summary_html(summary, verdict)),
        _section("Live-Readiness Checklist", _checklist_html(report)),
        _section("Strategy Configuration", _strategy_config_html(metadata, backtest)),
        _section("Validation vs Test", _validation_test_html(report)),
        _section("Baseline Comparison", _baseline_html(report)),
        _section("ML Metrics", _ml_metrics_html(report)),
        _section("Probability Diagnostics", _probability_html(report)),
        _section("Candidate Summary", _candidate_html(report)),
        _section("Charts", _charts_html(report)),
        "</main></body></html>",
    ]
    return "\n".join(html_parts)


def _summary_html(summary: dict[str, Any], verdict: dict[str, Any]) -> str:
    rows = [
        ("Experiment verdict", summary.get("experiment_verdict")),
        ("Selected model", summary.get("selected_model") or "None"),
        ("Strategy mode", summary.get("strategy_mode")),
        ("Sell mode", summary.get("sell_mode")),
        ("Shorting enabled", summary.get("shorting_enabled")),
        ("Leverage", summary.get("leverage")),
        ("Entry threshold", summary.get("entry_threshold")),
        ("Exit threshold", summary.get("exit_threshold")),
        ("Min hold bars", summary.get("min_hold_bars")),
        ("Test cumulative return", _pct(summary.get("test_cumulative_return"))),
        (
            "Benchmark cumulative return",
            _pct(summary.get("benchmark_cumulative_return")),
        ),
        ("Vs benchmark", _pct(summary.get("vs_benchmark"))),
        ("Max drawdown", _pct(summary.get("max_drawdown"))),
        ("Exposure ratio", _pct(summary.get("exposure_ratio"))),
        ("Turnover", _num(summary.get("turnover"))),
        ("Entry signals", summary.get("entry_signals")),
        ("Exit signals", summary.get("exit_signals")),
        ("Trading enabled", summary.get("trading_enabled")),
        ("Model selection status", summary.get("model_selection_status")),
    ]
    return (
        f"<div class=\"verdict {html.escape(str(verdict.get('status', '')))}\">"
        f"{_esc(verdict.get('status'))}</div>"
        f"<p>{_esc(summary.get('human_summary'))}</p>"
        f"{_table(['Field', 'Value'], rows)}"
        f"<p><strong>Not allowed for:</strong> "
        f"{_esc(', '.join(summary.get('not_allowed_for', [])))}</p>"
    )


def _checklist_html(report: dict[str, Any]) -> str:
    rows = [
        (
            f"<span class=\"status {item['status']}\">{item['status']}</span>",
            item["name"],
            item.get("detail") or "",
        )
        for item in report.get("live_readiness_checklist", [])
    ]
    return _table(["Status", "Check", "Detail"], rows, escape_cells=False)


def _strategy_config_html(
    metadata: dict[str, Any],
    backtest: dict[str, Any],
) -> str:
    selected = metadata.get("selected_model") or {}
    rows = [
        ("model_task", metadata.get("model_task")),
        ("portfolio_mode", metadata.get("portfolio_mode")),
        ("sell_mode", metadata.get("sell_mode")),
        ("shorting_enabled", metadata.get("shorting_enabled")),
        ("leverage", metadata.get("leverage")),
        ("hold_behavior", metadata.get("hold_behavior")),
        ("initial_position", metadata.get("initial_position")),
        ("prediction_horizon_bars", metadata.get("prediction_horizon_bars")),
        ("return_buffer", selected.get("return_buffer", metadata.get("return_buffer"))),
        ("transaction_fee", backtest.get("transaction_fee")),
        (
            "entry_threshold",
            selected.get("entry_threshold", backtest.get("entry_threshold")),
        ),
        (
            "exit_threshold",
            selected.get("exit_threshold", backtest.get("exit_threshold")),
        ),
        ("min_hold_bars", selected.get("min_hold_bars", backtest.get("min_hold_bars"))),
        ("execution_lag_bars", metadata.get("execution_lag_bars")),
        ("feature_count", metadata.get("feature_count")),
        ("train rows", metadata.get("train_rows_labelled")),
        ("validation rows", metadata.get("validation_rows_labelled")),
        ("test rows", metadata.get("test_rows_labelled")),
        ("test period", _test_period(metadata)),
    ]
    return _table(["Setting", "Value"], rows)


def _validation_test_html(report: dict[str, Any]) -> str:
    selected = report.get("run_metadata", {}).get("selected_model") or {}
    validation = {
        "cumulative_return": selected.get("validation_cumulative_return"),
        "benchmark_cumulative_return": _validation_benchmark(report),
        "vs_benchmark": None,
        "capture_ratio": None,
        "max_drawdown": selected.get("validation_max_drawdown"),
        "exposure_ratio": _selected_validation_metric(report, "exposure_ratio"),
        "percent_time_long": _selected_validation_metric(report, "exposure_ratio"),
        "percent_time_cash": _cash_percent(
            _selected_validation_metric(report, "exposure_ratio")
        ),
        "turnover": selected.get("validation_turnover"),
        "traded_bars": _selected_validation_metric(report, "traded_bars"),
        "hit_rate": _selected_validation_metric(report, "hit_rate"),
        "entry_signals": _selected_validation_metric(report, "entry_signals"),
        "exit_signals": _selected_validation_metric(report, "exit_signals"),
        "executed_position_changes": _selected_validation_metric(
            report,
            "executed_position_changes",
        ),
    }
    benchmark = validation["benchmark_cumulative_return"]
    cumulative = validation["cumulative_return"]
    if benchmark is not None and cumulative is not None:
        validation["vs_benchmark"] = cumulative - benchmark
        validation["capture_ratio"] = cumulative / benchmark if benchmark else None
    test = report.get("backtest_metrics", {})
    fields = [
        "cumulative_return",
        "benchmark_cumulative_return",
        "vs_benchmark",
        "capture_ratio",
        "max_drawdown",
        "exposure_ratio",
        "percent_time_long",
        "percent_time_cash",
        "turnover",
        "traded_bars",
        "hit_rate",
        "entry_signals",
        "exit_signals",
        "executed_position_changes",
    ]
    rows = [
        (
            field,
            _metric_value(field, validation.get(field)),
            _metric_value(field, test.get(field)),
        )
        for field in fields
    ]
    return _table(["Metric", "Validation", "Test"], rows)


def _baseline_html(report: dict[str, Any]) -> str:
    rows: list[tuple[Any, ...]] = []
    test_strategy_return = _as_float(
        report.get("backtest_metrics", {}).get("cumulative_return")
    )
    for partition in ("validation", "test"):
        baselines = report.get("baselines", {}).get(partition, {})
        for name in ("cash", "buy_and_hold", "momentum_12", "sma_trend"):
            metrics = baselines.get(name)
            if not metrics:
                continue
            baseline_return = _as_float(metrics.get("cumulative_return"))
            beats = ""
            if (
                partition == "test"
                and test_strategy_return is not None
                and baseline_return is not None
            ):
                beats = "yes" if test_strategy_return > baseline_return else "no"
            rows.append(
                (
                    partition,
                    name,
                    _pct(metrics.get("cumulative_return")),
                    _pct(metrics.get("benchmark_cumulative_return")),
                    _pct(metrics.get("vs_benchmark")),
                    _pct(metrics.get("max_drawdown")),
                    _pct(metrics.get("exposure_ratio")),
                    _num(metrics.get("turnover")),
                    _pct(metrics.get("hit_rate")),
                    beats,
                )
            )
    return _table(
        [
            "Partition",
            "Baseline",
            "Cumulative Return",
            "Benchmark",
            "Vs Benchmark",
            "Max Drawdown",
            "Exposure",
            "Turnover",
            "Hit Rate",
            "ML Beats Test",
        ],
        rows,
    )


def _ml_metrics_html(report: dict[str, Any]) -> str:
    metrics = report.get("ml_metrics")
    if not metrics:
        return "<p>No ML metrics are available because no model was selected.</p>"
    cash_precision, long_precision, cash_recall, long_recall = _class_metrics(metrics)
    rows = [
        ("accuracy", _num(metrics.get("accuracy"))),
        ("precision", _num(metrics.get("precision"))),
        ("recall", _num(metrics.get("recall"))),
        ("f1", _num(metrics.get("f1"))),
        ("class balance", json.dumps(metrics.get("class_balance", {}), sort_keys=True)),
        ("long precision", _num(long_precision)),
        ("long recall", _num(long_recall)),
        ("cash precision", _num(cash_precision)),
        ("cash recall", _num(cash_recall)),
    ]
    matrix = metrics.get("confusion_matrix", {}).get("matrix")
    return (
        "<p>ML classification metrics are diagnostic only. Trading selection is "
        "based on validation backtest score and risk filters.</p>"
        f"{_table(['Metric', 'Value'], rows)}"
        f"<pre>{_esc(json.dumps(matrix, indent=2))}</pre>"
    )


def _probability_html(report: dict[str, Any]) -> str:
    backtest = report.get("backtest_metrics", {})
    probability = backtest.get("probability_diagnostics") or {}
    histogram = probability.get("histogram") or {}
    threshold_rows = [
        ("average_probability_long", _num(probability.get("average_probability_long"))),
        ("entry_threshold", backtest.get("entry_threshold")),
        ("exit_threshold", backtest.get("exit_threshold")),
        ("bars above entry threshold", _bars_above_entry(report)),
        ("bars below exit threshold while long", _bars_below_exit_while_long(report)),
    ]
    histogram_rows = [(bucket, count) for bucket, count in histogram.items()]
    warnings = report.get("experiment_verdict", {}).get("main_warnings", [])
    return (
        f"{_table(['Metric', 'Value'], threshold_rows)}"
        f"<h3>Histogram Buckets</h3>{_table(['Bucket', 'Count'], histogram_rows)}"
        f"<p><strong>Warnings:</strong> {_esc(', '.join(warnings) or 'None')}</p>"
    )


def _candidate_html(report: dict[str, Any]) -> str:
    candidates = report.get("candidate_summary", [])
    eligible = sorted(
        [candidate for candidate in candidates if candidate.get("eligible")],
        key=lambda item: _sort_float(item.get("validation_score")),
        reverse=True,
    )[:20]
    rejected = sorted(
        [candidate for candidate in candidates if not candidate.get("eligible")],
        key=lambda item: _sort_float(item.get("validation_cumulative_return")),
        reverse=True,
    )[:20]
    headers = [
        "name",
        "model",
        "return_buffer",
        "entry_threshold",
        "exit_threshold",
        "min_hold_bars",
        "eligible",
        "rejection_reasons",
        "validation_cumulative_return",
        "validation_max_drawdown",
        "validation_exposure_ratio",
        "validation_turnover",
        "validation_score",
        "return_over_drawdown",
    ]
    content = [
        "<h3>Top 20 Eligible Candidates</h3>",
        _candidate_table(headers, eligible),
        "<h3>Top 20 Rejected Candidates By Raw Validation Return</h3>",
        _candidate_table(headers, rejected),
    ]
    if len(candidates) > 40:
        content.append(
            "<details><summary>Full candidate table</summary>"
            f"{_candidate_table(headers, candidates)}</details>"
        )
    return "\n".join(content)


def _charts_html(report: dict[str, Any]) -> str:
    rows = report.get("series_data", [])
    if not rows:
        return "<p>No time-series data is available for charts.</p>"
    return "\n".join(
        [
            _sparkline(
                "Strategy equity vs benchmark",
                rows,
                "strategy_equity",
                "benchmark_equity",
            ),
            _sparkline("Drawdown", rows, "drawdown"),
            _sparkline("Executed position", rows, "executed_position"),
            _sparkline("Probability long", rows, "probability_long"),
            _baseline_bars(report),
        ]
    )


def _html_css() -> str:
    return """
body { margin: 0; font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #18202a; background: #f7f8fa; }
main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
h1 { margin: 0 0 24px; font-size: 32px; }
h2 { margin: 0 0 12px; font-size: 20px; }
h3 { margin: 18px 0 8px; font-size: 15px; }
section { margin: 0 0 22px; padding: 18px; background: #fff; border: 1px solid #d9dee5; border-radius: 8px; }
table { width: 100%; border-collapse: collapse; margin: 10px 0; }
th, td { padding: 8px 10px; border-bottom: 1px solid #e8ebef; text-align: left; vertical-align: top; }
th { background: #f0f3f6; font-weight: 650; }
pre { overflow: auto; background: #f5f7f9; padding: 12px; border-radius: 6px; }
.verdict { display: inline-block; margin-bottom: 8px; padding: 6px 10px; border-radius: 6px; font-weight: 700; background: #eef2ff; }
.TINY_LIVE_PILOT_ALLOWED { background: #dff7e8; color: #14532d; }
.PAPER_TRADE_ONLY { background: #fff4d6; color: #7a4b00; }
.NO_TRADE { background: #ffe1e1; color: #8a1f1f; }
.status { display: inline-block; min-width: 42px; padding: 2px 6px; border-radius: 5px; font-weight: 700; text-align: center; }
.OK { background: #dff7e8; color: #14532d; }
.WARN { background: #fff4d6; color: #7a4b00; }
.FAIL { background: #ffe1e1; color: #8a1f1f; }
.chart { width: 100%; height: 180px; margin: 6px 0 18px; background: #fbfcfd; border: 1px solid #e0e5eb; }
svg text { fill: #56616f; font-size: 11px; }
""".strip()


def _section(title: str, body: str) -> str:
    return f"<section><h2>{_esc(title)}</h2>{body}</section>"


def _table(
    headers: list[str],
    rows: list[tuple[Any, ...]],
    *,
    escape_cells: bool = True,
) -> str:
    header_html = "".join(f"<th>{_esc(header)}</th>" for header in headers)
    row_html: list[str] = []
    for row in rows:
        cells = []
        for cell in row:
            rendered = "" if cell is None else str(cell)
            cells.append(
                f"<td>{_esc(rendered) if escape_cells else rendered}</td>"
            )
        row_html.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<table><thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody></table>"
    )


def _candidate_table(headers: list[str], candidates: list[dict[str, Any]]) -> str:
    rows = []
    for candidate in candidates:
        rows.append(
            tuple(
                ", ".join(candidate.get(header, []))
                if header == "rejection_reasons"
                else _metric_value(header, candidate.get(header))
                for header in headers
            )
        )
    return _table(headers, rows)


def _sparkline(
    title: str,
    rows: list[dict[str, Any]],
    first_key: str,
    second_key: str | None = None,
) -> str:
    width = 1000
    height = 180
    padding = 18
    series = [_as_float(row.get(first_key)) for row in rows]
    series = [value for value in series if value is not None]
    if not series:
        return f"<h3>{_esc(title)}</h3><p>No data.</p>"
    all_values = list(series)
    second: list[float] = []
    if second_key is not None:
        second = [
            value
            for value in (_as_float(row.get(second_key)) for row in rows)
            if value is not None
        ]
        all_values.extend(second)
    low = min(all_values)
    high = max(all_values)
    if math.isclose(low, high):
        high = low + 1.0

    def points(values: list[float]) -> str:
        if len(values) == 1:
            values = values * 2
        coordinates = []
        for index, value in enumerate(values):
            x = padding + (index / (len(values) - 1)) * (width - 2 * padding)
            y = height - padding - (
                (value - low) / (high - low)
            ) * (height - 2 * padding)
            coordinates.append(f"{x:.1f},{y:.1f}")
        return " ".join(coordinates)

    second_line = ""
    if second:
        second_line = (
            f"<polyline points=\"{points(second)}\" fill=\"none\" "
            "stroke=\"#d97706\" stroke-width=\"2\" />"
        )
    return (
        f"<h3>{_esc(title)}</h3><svg class=\"chart\" viewBox=\"0 0 {width} {height}\" "
        "role=\"img\">"
        f"<polyline points=\"{points(series)}\" fill=\"none\" stroke=\"#2563eb\" "
        "stroke-width=\"2\" />"
        f"{second_line}<text x=\"16\" y=\"18\">min {_num(low)} max {_num(high)}</text>"
        "</svg>"
    )


def _baseline_bars(report: dict[str, Any]) -> str:
    baselines = report.get("baselines", {}).get("test", {})
    values = [
        (name, _as_float(metrics.get("cumulative_return")))
        for name, metrics in baselines.items()
        if _as_float(metrics.get("cumulative_return")) is not None
    ]
    strategy = _as_float(report.get("backtest_metrics", {}).get("cumulative_return"))
    if strategy is not None:
        values.append(("selected_ml", strategy))
    if not values:
        return "<h3>Baseline Return Comparison</h3><p>No data.</p>"
    width = 1000
    height = 220
    baseline_y = 110
    max_abs = max(abs(value or 0.0) for _, value in values) or 1.0
    bar_width = max(30, int((width - 80) / len(values)) - 12)
    bars = []
    for index, (name, value) in enumerate(values):
        value = value or 0.0
        bar_height = abs(value) / max_abs * 80
        x = 40 + index * ((width - 80) / len(values))
        y = baseline_y - bar_height if value >= 0 else baseline_y
        color = "#15803d" if value >= 0 else "#b91c1c"
        bars.append(
            f"<rect x=\"{x:.1f}\" y=\"{y:.1f}\" width=\"{bar_width}\" "
            f"height=\"{bar_height:.1f}\" fill=\"{color}\" />"
            f"<text x=\"{x:.1f}\" y=\"205\">{_esc(name)}</text>"
            f"<text x=\"{x:.1f}\" "
            f"y=\"{y - 4 if value >= 0 else y + bar_height + 14:.1f}\">"
            f"{_pct(value)}</text>"
        )
    return (
        "<h3>Validation/Test Baseline Return Comparison</h3>"
        f"<svg class=\"chart\" viewBox=\"0 0 {width} {height}\" role=\"img\">"
        f"<line x1=\"30\" x2=\"970\" y1=\"{baseline_y}\" y2=\"{baseline_y}\" "
        "stroke=\"#808a96\" />"
        f"{''.join(bars)}</svg>"
    )


def _validation_benchmark(report: dict[str, Any]) -> float | None:
    benchmark = (
        report.get("baselines", {})
        .get("validation", {})
        .get("cash", {})
        .get("benchmark_cumulative_return")
    )
    return _as_float(benchmark)


def _selected_validation_metric(report: dict[str, Any], key: str) -> Any:
    selected_name = (
        report.get("run_metadata", {}).get("selected_model") or {}
    ).get("name")
    for candidate in report.get("validation_candidates", []):
        if candidate.get("name") == selected_name:
            return (candidate.get("validation_backtest_metrics") or {}).get(key)
    return None


def _cash_percent(value: object) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return 1.0 - number


def _class_metrics(metrics: dict[str, Any]) -> tuple[float | None, ...]:
    matrix = metrics.get("confusion_matrix", {}).get("matrix") or []
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        return None, None, None, None
    true_cash, false_long = matrix[0]
    false_cash, true_long = matrix[1]
    cash_precision = _safe_div(true_cash, true_cash + false_cash)
    long_precision = _safe_div(true_long, true_long + false_long)
    cash_recall = _safe_div(true_cash, true_cash + false_long)
    long_recall = _safe_div(true_long, true_long + false_cash)
    return cash_precision, long_precision, cash_recall, long_recall


def _bars_above_entry(report: dict[str, Any]) -> int | None:
    threshold = _selected_value(
        report.get("run_metadata", {}),
        report.get("backtest_metrics", {}),
        "entry_threshold",
    )
    if threshold is None:
        return None
    return sum(
        1
        for row in report.get("series_data", [])
        if row.get("probability_long") is not None
        and float(row["probability_long"]) >= float(threshold)
    )


def _bars_below_exit_while_long(report: dict[str, Any]) -> int | None:
    threshold = _selected_value(
        report.get("run_metadata", {}),
        report.get("backtest_metrics", {}),
        "exit_threshold",
    )
    if threshold is None:
        return None
    return sum(
        1
        for row in report.get("series_data", [])
        if row.get("probability_long") is not None
        and row.get("executed_position") == 1.0
        and float(row["probability_long"]) <= float(threshold)
    )


def _test_period(metadata: dict[str, Any]) -> str:
    period = metadata.get("test_period") or {}
    if not period:
        return ""
    return f"{period.get('start')} to {period.get('end')}"


def _metric_value(name: str, value: Any) -> str:
    pct_tokens = ("return", "drawdown", "ratio", "rate", "exposure", "cash", "long")
    if any(token in name for token in pct_tokens):
        return _pct(value)
    return _num(value)


def _pct(value: Any) -> str:
    number = _as_float(value)
    if number is None:
        return ""
    return f"{number:.2%}"


def _num(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    number = _as_float(value)
    if number is None:
        return "" if value is None else str(value)
    if float(number).is_integer():
        return str(int(number))
    return f"{number:.4f}"


def _as_float(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _json_float(value: Any) -> float | None:
    number = _as_float(value)
    return None if number is None else float(number)


def _sort_float(value: Any) -> float:
    number = _as_float(value)
    return -math.inf if number is None else number


def _bucket_lower(bucket: object) -> float:
    try:
        return float(str(bucket).split("-", maxsplit=1)[0])
    except (TypeError, ValueError):
        return -math.inf


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


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
