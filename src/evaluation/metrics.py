"""Reusable ML metrics and long/cash backtest calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score

CLASS_LABELS = [0, 1]
RETURN_OVER_DRAWDOWN_METRIC = "validation_high_risk_score"
VALIDATION_SCORE_EPSILON = 1e-9
EXECUTION_LAG_BARS = 1
PORTFOLIO_MODE = "all_in_long_cash"
SELL_MODE = "cash"


def classification_metrics(
    y_true: pd.Series,
    predictions: object,
) -> dict[str, Any]:
    """Return JSON-serializable binary classification metrics."""
    y_true_binary = pd.Series(y_true).map(lambda value: 1 if int(value) == 1 else 0)
    predictions_binary = pd.Series(predictions).map(
        lambda value: 1 if int(value) == 1 else 0
    )
    matrix = confusion_matrix(y_true_binary, predictions_binary, labels=CLASS_LABELS)
    class_counts = y_true_binary.value_counts().sort_index().to_dict()
    return {
        "accuracy": float(accuracy_score(y_true_binary, predictions_binary)),
        "precision": float(
            precision_score(y_true_binary, predictions_binary, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true_binary, predictions_binary, zero_division=0)
        ),
        "f1": float(f1_score(y_true_binary, predictions_binary, zero_division=0)),
        "precision_macro": float(
            precision_score(
                y_true_binary,
                predictions_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_binary,
                predictions_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_binary,
                predictions_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "confusion_matrix": {
            "labels": CLASS_LABELS,
            "matrix": matrix.astype(int).tolist(),
        },
        "support": {
            str(label): int((y_true_binary == label).sum()) for label in CLASS_LABELS
        },
        "class_balance": {str(key): int(value) for key, value in class_counts.items()},
    }


def backtest_metrics(
    *,
    predictions: object,
    future_returns: pd.Series,
    transaction_fee: float,
    initial_position: int = 0,
    portfolio_mode: str = PORTFOLIO_MODE,
) -> dict[str, Any]:
    """Backtest explicit BUY/HOLD/SELL class signals in long/cash mode."""
    target_positions = signals_to_target_positions(
        predictions=predictions,
        index=future_returns.index,
        initial_position=initial_position,
    )
    return _backtest_from_target_positions(
        target_positions=target_positions,
        future_returns=future_returns,
        transaction_fee=transaction_fee,
        initial_position=initial_position,
        portfolio_mode=portfolio_mode,
    )


def probability_policy_backtest(
    *,
    probabilities: object,
    future_returns: pd.Series,
    transaction_fee: float,
    entry_threshold: float,
    exit_threshold: float,
    min_hold_bars: int,
    initial_position: int = 0,
    portfolio_mode: str = PORTFOLIO_MODE,
) -> dict[str, Any]:
    """Backtest a probability-threshold long/cash policy with hysteresis."""
    if entry_threshold <= exit_threshold:
        raise ValueError("entry_threshold must be greater than exit_threshold.")
    if min_hold_bars < 0:
        raise ValueError("min_hold_bars must be non-negative.")
    probabilities_series = pd.Series(
        np.asarray(probabilities, dtype=float),
        index=future_returns.index,
    )
    target_positions, entry_signals, exit_signals = probabilities_to_target_positions(
        probabilities=probabilities_series,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        min_hold_bars=min_hold_bars,
        initial_position=initial_position,
    )
    metrics = _backtest_from_target_positions(
        target_positions=target_positions,
        future_returns=future_returns,
        transaction_fee=transaction_fee,
        initial_position=initial_position,
        portfolio_mode=portfolio_mode,
    )
    metrics.update(
        {
            "entry_threshold": float(entry_threshold),
            "exit_threshold": float(exit_threshold),
            "min_hold_bars": int(min_hold_bars),
            "entry_signals": int(entry_signals),
            "exit_signals": int(exit_signals),
            "executed_position_changes": int(metrics["turnover"]),
            "percent_time_long": metrics["exposure_ratio"],
            "percent_time_cash": 1.0 - metrics["exposure_ratio"],
            "probability_diagnostics": probability_diagnostics(probabilities_series),
        }
    )
    return metrics


def probabilities_to_target_positions(
    *,
    probabilities: pd.Series,
    entry_threshold: float,
    exit_threshold: float,
    min_hold_bars: int,
    initial_position: int,
) -> tuple[pd.Series, int, int]:
    """Convert probabilities to target positions using hysteresis and min hold."""
    if initial_position not in (0, 1):
        raise ValueError("initial_position must be 0 (cash) or 1 (long).")

    current_position = float(initial_position)
    bars_since_change = min_hold_bars
    positions: list[float] = []
    entry_signals = 0
    exit_signals = 0

    for probability in probabilities:
        can_change = bars_since_change >= min_hold_bars
        if current_position == 0.0 and probability >= entry_threshold and can_change:
            current_position = 1.0
            bars_since_change = 0
            entry_signals += 1
        elif current_position == 1.0 and probability <= exit_threshold and can_change:
            current_position = 0.0
            bars_since_change = 0
            exit_signals += 1
        else:
            bars_since_change += 1
        positions.append(current_position)

    return pd.Series(positions, index=probabilities.index, dtype=float), entry_signals, exit_signals


def signals_to_target_positions(
    *,
    predictions: object,
    index: pd.Index,
    initial_position: int,
) -> pd.Series:
    """Map class signals into all-in long/cash target positions."""
    signals = pd.Series(np.asarray(predictions, dtype=int), index=index)
    positions: list[float] = []
    current_position = float(initial_position)
    for signal in signals:
        if signal == 1:
            current_position = 1.0
        elif signal == -1:
            current_position = 0.0
        elif signal != 0:
            raise ValueError(f"Unsupported signal class for long/cash mode: {signal!r}.")
        positions.append(current_position)
    return pd.Series(positions, index=index, dtype=float)


def probability_diagnostics(probabilities: pd.Series) -> dict[str, Any]:
    """Return compact probability distribution diagnostics."""
    bins = [index / 10 for index in range(11)]
    labels = [f"{bins[index]:.1f}-{bins[index + 1]:.1f}" for index in range(10)]
    histogram = pd.cut(
        probabilities.clip(lower=0.0, upper=1.0),
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    ).value_counts(sort=False)
    return {
        "average_probability_long": float(probabilities.mean()),
        "histogram": {label: int(histogram.get(label, 0)) for label in labels},
    }


def validation_return_over_drawdown_score(backtest: dict[str, Any]) -> float:
    """Return cumulative return divided by absolute drawdown."""
    try:
        cumulative_return = float(backtest["cumulative_return"])
        max_drawdown = float(backtest["max_drawdown"])
        denominator = max(abs(max_drawdown), VALIDATION_SCORE_EPSILON)
        score = cumulative_return / denominator
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return -math.inf

    if not math.isfinite(score):
        return -math.inf
    return float(score)


def validation_high_risk_score(backtest: dict[str, Any]) -> float:
    """Score eligible candidates by upside with drawdown and turnover penalties."""
    try:
        cumulative_return = float(backtest["cumulative_return"])
        max_drawdown = float(backtest["max_drawdown"])
        turnover = float(backtest["turnover"])
        score = cumulative_return - (0.25 * abs(max_drawdown)) - (0.001 * turnover)
    except (KeyError, TypeError, ValueError):
        return -math.inf

    if not math.isfinite(score):
        return -math.inf
    return float(score)


def validation_risk_filter_rejection_reason(
    backtest: dict[str, Any],
    *,
    min_validation_cumulative_return: float,
    min_validation_exposure_ratio: float,
    min_validation_traded_bars: int,
    max_validation_drawdown: float,
    max_validation_turnover: float,
) -> list[str]:
    """Return rejection reasons when validation eligibility/risk filters fail."""
    try:
        cumulative_return = float(backtest["cumulative_return"])
        exposure_ratio = float(backtest["exposure_ratio"])
        traded_bars = int(backtest["traded_bars"])
        max_drawdown = float(backtest["max_drawdown"])
        bars = float(backtest["bars"])
        turnover = float(backtest["turnover"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return ["invalid_validation_backtest_metrics"]

    reasons: list[str] = []
    if bars <= 0.0:
        reasons.append("validation_bars_not_positive")
    if cumulative_return < min_validation_cumulative_return:
        reasons.append("validation_return_below_minimum")
    if exposure_ratio < min_validation_exposure_ratio:
        reasons.append("validation_exposure_ratio_below_minimum")
    if traded_bars < min_validation_traded_bars:
        reasons.append("validation_traded_bars_below_minimum")
    if max_drawdown < max_validation_drawdown:
        reasons.append("validation_max_drawdown_below_limit")
    if turnover > max_validation_turnover:
        reasons.append("validation_turnover_above_maximum")
    return reasons


def has_only_finite_numbers(value: object) -> bool:
    """Return False when a nested metric payload contains NaN or infinity."""
    if isinstance(value, dict):
        return all(has_only_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(has_only_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _backtest_from_target_positions(
    *,
    target_positions: pd.Series,
    future_returns: pd.Series,
    transaction_fee: float,
    initial_position: int,
    portfolio_mode: str,
) -> dict[str, Any]:
    if portfolio_mode != PORTFOLIO_MODE:
        raise ValueError(f"Unsupported portfolio_mode: {portfolio_mode!r}.")
    if initial_position not in (0, 1):
        raise ValueError("initial_position must be 0 (cash) or 1 (long).")
    executed_positions = target_positions.shift(EXECUTION_LAG_BARS).fillna(
        float(initial_position)
    )
    if not set(executed_positions.unique()).issubset({0.0, 1.0}):
        raise ValueError("Executed positions must contain only 0 or 1.")

    realized_returns = future_returns.astype(float)
    gross_strategy_returns = executed_positions * realized_returns
    turnover = executed_positions.diff().abs().fillna(0.0)
    net_strategy_returns = gross_strategy_returns - (turnover * transaction_fee)
    equity_curve = (1.0 + net_strategy_returns).cumprod()

    traded = executed_positions != 0
    bars = int(len(net_strategy_returns))
    traded_bars = int(traded.sum())
    cumulative_return = float(equity_curve.iloc[-1] - 1.0) if bars else 0.0
    benchmark_return = float((1.0 + realized_returns).prod() - 1.0) if bars else 0.0
    hit_rate = 0.0
    if traded.any():
        hit_rate = float((gross_strategy_returns[traded] > 0.0).mean())

    return {
        "transaction_fee": float(transaction_fee),
        "bars": bars,
        "traded_bars": traded_bars,
        "exposure_ratio": float(traded_bars / bars) if bars else 0.0,
        "mean_strategy_return": float(net_strategy_returns.mean()) if bars else 0.0,
        "strategy_return_sum": float(net_strategy_returns.sum()),
        "cumulative_return": cumulative_return,
        "benchmark_cumulative_return": benchmark_return,
        "vs_benchmark": cumulative_return - benchmark_return,
        "capture_ratio": _capture_ratio(cumulative_return, benchmark_return),
        "hit_rate": hit_rate,
        "max_drawdown": _max_drawdown(equity_curve) if bars else 0.0,
        "turnover": float(turnover.sum()),
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "portfolio_mode": portfolio_mode,
        "sell_mode": SELL_MODE,
        "shorting_enabled": False,
        "leverage": 1,
        "hold_behavior": "keep_previous_position",
        "initial_position": int(initial_position),
        "executed_position_min": float(executed_positions.min()) if bars else 0.0,
        "executed_position_max": float(executed_positions.max()) if bars else 0.0,
        "executed_positions_are_long_cash": bool(
            set(executed_positions.unique()).issubset({0.0, 1.0})
        ),
    }


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return float(drawdown.min())


def _capture_ratio(strategy_return: float, benchmark_return: float) -> float | None:
    if abs(benchmark_return) < VALIDATION_SCORE_EPSILON:
        return None
    return float(strategy_return / benchmark_return)
