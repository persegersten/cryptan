"""Reusable ML metrics and simple strategy backtest calculations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score

CLASS_LABELS = [-1, 0, 1]
RETURN_OVER_DRAWDOWN_METRIC = "validation_return_over_drawdown_with_risk_filters"
VALIDATION_SCORE_EPSILON = 1e-9
EXECUTION_LAG_BARS = 1
PORTFOLIO_MODE = "all_in_long_cash"
SELL_MODE = "cash"


def classification_metrics(
    y_true: pd.Series,
    predictions: object,
) -> dict[str, Any]:
    """Return JSON-serializable multiclass classification metrics."""
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


def backtest_metrics(
    *,
    predictions: object,
    future_returns: pd.Series,
    transaction_fee: float,
    initial_position: int = 0,
    portfolio_mode: str = PORTFOLIO_MODE,
) -> dict[str, Any]:
    """Run the long/cash prediction strategy simulator.

    Predictions are generated from features available at bar ``t`` and are
    therefore executed no earlier than bar ``t+1``. Returns at row ``t`` are
    earned by the previous in-split signal.
    """
    if portfolio_mode != PORTFOLIO_MODE:
        raise ValueError(f"Unsupported portfolio_mode: {portfolio_mode!r}.")
    if initial_position not in (0, 1):
        raise ValueError("initial_position must be 0 (cash) or 1 (long).")

    target_positions = signals_to_target_positions(
        predictions=predictions,
        index=future_returns.index,
        initial_position=initial_position,
    )
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
    hit_rate = 0.0
    if traded.any():
        hit_rate = float((gross_strategy_returns[traded] > 0.0).mean())

    return {
        "transaction_fee": float(transaction_fee),
        "bars": bars,
        "traded_bars": traded_bars,
        "exposure_ratio": float(traded_bars / bars) if bars else 0.0,
        "mean_strategy_return": float(net_strategy_returns.mean()),
        "strategy_return_sum": float(net_strategy_returns.sum()),
        "cumulative_return": float(equity_curve.iloc[-1] - 1.0),
        "benchmark_cumulative_return": float((1.0 + realized_returns).prod() - 1.0),
        "hit_rate": hit_rate,
        "max_drawdown": _max_drawdown(equity_curve),
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


def signals_to_target_positions(
    *,
    predictions: object,
    index: pd.Index,
    initial_position: int,
) -> pd.Series:
    """Map class predictions into all-in long/cash target positions."""
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


def validation_return_over_drawdown_score(backtest: dict[str, Any]) -> float:
    """Score validation backtests by cumulative return over absolute drawdown."""
    try:
        cumulative_return = float(backtest["cumulative_return"])
        max_drawdown = float(backtest["max_drawdown"])
        denominator = abs(max_drawdown)
        if denominator == 0.0:
            denominator = VALIDATION_SCORE_EPSILON
        score = cumulative_return / denominator
        if cumulative_return <= 0.0 and score >= 0.0:
            score = -VALIDATION_SCORE_EPSILON
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return -math.inf

    if not math.isfinite(score):
        return -math.inf
    return float(score)


def validation_risk_filter_rejection_reason(
    backtest: dict[str, Any],
    *,
    max_validation_drawdown: float,
    max_validation_turnover: float,
) -> str | None:
    """Return a rejection reason when validation risk filters fail."""
    try:
        max_drawdown = float(backtest["max_drawdown"])
        bars = float(backtest["bars"])
        turnover = float(backtest["turnover"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return "invalid_validation_backtest_metrics"

    if bars <= 0.0:
        return "validation_bars_not_positive"
    if max_drawdown < max_validation_drawdown:
        return f"validation_max_drawdown_below_{max_validation_drawdown:g}"
    if turnover > max_validation_turnover:
        return f"validation_turnover_above_{max_validation_turnover:g}"
    return None


def has_only_finite_numbers(value: object) -> bool:
    """Return False when a nested metric payload contains NaN or infinity."""
    if isinstance(value, dict):
        return all(has_only_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(has_only_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = (equity_curve / running_peak) - 1.0
    return float(drawdown.min())
