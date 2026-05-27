"""Train configured model candidates and select the best validation backtest."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.model import ModelCandidateConfig, TrainingConfig
from src.evaluation.metrics import EXECUTION_LAG_BARS, RETURN_OVER_DRAWDOWN_METRIC
from src.evaluation.metrics import classification_metrics, probability_policy_backtest
from src.evaluation.metrics import has_only_finite_numbers
from src.evaluation.metrics import validation_high_risk_score
from src.evaluation.metrics import validation_return_over_drawdown_score
from src.evaluation.metrics import validation_risk_filter_rejection_reason
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.registry import build_estimator
from src.splitting.chronological import ChronologicalSplit

logger = logging.getLogger(__name__)
backtest_metrics = probability_policy_backtest

_NON_FEATURE_COLUMNS = {
    "timestamp",
    TARGET_LABEL_COLUMN,
    TARGET_RETURN_COLUMN,
}


@dataclass(frozen=True)
class CandidateTrainingResult:
    """Validation result for one trained candidate."""

    name: str
    model_type: str
    model_params: dict[str, Any]
    estimator: BaseEstimator
    entry_threshold: float
    exit_threshold: float
    min_hold_bars: int
    return_buffer: float
    validation_metrics: dict[str, Any]
    validation_backtest_metrics: dict[str, Any]
    validation_score: float
    return_over_drawdown: float
    rejection_reasons: list[str]


@dataclass(frozen=True)
class ModelSelectionResult:
    """Best model plus all candidate validation scores."""

    best_candidate: CandidateTrainingResult | None
    candidates: list[CandidateTrainingResult]
    feature_columns: list[str]
    selection_metric: str

    @property
    def estimator(self) -> BaseEstimator:
        """Return the selected fitted estimator."""
        if self.best_candidate is None:
            raise ValueError("No eligible model candidate was selected.")
        return self.best_candidate.estimator

    @property
    def eligible_candidates(self) -> list[CandidateTrainingResult]:
        """Return candidates that passed validation risk filters."""
        return [
            candidate
            for candidate in self.candidates
            if math.isfinite(candidate.validation_score)
            and not candidate.rejection_reasons
        ]

    @property
    def eligible_candidate_count(self) -> int:
        """Return the number of candidates eligible for selection."""
        return len(self.eligible_candidates)

    @property
    def rejected_candidate_count(self) -> int:
        """Return the number of rejected candidates."""
        return len(self.candidates) - self.eligible_candidate_count

    @property
    def model_selection_status(self) -> str:
        """Return model selection status for reporting."""
        if self.best_candidate is None:
            return "no_eligible_model"
        return "selected"


def train_and_select_model(
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> ModelSelectionResult:
    """Train candidates and select the best validation backtest performer.

    The function fits each candidate on the chronological train partition and
    scores it using validation predictions only. It does not use test rows.
    """
    feature_columns = _select_feature_columns(data_split.train)
    candidates = _candidate_configs(config)
    if not candidates:
        raise ValueError("At least one model candidate must be configured.")

    results: list[CandidateTrainingResult] = []
    for index, candidate in enumerate(candidates, start=1):
        name = candidate.name or candidate.model_type
        logger.info(
            "Training candidate %d/%d: %s (%s)",
            index,
            len(candidates),
            name,
            candidate.model_type,
        )
        estimator = _build_pipeline(candidate)
        estimator.fit(
            data_split.train[feature_columns],
            data_split.train[TARGET_LABEL_COLUMN],
        )
        probabilities = predict_long_probabilities(
            estimator,
            data_split.validation[feature_columns],
        )
        predictions = (probabilities >= 0.5).astype("int64")
        metrics = classification_metrics(
            data_split.validation[TARGET_LABEL_COLUMN],
            predictions,
        )
        for entry_threshold in config.backtest.entry_thresholds:
            for exit_threshold in config.backtest.exit_thresholds:
                if entry_threshold <= exit_threshold:
                    continue
                for min_hold_bars in config.backtest.min_hold_bars_grid:
                    (
                        validation_backtest,
                        validation_score,
                        return_over_drawdown,
                        rejection_reasons,
                    ) = _score_validation_backtest(
                        probabilities=probabilities,
                        validation=data_split.validation,
                        config=config,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold,
                        min_hold_bars=min_hold_bars,
                    )
                    policy_name = (
                        f"{name}|entry={entry_threshold:g}|exit={exit_threshold:g}"
                        f"|hold={min_hold_bars}"
                    )
                    results.append(
                        CandidateTrainingResult(
                            name=policy_name,
                            model_type=candidate.model_type,
                            model_params=dict(candidate.model_params),
                            estimator=estimator,
                            entry_threshold=entry_threshold,
                            exit_threshold=exit_threshold,
                            min_hold_bars=min_hold_bars,
                            return_buffer=config.backtest.return_buffer,
                            validation_metrics=metrics,
                            validation_backtest_metrics=validation_backtest,
                            validation_score=validation_score,
                            return_over_drawdown=return_over_drawdown,
                            rejection_reasons=rejection_reasons,
                        )
                    )
        logger.info(
            "Trained candidate model %s and evaluated %d validation policies.",
            name,
            sum(
                1
                for entry_threshold in config.backtest.entry_thresholds
                for exit_threshold in config.backtest.exit_thresholds
                if entry_threshold > exit_threshold
            )
            * len(config.backtest.min_hold_bars_grid),
        )

    eligible_candidates = [
        result
        for result in results
        if math.isfinite(result.validation_score) and not result.rejection_reasons
    ]
    best = None
    if eligible_candidates:
        best = max(eligible_candidates, key=lambda result: result.validation_score)
        logger.info(
            "Selected model candidate: %s (%s=%.6f)",
            best.name,
            RETURN_OVER_DRAWDOWN_METRIC,
            best.validation_score,
        )
    else:
        logger.warning("No model candidate passed validation risk filters.")
    return ModelSelectionResult(
        best_candidate=best,
        candidates=results,
        feature_columns=feature_columns,
        selection_metric=RETURN_OVER_DRAWDOWN_METRIC,
    )


def _candidate_configs(config: TrainingConfig) -> list[ModelCandidateConfig]:
    if config.model_candidates is not None:
        return config.model_candidates
    return [
        ModelCandidateConfig(
            name=config.model_type,
            model_type=config.model_type,
            model_params=dict(config.model_params),
        )
    ]


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_columns = [
        column
        for column in df.columns
        if column not in _NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns found for model training.")
    return feature_columns


def _build_pipeline(candidate: ModelCandidateConfig) -> Pipeline:
    estimator = build_estimator(candidate.model_type, dict(candidate.model_params))
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if candidate.model_type == "LogisticRegression":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _score_validation_backtest(
    *,
    probabilities: object,
    validation: pd.DataFrame,
    config: TrainingConfig,
    entry_threshold: float,
    exit_threshold: float,
    min_hold_bars: int,
) -> tuple[dict[str, Any], float, float, list[str]]:
    try:
        metrics = backtest_metrics(
            probabilities=probabilities,
            future_returns=validation[TARGET_RETURN_COLUMN],
            transaction_fee=config.backtest.transaction_fee,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
            min_hold_bars=min_hold_bars,
            initial_position=config.backtest.initial_position,
            portfolio_mode=config.backtest.portfolio_mode,
        )
    except Exception as exc:
        logger.warning("Validation backtest failed; assigning -inf score: %s", exc)
        return (
            _failed_validation_backtest_metrics(str(exc)),
            -math.inf,
            -math.inf,
            ["validation_backtest_failed"],
        )

    if not has_only_finite_numbers(metrics):
        logger.warning("Validation backtest produced NaN/inf; assigning -inf score.")
        return metrics, -math.inf, -math.inf, ["validation_backtest_non_finite"]

    rejection_reasons = validation_risk_filter_rejection_reason(
        metrics,
        min_validation_cumulative_return=config.backtest.min_validation_cumulative_return,
        min_validation_exposure_ratio=config.backtest.min_validation_exposure_ratio,
        min_validation_traded_bars=config.backtest.min_validation_traded_bars,
        max_validation_drawdown=config.backtest.max_validation_drawdown,
        max_validation_turnover=config.backtest.max_validation_turnover,
    )
    return_over_drawdown = validation_return_over_drawdown_score(metrics)
    if rejection_reasons:
        logger.info(
            "Validation backtest rejected by filters %s; assigning -inf score.",
            ", ".join(rejection_reasons),
        )
        return metrics, -math.inf, return_over_drawdown, rejection_reasons

    score = validation_high_risk_score(metrics)
    return metrics, score, return_over_drawdown, []


def _failed_validation_backtest_metrics(error: str) -> dict[str, Any]:
    return {
        "transaction_fee": None,
        "bars": 0,
        "traded_bars": 0,
        "mean_strategy_return": None,
        "strategy_return_sum": None,
        "cumulative_return": 0.0,
        "benchmark_cumulative_return": None,
        "hit_rate": None,
        "max_drawdown": 0.0,
        "turnover": 0.0,
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "exposure_ratio": 0.0,
        "entry_signals": 0,
        "exit_signals": 0,
        "executed_position_changes": 0,
        "probability_diagnostics": {
            "average_probability_long": None,
            "histogram": {},
        },
        "error": error,
    }


def predict_long_probabilities(estimator: BaseEstimator, features: pd.DataFrame) -> pd.Series:
    """Return P(target_long=1) for binary long/cash classifiers."""
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)
        classes = list(getattr(estimator, "classes_", [0, 1]))
        class_index = classes.index(1) if 1 in classes else -1
        return pd.Series(probabilities[:, class_index], index=features.index)
    predictions = estimator.predict(features)
    probabilities = [1.0 if int(prediction) == 1 else 0.0 for prediction in predictions]
    return pd.Series(probabilities, index=features.index, dtype=float)
