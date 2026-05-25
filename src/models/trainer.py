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
from src.evaluation.metrics import backtest_metrics, classification_metrics
from src.evaluation.metrics import has_only_finite_numbers
from src.evaluation.metrics import validation_return_over_drawdown_score
from src.evaluation.metrics import validation_risk_filter_rejection_reason
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.registry import build_estimator
from src.splitting.chronological import ChronologicalSplit

logger = logging.getLogger(__name__)

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
    validation_metrics: dict[str, Any]
    validation_backtest_metrics: dict[str, Any]
    validation_score: float
    rejection_reason: str | None = None


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
            and candidate.rejection_reason is None
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
        predictions = estimator.predict(data_split.validation[feature_columns])
        metrics = classification_metrics(
            data_split.validation[TARGET_LABEL_COLUMN],
            predictions,
        )
        validation_backtest, validation_score, rejection_reason = _score_validation_backtest(
            predictions=predictions,
            validation=data_split.validation,
            transaction_fee=config.backtest.transaction_fee,
        )
        logger.info(
            "Candidate %s validation %s=%.6f",
            name,
            RETURN_OVER_DRAWDOWN_METRIC,
            validation_score,
        )
        results.append(
            CandidateTrainingResult(
                name=name,
                model_type=candidate.model_type,
                model_params=dict(candidate.model_params),
                estimator=estimator,
                validation_metrics=metrics,
                validation_backtest_metrics=validation_backtest,
                validation_score=validation_score,
                rejection_reason=rejection_reason,
            )
        )

    eligible_candidates = [
        result
        for result in results
        if math.isfinite(result.validation_score) and result.rejection_reason is None
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
    predictions: object,
    validation: pd.DataFrame,
    transaction_fee: float,
) -> tuple[dict[str, Any], float, str | None]:
    try:
        metrics = backtest_metrics(
            predictions=predictions,
            future_returns=validation[TARGET_RETURN_COLUMN],
            transaction_fee=transaction_fee,
        )
    except Exception as exc:
        logger.warning("Validation backtest failed; assigning -inf score: %s", exc)
        return (
            _failed_validation_backtest_metrics(str(exc)),
            -math.inf,
            "validation_backtest_failed",
        )

    if not has_only_finite_numbers(metrics):
        logger.warning("Validation backtest produced NaN/inf; assigning -inf score.")
        return metrics, -math.inf, "validation_backtest_non_finite"

    rejection_reason = validation_risk_filter_rejection_reason(metrics)
    if rejection_reason is not None:
        logger.info(
            "Validation backtest rejected by risk filter %s; assigning -inf score.",
            rejection_reason,
        )
        return metrics, -math.inf, rejection_reason

    score = validation_return_over_drawdown_score(metrics)
    return metrics, score, None


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
        "error": error,
    }
