"""Train configured model candidates and select the best validation performer."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.model import ModelCandidateConfig, TrainingConfig
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
    validation_metrics: dict[str, float]


@dataclass(frozen=True)
class ModelSelectionResult:
    """Best model plus all candidate validation scores."""

    best_candidate: CandidateTrainingResult
    candidates: list[CandidateTrainingResult]
    feature_columns: list[str]
    selection_metric: str

    @property
    def estimator(self) -> BaseEstimator:
        """Return the selected fitted estimator."""
        return self.best_candidate.estimator


def train_and_select_model(
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> ModelSelectionResult:
    """Train configured candidates and select the best validation model.

    The function fits each candidate on the chronological train partition and
    scores it on the validation partition. It does not use test rows.
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
        metrics = _classification_metrics(
            data_split.validation[TARGET_LABEL_COLUMN],
            predictions,
        )
        logger.info(
            "Candidate %s validation %s=%.6f",
            name,
            config.model_selection_metric,
            _metric_value(metrics, config.model_selection_metric),
        )
        results.append(
            CandidateTrainingResult(
                name=name,
                model_type=candidate.model_type,
                model_params=dict(candidate.model_params),
                estimator=estimator,
                validation_metrics=metrics,
            )
        )

    best = max(
        results,
        key=lambda result: _metric_value(
            result.validation_metrics,
            config.model_selection_metric,
        ),
    )
    logger.info(
        "Selected model candidate: %s (%s=%.6f)",
        best.name,
        config.model_selection_metric,
        _metric_value(best.validation_metrics, config.model_selection_metric),
    )
    return ModelSelectionResult(
        best_candidate=best,
        candidates=results,
        feature_columns=feature_columns,
        selection_metric=config.model_selection_metric,
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


def _classification_metrics(
    y_true: pd.Series,
    y_pred: object,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _metric_value(metrics: dict[str, float], metric_name: str) -> float:
    try:
        return metrics[metric_name]
    except KeyError as exc:
        supported = ", ".join(sorted(metrics))
        raise ValueError(
            f"Unsupported model_selection_metric {metric_name!r}. "
            f"Supported metrics: {supported}."
        ) from exc
