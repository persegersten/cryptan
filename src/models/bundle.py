"""Serializable production inference wrapper for trained sklearn pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class FeatureOrderedModel:
    """Wrap a fitted estimator and enforce the training feature schema."""

    estimator: BaseEstimator
    feature_names: list[str]

    def predict(self, features: pd.DataFrame) -> Any:
        """Validate and predict with features in the selected training order."""
        return self.estimator.predict(self._ordered_frame(features))

    def predict_proba(self, features: pd.DataFrame) -> Any:
        """Validate and return class probabilities."""
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError("Wrapped estimator does not support predict_proba.")
        return self.estimator.predict_proba(self._ordered_frame(features))

    @property
    def classes_(self) -> Any:
        """Expose sklearn classes from the fitted pipeline."""
        return getattr(self.estimator, "classes_")

    def _ordered_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")
        actual = list(features.columns)
        if actual != self.feature_names:
            raise ValueError(
                "Feature columns do not match the model schema exactly. "
                f"Expected {self.feature_names}; got {actual}."
            )
        return features.loc[:, self.feature_names]
