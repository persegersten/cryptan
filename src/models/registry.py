"""Small registry of supported baseline classifiers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_estimator(model_type: str, model_params: dict[str, Any]) -> object:
    """Instantiate a configured classifier from the model registry.

    Raises
    ------
    ValueError
        If *model_type* is not registered.
    """
    registry = {
        "LogisticRegression": LogisticRegression,
        "RandomForestClassifier": RandomForestClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    }
    try:
        model_cls = registry[model_type]
    except KeyError as exc:
        supported = ", ".join(sorted(registry))
        raise ValueError(
            f"Unsupported model_type {model_type!r}. Supported models: {supported}."
        ) from exc

    params = dict(model_params)
    if model_type == "LogisticRegression":
        params.setdefault("max_iter", 1000)
    return model_cls(**params)
