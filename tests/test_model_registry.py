"""Tests for supported model registry entries."""

from __future__ import annotations

import pytest
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from src.models.registry import build_estimator


@pytest.mark.parametrize(
    ("model_type", "expected_type"),
    [
        ("LogisticRegression", LogisticRegression),
        ("RandomForestClassifier", RandomForestClassifier),
        ("ExtraTreesClassifier", ExtraTreesClassifier),
        ("GradientBoostingClassifier", GradientBoostingClassifier),
        ("HistGradientBoostingClassifier", HistGradientBoostingClassifier),
    ],
)
def test_build_estimator_supports_registered_models(
    model_type: str,
    expected_type: type,
) -> None:
    estimator = build_estimator(model_type, {})

    assert isinstance(estimator, expected_type)


def test_logistic_regression_gets_safe_default_max_iter() -> None:
    estimator = build_estimator("LogisticRegression", {})

    assert isinstance(estimator, LogisticRegression)
    assert estimator.max_iter == 1000


def test_explicit_model_params_are_forwarded() -> None:
    estimator = build_estimator(
        "ExtraTreesClassifier",
        {"n_estimators": 17, "class_weight": "balanced", "random_state": 7},
    )

    assert isinstance(estimator, ExtraTreesClassifier)
    assert estimator.n_estimators == 17
    assert estimator.class_weight == "balanced"
    assert estimator.random_state == 7


def test_unknown_model_type_raises_with_supported_models() -> None:
    with pytest.raises(ValueError, match="ExtraTreesClassifier"):
        build_estimator("UnknownModel", {})
