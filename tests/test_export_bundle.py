"""Tests for production model bundle export."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.model import TrainingConfig
from src.evaluation import evaluate_and_save_report
from src.evaluation.metrics import RETURN_OVER_DRAWDOWN_METRIC
from src.exporting.bundle import export_model_bundle, load_persisted_model
from src.labels.target import TARGET_LABEL_COLUMN, TARGET_RETURN_COLUMN
from src.models.trainer import CandidateTrainingResult, ModelSelectionResult
from src.splitting.chronological import ChronologicalSplit


_API_KEY = "test-api-key-abc123"
_API_SECRET = "test-api-secret-xyz789"


def _make_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        trading_symbol="ETH",
        signal_symbols=["ETH", "BNB", "SOL"],
        timeframe="1h",
        start_date=-365,
        end_date=-1,
        prediction_horizon_bars=12,
        backtest={
            "transaction_fee": 0.001,
            "return_buffer": 0.005,
            "entry_thresholds": [0.55],
            "exit_thresholds": [0.45],
            "min_hold_bars_grid": [3],
            "min_validation_cumulative_return": -999.0,
            "min_validation_exposure_ratio": 0.0,
            "min_validation_traded_bars": 0,
        },
        artifacts_dir=tmp_path / "artifacts",
        data_api_key=_API_KEY,
        data_api_secret=_API_SECRET,
    )


def _frame(
    values: list[tuple[float, float]],
    labels: list[int],
    returns: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(values), freq="1h", tz="UTC"
            ),
            "signal_a": [row[0] for row in values],
            "ETH_close": [row[1] for row in values],
            TARGET_LABEL_COLUMN: labels,
            TARGET_RETURN_COLUMN: returns,
        }
    )


def _make_split() -> ChronologicalSplit:
    return ChronologicalSplit(
        train=_frame(
            [(-2.0, 100.0), (-1.0, 101.0), (1.0, 102.0), (2.0, 103.0)],
            [0, 0, 1, 1],
            [-0.02, -0.01, 0.03, 0.04],
        ),
        validation=_frame(
            [(-1.5, 104.0), (1.5, 105.0), (2.5, 106.0), (-2.5, 107.0)],
            [0, 1, 1, 0],
            [-0.03, 0.04, 0.02, -0.02],
        ),
        test=_frame(
            [(-3.0, 108.0), (3.0, 109.0), (1.0, 110.0), (-1.0, 111.0)],
            [0, 1, 1, 0],
            [-0.04, 0.05, 0.03, -0.01],
        ),
        raw_row_counts={"train": 6, "validation": 6, "test": 6},
    )


def _make_selection(split: ChronologicalSplit) -> ModelSelectionResult:
    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(random_state=42)),
        ]
    )
    feature_columns = ["signal_a", "ETH_close"]
    estimator.fit(split.train[feature_columns], split.train[TARGET_LABEL_COLUMN])
    candidate = CandidateTrainingResult(
        name="logistic|entry=0.55|exit=0.45|hold=3",
        model_type="LogisticRegression",
        model_params={"random_state": 42},
        estimator=estimator,
        entry_threshold=0.55,
        exit_threshold=0.45,
        min_hold_bars=3,
        return_buffer=0.005,
        validation_metrics={
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        validation_backtest_metrics={
            "cumulative_return": 0.12,
            "max_drawdown": -0.03,
            "turnover": 2.0,
            "traded_bars": 2,
            "exposure_ratio": 0.50,
            "hit_rate": 0.50,
            "entry_signals": 1,
            "exit_signals": 1,
        },
        validation_score=4.0,
        return_over_drawdown=4.0,
        rejection_reasons=[],
    )
    return ModelSelectionResult(
        best_candidate=candidate,
        candidates=[candidate],
        feature_columns=feature_columns,
        selection_metric=RETURN_OVER_DRAWDOWN_METRIC,
    )


def test_export_model_bundle_writes_expected_files_and_loadable_model(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    split = _make_split()
    selection = _make_selection(split)
    evaluation = evaluate_and_save_report(selection, split, config)

    artifact = export_model_bundle(
        model_selection=selection,
        data_split=split,
        config=config,
        evaluation=evaluation,
        output_dir=tmp_path / "dist" / "model_bundle",
    )

    assert artifact.bundle_dir.exists()
    assert artifact.archive_path == tmp_path / "dist" / "model_bundle.tgz"
    assert artifact.archive_path.exists()

    expected_names = {
        "manifest.json",
        "feature_schema.json",
        "strategy_config.json",
        "evaluation_report.json",
        "evaluation_report.html",
        "model_card.md",
        "sha256sums.txt",
    }
    actual_names = {path.name for path in artifact.bundle_dir.iterdir()}
    assert expected_names <= actual_names

    manifest = json.loads(
        (artifact.bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_task"] == "binary_long_cash"
    assert manifest["trading_symbol"] == "ETH"
    assert manifest["model_file"] in {"model.joblib", "model.skops"}
    assert manifest["evaluation_report_html_file"] == "evaluation_report.html"
    assert (artifact.bundle_dir / manifest["model_file"]).exists()

    feature_schema = json.loads(
        (artifact.bundle_dir / "feature_schema.json").read_text(encoding="utf-8")
    )
    assert feature_schema["feature_names"] == ["signal_a", "ETH_close"]
    assert feature_schema["required_input_timeframe"] == "1h"
    assert "schema_hash" in feature_schema

    strategy_config = json.loads(
        (artifact.bundle_dir / "strategy_config.json").read_text(encoding="utf-8")
    )
    assert strategy_config["portfolio_mode"] == "all_in_long_cash"
    assert strategy_config["sell_mode"] == "cash"
    assert strategy_config["shorting_enabled"] is False
    assert strategy_config["leverage"] == 1
    assert strategy_config["entry_threshold"] == 0.55
    assert strategy_config["exit_threshold"] == 0.45

    checksum_names = {
        line.split("  ", 1)[1]
        for line in (artifact.bundle_dir / "sha256sums.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert expected_names - {"sha256sums.txt"} <= checksum_names
    assert manifest["model_file"] in checksum_names
    assert "evaluation_report.html" in checksum_names

    loaded = load_persisted_model(artifact.bundle_dir / manifest["model_file"])
    sample = split.test[feature_schema["feature_names"]].head(2)
    probabilities = loaded.predict_proba(sample)
    assert probabilities.shape == (2, 2)


def test_loaded_model_rejects_non_matching_feature_order(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    split = _make_split()
    selection = _make_selection(split)
    evaluation = evaluate_and_save_report(selection, split, config)
    artifact = export_model_bundle(
        model_selection=selection,
        data_split=split,
        config=config,
        evaluation=evaluation,
        output_dir=tmp_path / "dist" / "model_bundle",
    )
    manifest = json.loads(
        (artifact.bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    loaded = load_persisted_model(artifact.bundle_dir / manifest["model_file"])

    with pytest.raises(ValueError, match="Feature columns do not match"):
        loaded.predict_proba(split.test[["ETH_close", "signal_a"]].head(1))
