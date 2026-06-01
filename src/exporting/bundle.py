"""Export a self-contained production model bundle."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config.model import TrainingConfig
from src.evaluation.metrics import EXECUTION_LAG_BARS, SELL_MODE
from src.evaluation.report import EvaluationArtifact
from src.models.bundle import FeatureOrderedModel
from src.models.trainer import ModelSelectionResult
from src.splitting.chronological import ChronologicalSplit

PACKAGE_FORMAT_VERSION = "1.0"
FEATURE_VERSION = "technical_v1"


@dataclass(frozen=True)
class BundleArtifact:
    """Paths produced by a bundle export."""

    bundle_dir: Path
    archive_path: Path
    model_file: Path


def export_model_bundle(
    *,
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
    evaluation: EvaluationArtifact,
    output_dir: Path,
) -> BundleArtifact:
    """Write a portable inference bundle and compressed archive."""
    if model_selection.best_candidate is None:
        raise ValueError("Cannot export a production bundle without a selected model.")

    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created_at = _utc_now().isoformat()
    model_id = _model_id(config, created_at)
    feature_schema = _feature_schema(model_selection, data_split, config)
    model_file = output_dir / _model_filename()
    _dump_model(
        FeatureOrderedModel(
            estimator=model_selection.best_candidate.estimator,
            feature_names=list(model_selection.feature_columns),
        ),
        model_file,
    )

    evaluation_json = output_dir / "evaluation_report.json"
    evaluation_html = output_dir / "evaluation_report.html"
    shutil.copy2(evaluation.report_path, evaluation_json)
    shutil.copy2(evaluation.html_report_path, evaluation_html)

    _write_json(output_dir / "feature_schema.json", feature_schema)
    _write_json(output_dir / "strategy_config.json", _strategy_config(model_selection, config))
    _write_json(
        output_dir / "manifest.json",
        _manifest(
            model_id=model_id,
            created_at=created_at,
            model_file=model_file.name,
            model_selection=model_selection,
            data_split=data_split,
            config=config,
            evaluation=evaluation,
        ),
    )
    (output_dir / "model_card.md").write_text(
        _model_card(model_id, model_selection, config, evaluation.report),
        encoding="utf-8",
    )
    _write_sha256sums(output_dir)
    archive_path = _write_archive(output_dir)
    return BundleArtifact(
        bundle_dir=output_dir,
        archive_path=archive_path,
        model_file=model_file,
    )


def load_persisted_model(path: Path) -> FeatureOrderedModel:
    """Load a persisted model bundle object saved by ``export_model_bundle``."""
    path = Path(path)
    if path.suffix == ".skops":
        try:
            import skops.io as sio  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("skops is required to load .skops model files.") from exc
        return sio.load(path, trusted=True)
    return joblib.load(path)


def _dump_model(model: FeatureOrderedModel, path: Path) -> None:
    if path.suffix == ".skops":
        import skops.io as sio  # type: ignore[import-not-found]

        sio.dump(model, path)
        return
    joblib.dump(model, path)


def _model_filename() -> str:
    try:
        import skops.io  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return "model.joblib"
    return "model.skops"


def _manifest(
    *,
    model_id: str,
    created_at: str,
    model_file: str,
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
    evaluation: EvaluationArtifact,
) -> dict[str, Any]:
    best = model_selection.best_candidate
    if best is None:
        raise ValueError("Cannot build manifest without selected model.")
    report = evaluation.report
    verdict = report.get("experiment_verdict", {})
    return {
        "model_id": model_id,
        "created_at_utc": created_at,
        "git_commit": _git_commit(),
        "model_task": "binary_long_cash",
        "trading_symbol": config.trading_symbol,
        "signal_symbols": config.signal_symbols,
        "timeframe": config.timeframe,
        "prediction_horizon_bars": config.prediction_horizon_bars,
        "return_buffer": best.return_buffer,
        "model_type": best.model_type,
        "model_file": model_file,
        "feature_schema_file": "feature_schema.json",
        "strategy_config_file": "strategy_config.json",
        "evaluation_report_file": "evaluation_report.json",
        "approved_for_live_pilot": bool(verdict.get("live_pilot_allowed", False)),
        "experiment_verdict": verdict.get("status"),
        "training_row_counts": {
            "train_raw": len(data_split.train)
            if data_split.raw_row_counts is None
            else data_split.raw_row_counts["train"],
            "train_labelled": len(data_split.train),
            "validation_raw": len(data_split.validation)
            if data_split.raw_row_counts is None
            else data_split.raw_row_counts["validation"],
            "validation_labelled": len(data_split.validation),
            "test_raw": len(data_split.test)
            if data_split.raw_row_counts is None
            else data_split.raw_row_counts["test"],
            "test_labelled": len(data_split.test),
        },
        "validation_period": _period(data_split.validation),
        "test_period": _period(data_split.test),
        "package_format_version": PACKAGE_FORMAT_VERSION,
    }


def _feature_schema(
    model_selection: ModelSelectionResult,
    data_split: ChronologicalSplit,
    config: TrainingConfig,
) -> dict[str, Any]:
    feature_names = list(model_selection.feature_columns)
    dtypes = {
        name: str(data_split.train[name].dtype)
        for name in feature_names
        if name in data_split.train.columns
    }
    schema = {
        "feature_version": FEATURE_VERSION,
        "required_input_timeframe": config.timeframe,
        "required_history_bars": _required_history_bars(config),
        "feature_names": feature_names,
        "feature_dtypes": dtypes,
        "sort_order_requirement": "timestamp ascending, UTC, one row per bar",
        "missing_value_policy": (
            "Input columns must match feature_names exactly. The persisted sklearn "
            "pipeline applies its fitted imputer to missing numeric values."
        ),
    }
    schema["schema_hash"] = _payload_hash(schema)
    return schema


def _strategy_config(
    model_selection: ModelSelectionResult,
    config: TrainingConfig,
) -> dict[str, Any]:
    best = model_selection.best_candidate
    if best is None:
        raise ValueError("Cannot build strategy config without selected model.")
    return {
        "portfolio_mode": "all_in_long_cash",
        "sell_mode": SELL_MODE,
        "shorting_enabled": False,
        "leverage": 1,
        "hold_behavior": "keep_previous_position",
        "initial_position": config.backtest.initial_position,
        "entry_threshold": best.entry_threshold,
        "exit_threshold": best.exit_threshold,
        "min_hold_bars": best.min_hold_bars,
        "execution_lag_bars": EXECUTION_LAG_BARS,
        "transaction_fee": config.backtest.transaction_fee,
        "max_live_allocation_quote": None,
        "warning_drawdown": -0.40,
        "kill_switch_drawdown": config.backtest.max_validation_drawdown,
    }


def _model_card(
    model_id: str,
    model_selection: ModelSelectionResult,
    config: TrainingConfig,
    report: dict[str, Any],
) -> str:
    selected = report.get("run_metadata", {}).get("selected_model") or {}
    validation = {
        "cumulative_return": selected.get("validation_cumulative_return"),
        "max_drawdown": selected.get("validation_max_drawdown"),
        "turnover": selected.get("validation_turnover"),
        "return_over_drawdown": selected.get("return_over_drawdown"),
    }
    test = report.get("backtest_metrics", {})
    ml = report.get("ml_metrics") or {}
    features = model_selection.feature_columns
    feature_summary = ", ".join(features[:20])
    if len(features) > 20:
        feature_summary += f", ... ({len(features)} total)"
    return "\n".join(
        [
            f"# Model Card: {model_id}",
            "",
            "## What The Model Does",
            (
                f"Binary long/cash classifier for {config.trading_symbol}. It returns "
                "the probability that the next configured horizon is favorable for "
                "being long instead of cash."
            ),
            "",
            "## Intended Use",
            "Tiny monitored live pilot only, using hourly stateless inference.",
            "",
            "## Not Intended For",
            "Leverage, shorting, large capital, or unmonitored automated trading.",
            "",
            "## Latest Validation Metrics",
            _markdown_json(validation),
            "",
            "## Latest Test Metrics",
            _markdown_json(
                {
                    "accuracy": ml.get("accuracy"),
                    "precision": ml.get("precision"),
                    "recall": ml.get("recall"),
                    "f1": ml.get("f1"),
                    "cumulative_return": test.get("cumulative_return"),
                    "benchmark_cumulative_return": test.get(
                        "benchmark_cumulative_return"
                    ),
                    "hit_rate": test.get("hit_rate"),
                    "max_drawdown": test.get("max_drawdown"),
                    "turnover": test.get("turnover"),
                }
            ),
            "",
            "## Main Risks",
            "- Financial time-series drift can invalidate historical performance.",
            "- Threshold behavior can be sensitive to probability calibration.",
            "- Exchange fees, slippage, outages, and delayed execution are not fully modeled.",
            "",
            "## Known Limitations",
            "- Offline historical evaluation only.",
            "- No live order execution or portfolio reconciliation is included.",
            "- Raw OHLCV feature generation must be reproduced before inference.",
            "",
            "## Feature List Summary",
            feature_summary,
            "",
            "## How To Reproduce Training",
            (
                "Run `python -m trading_model.export_bundle --config "
                "configs/eth_long_cash.yml --output dist/model_bundle` with the "
                "required data API environment variables set."
            ),
            "",
        ]
    )


def _required_history_bars(config: TrainingConfig) -> int:
    feature_config = config.feature_config
    windows = [
        *feature_config.return_windows,
        feature_config.ma_short_window,
        feature_config.ma_long_window,
        feature_config.volatility_window,
        feature_config.volume_window,
        feature_config.correlation_window,
    ]
    return int(max(windows))


def _period(frame: pd.DataFrame) -> dict[str, str | None]:
    if frame.empty or "timestamp" not in frame:
        return {"start": None, "end": None}
    return {
        "start": _timestamp_to_iso(frame["timestamp"].iloc[0]),
        "end": _timestamp_to_iso(frame["timestamp"].iloc[-1]),
    }


def _timestamp_to_iso(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_sha256sums(bundle_dir: Path) -> None:
    rows = []
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.name == "sha256sums.txt":
            continue
        rows.append(f"{_file_sha256(path)}  {path.name}")
    (bundle_dir / "sha256sums.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_archive(bundle_dir: Path) -> Path:
    archive_path = bundle_dir.with_suffix(".tgz")
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
            archive.add(path, arcname=f"{bundle_dir.name}/{path.name}")
    return archive_path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _model_id(config: TrainingConfig, created_at: str) -> str:
    suffix = hashlib.sha256(created_at.encode("utf-8")).hexdigest()[:10]
    return (
        f"{config.trading_symbol.lower()}-{config.model_task}-"
        f"{config.timeframe}-{suffix}"
    )


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _markdown_json(payload: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```"
