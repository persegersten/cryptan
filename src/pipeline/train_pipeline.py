"""CLI entry point for the training pipeline.

Usage
-----
From the project root::

    python -m src.pipeline.train_pipeline --config config/training.yaml

    # With an optional local override file:
    python -m src.pipeline.train_pipeline \\
        --config config/training.yaml \\
        --local-config config/local.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.loader import load_config
from src.config.model import TrainingConfig
from src.evaluation import evaluate_and_save_report
from src.evaluation.report import EvaluationArtifact
from src.exporting import BundleArtifact, export_model_bundle
from src.evaluation.metrics import RETURN_OVER_DRAWDOWN_METRIC
from src.features.builder import build_features
from src.ingestion.market_data import BinanceMarketDataSource
from src.labels.target import add_target_labels
from src.models import train_and_select_model
from src.models.trainer import ModelSelectionResult
from src.preprocessing.cleaner import clean_market_data
from src.preprocessing.merger import merge_symbol_frames
from src.splitting.chronological import ChronologicalSplit, split_features_chronologically

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunArtifact:
    """Outputs from one completed training pipeline run."""

    model_selection: ModelSelectionResult
    data_split: ChronologicalSplit
    evaluation: EvaluationArtifact
    bundle: BundleArtifact | None


def run(
    config: TrainingConfig,
    *,
    export_bundle_output: Path | None = Path("dist/model_bundle"),
) -> TrainingRunArtifact:
    """Execute all pipeline steps for a single training run.

    Parameters
    ----------
    config:
        Validated training configuration.
    """
    logger.info("=== cryptan training pipeline start ===")
    logger.info(
        "Target: %s | Signals: %s | Timeframe: %s | Date offsets: %s → %s",
        config.trading_symbol,
        ", ".join(config.signal_symbols),
        config.timeframe,
        config.start_date,
        config.end_date,
    )
    if config.model_candidates:
        candidate_names = [
            candidate.name or candidate.model_type for candidate in config.model_candidates
        ]
        logger.info(
            "Model candidates: %s | Selection metric: %s",
            ", ".join(candidate_names),
            RETURN_OVER_DRAWDOWN_METRIC,
        )
    else:
        logger.info(
            "Model: %s | Params: %s | Selection metric: %s",
            config.model_type,
            config.model_params or "{}",
            RETURN_OVER_DRAWDOWN_METRIC,
        )
    logger.info("Artifacts dir: %s", config.artifacts_dir)

    # ------------------------------------------------------------------
    # Step 2: Ingest historical OHLCV data for all signal symbols
    # ------------------------------------------------------------------
    source = BinanceMarketDataSource(
        api_key=config.data_api_key,
        api_secret=config.data_api_secret,
    )

    start_dt = config.resolve_start_datetime()
    end_dt = config.resolve_end_datetime()

    raw_frames: dict[str, pd.DataFrame] = {}
    for symbol in config.signal_symbols:
        binance_symbol = f"{symbol}USDT"
        logger.info("Ingesting %s ...", binance_symbol)
        raw_frames[symbol] = source.fetch_ohlcv(
            symbol=binance_symbol,
            start=start_dt,
            end=end_dt,
            timeframe=config.timeframe,
        )
        logger.info("Ingested %d bars for %s", len(raw_frames[symbol]), symbol)

    # ------------------------------------------------------------------
    # Step 3: Preprocess each symbol frame and merge into one wide DataFrame
    # ------------------------------------------------------------------
    cleaned_frames: dict[str, pd.DataFrame] = {}
    for symbol, raw_df in raw_frames.items():
        logger.info("Cleaning %s ...", symbol)
        cleaned_frames[symbol] = clean_market_data(raw_df)
        logger.info("Cleaned %d bars for %s", len(cleaned_frames[symbol]), symbol)

    logger.info("Merging %d symbol frame(s) ...", len(cleaned_frames))
    merged_df = merge_symbol_frames(cleaned_frames)
    logger.info(
        "Merged DataFrame: %d rows × %d columns",
        len(merged_df),
        len(merged_df.columns),
    )

    # ------------------------------------------------------------------
    # Step 4: Build features for all configured signal symbols
    # ------------------------------------------------------------------
    logger.info("Building features ...")
    feature_df = build_features(merged_df, config)
    logger.info(
        "Feature DataFrame: %d rows × %d columns",
        len(feature_df),
        len(feature_df.columns),
    )

    # ------------------------------------------------------------------
    # Step 6: Split chronologically, then create split-local target labels
    # ------------------------------------------------------------------
    logger.info("Splitting feature data chronologically before target generation ...")
    raw_split = split_features_chronologically(feature_df, config)
    logger.info(
        "Raw split sizes: train=%d, validation=%d, test=%d",
        len(raw_split.train),
        len(raw_split.validation),
        len(raw_split.test),
    )

    logger.info("Creating split-local target labels ...")
    data_split = _label_split_partitions(raw_split, config)
    logger.info(
        "Labelled split sizes after horizon purge: train=%d, validation=%d, test=%d",
        len(data_split.train),
        len(data_split.validation),
        len(data_split.test),
    )

    # ------------------------------------------------------------------
    # Step 7: Train configured candidate model(s) and select the best one
    # ------------------------------------------------------------------
    logger.info("Training model candidate(s) ...")
    model_selection = train_and_select_model(data_split, config)
    if model_selection.best_candidate is None:
        logger.warning("No eligible model selected; trading evaluation will use cash baseline.")
    else:
        logger.info(
            "Selected model: %s | Validation %s=%.6f | Features=%d",
            model_selection.best_candidate.name,
            model_selection.selection_metric,
            model_selection.best_candidate.validation_score,
            len(model_selection.feature_columns),
        )

    # ------------------------------------------------------------------
    # Step 8: Evaluate with ML metrics and a simple backtest report
    # ------------------------------------------------------------------
    logger.info("Evaluating selected model on test split ...")
    evaluation = evaluate_and_save_report(model_selection, data_split, config)
    logger.info("Evaluation report saved to %s", evaluation.report_path)

    # ------------------------------------------------------------------
    # Step 9: Export production inference bundle for selected long/cash model
    # ------------------------------------------------------------------
    bundle = None
    if export_bundle_output is not None:
        logger.info("Exporting production model bundle to %s ...", export_bundle_output)
        bundle = export_model_bundle(
            model_selection=model_selection,
            data_split=data_split,
            config=config,
            evaluation=evaluation,
            output_dir=export_bundle_output,
        )
        logger.info("Model bundle exported to %s", bundle.bundle_dir)
        logger.info("Model bundle archive exported to %s", bundle.archive_path)

    logger.info("=== cryptan training pipeline end ===")
    return TrainingRunArtifact(
        model_selection=model_selection,
        data_split=data_split,
        evaluation=evaluation,
        bundle=bundle,
    )


def _label_split_partitions(
    raw_split: ChronologicalSplit,
    config: TrainingConfig,
) -> ChronologicalSplit:
    """Generate labels independently inside each chronological partition."""
    raw_row_counts = raw_split.row_counts
    labelled_split = ChronologicalSplit(
        train=add_target_labels(raw_split.train, config, allow_empty=True),
        validation=add_target_labels(raw_split.validation, config, allow_empty=True),
        test=add_target_labels(raw_split.test, config, allow_empty=True),
        raw_row_counts=raw_row_counts,
    )
    empty_partitions = [
        name for name, count in labelled_split.row_counts.items() if count == 0
    ]
    if empty_partitions:
        raise ValueError(
            "Split-local target generation produced empty labelled partition(s) "
            f"{empty_partitions}. Increase history, reduce prediction_horizon_bars, "
            "or adjust split fractions."
        )
    return labelled_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a crypto trading model using the cryptan pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to the base YAML config file (e.g. config/training.yaml).",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=None,
        metavar="PATH",
        dest="local_config",
        help=(
            "Optional path to a local YAML override file (e.g. config/local.yaml). "
            "Keys in this file are merged on top of the base config."
        ),
    )
    parser.add_argument(
        "--export-bundle-output",
        type=Path,
        default=Path("dist/model_bundle"),
        metavar="PATH",
        help=(
            "Output directory for the production model bundle. Use 'none' to disable."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load config and run the training pipeline."""
    args = parse_args(argv)

    try:
        config = load_config(args.config, local_path=args.local_config)
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    export_output = (
        None
        if str(args.export_bundle_output).strip().lower() == "none"
        else args.export_bundle_output
    )
    run(config, export_bundle_output=export_output)


if __name__ == "__main__":
    main()
