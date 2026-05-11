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
import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

from src.config.loader import load_config
from src.config.model import TrainingConfig
from src.features.builder import build_features
from src.ingestion.market_data import BinanceMarketDataSource
from src.labels.target import add_target_labels
from src.models import train_and_select_model
from src.preprocessing.cleaner import clean_market_data
from src.preprocessing.merger import merge_symbol_frames
from src.splitting.chronological import split_chronologically

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run(config: TrainingConfig) -> None:
    """Execute all pipeline steps for a single training run.

    Parameters
    ----------
    config:
        Validated training configuration.
    """
    logger.info("=== cryptan training pipeline start ===")
    logger.info(
        "Target: %s | Signals: %s | Timeframe: %s | %s → %s",
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
            config.model_selection_metric,
        )
    else:
        logger.info(
            "Model: %s | Params: %s | Selection metric: %s",
            config.model_type,
            config.model_params or "{}",
            config.model_selection_metric,
        )
    logger.info("Artifacts dir: %s", config.artifacts_dir)

    # ------------------------------------------------------------------
    # Step 2: Ingest historical OHLCV data for all signal symbols
    # ------------------------------------------------------------------
    source = BinanceMarketDataSource(
        api_key=config.data_api_key,
        api_secret=config.data_api_secret,
    )

    start_dt = datetime.datetime.fromisoformat(config.start_date).replace(
        tzinfo=datetime.timezone.utc
    )
    end_dt = datetime.datetime.fromisoformat(config.end_date).replace(
        tzinfo=datetime.timezone.utc
    )

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
    # Step 5: Create target labels for the configured trading symbol
    # ------------------------------------------------------------------
    logger.info("Creating target labels ...")
    labelled_df = add_target_labels(feature_df, config)
    logger.info(
        "Labelled DataFrame: %d rows × %d columns",
        len(labelled_df),
        len(labelled_df.columns),
    )

    # ------------------------------------------------------------------
    # Step 6: Split chronologically into train / validation / test
    # ------------------------------------------------------------------
    logger.info("Splitting labelled data chronologically ...")
    data_split = split_chronologically(labelled_df, config)
    logger.info(
        "Split sizes: train=%d, validation=%d, test=%d",
        len(data_split.train),
        len(data_split.validation),
        len(data_split.test),
    )

    # ------------------------------------------------------------------
    # Step 7: Train configured candidate model(s) and select the best one
    # ------------------------------------------------------------------
    logger.info("Training model candidate(s) ...")
    model_selection = train_and_select_model(data_split, config)
    logger.info(
        "Selected model: %s | Validation %s=%.6f | Features=%d",
        model_selection.best_candidate.name,
        model_selection.selection_metric,
        model_selection.best_candidate.validation_metrics[model_selection.selection_metric],
        len(model_selection.feature_columns),
    )

    # ------------------------------------------------------------------
    # TODO: wire in the remaining pipeline steps as they are implemented
    # ------------------------------------------------------------------
    # 8. Evaluate with ML metrics and simple backtest
    # 9. Save model artifact and run metadata
    # ------------------------------------------------------------------

    logger.info("=== cryptan training pipeline end ===")


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Load config and run the training pipeline."""
    args = parse_args(argv)

    try:
        config = load_config(args.config, local_path=args.local_config)
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    run(config)


if __name__ == "__main__":
    main()
