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
from pathlib import Path

from src.config.loader import load_config
from src.config.model import TrainingConfig

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
    logger.info("Model: %s | Params: %s", config.model_type, config.model_params or "{}")
    logger.info("Artifacts dir: %s", config.artifacts_dir)

    # ------------------------------------------------------------------
    # TODO: wire in the remaining pipeline steps as they are implemented
    # ------------------------------------------------------------------
    # 1. Ingest historical OHLCV data for all signal symbols
    # 2. Preprocess and merge symbol frames
    # 3. Build features
    # 4. Create target labels for the trading symbol
    # 5. Split chronologically (train / validation / test)
    # 6. Train the configured model
    # 7. Evaluate with ML metrics and simple backtest
    # 8. Save model artifact and run metadata
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
