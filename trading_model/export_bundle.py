"""CLI for training and exporting a production model bundle."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.loader import load_config
from src.pipeline.train_pipeline import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and export a production cryptan model bundle.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML training config.",
    )
    parser.add_argument(
        "--local-config",
        type=Path,
        default=None,
        help="Optional local YAML override file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dist/model_bundle"),
        help="Output directory for the model bundle.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        config = load_config(args.config, local_path=args.local_config)
        artifact = run(config, export_bundle_output=args.output)
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("Failed to export model bundle: %s", exc)
        sys.exit(1)

    if artifact.bundle is None:
        logger.error("Training completed but bundle export was disabled.")
        sys.exit(1)
    print(artifact.bundle.bundle_dir)
    print(artifact.bundle.archive_path)


if __name__ == "__main__":
    main()
