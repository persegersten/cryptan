"""CLI for exporting a model bundle and publishing a GitHub Release."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config.loader import load_config
from src.exporting.release import publish_bundle_release
from src.pipeline.train_pipeline import run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/export a model bundle and publish it as a GitHub Release.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--local-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("dist/model_bundle"))
    parser.add_argument(
        "--repo",
        required=True,
        help="GitHub repository in owner/name form, e.g. persegersten/cryptan.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional release tag. Defaults to a manifest-derived model tag.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional release title. Defaults to the manifest model_id.",
    )
    parser.add_argument(
        "--prerelease",
        action="store_true",
        default=None,
        help="Force the GitHub Release to be marked as prerelease.",
    )
    parser.add_argument(
        "--stable",
        action="store_true",
        help="Force the GitHub Release to be marked as stable, not prerelease.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Export and validate, but do not call gh release create.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prerelease = args.prerelease
    if args.stable:
        prerelease = False

    try:
        config = load_config(args.config, local_path=args.local_config)
        training_artifact = run(config, export_bundle_output=args.output)
        if training_artifact.bundle is None:
            raise ValueError("Training completed without a bundle artifact.")
        release = publish_bundle_release(
            bundle_dir=training_artifact.bundle.bundle_dir,
            archive_path=training_artifact.bundle.archive_path,
            repo=args.repo,
            tag=args.tag,
            title=args.title,
            prerelease=prerelease,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, EnvironmentError, ValueError) as exc:
        logger.error("Failed to publish model release: %s", exc)
        sys.exit(1)

    print(training_artifact.bundle.bundle_dir)
    print(training_artifact.bundle.archive_path)
    print(f"release_tag={release.tag}")
    print(f"release_repo={release.repo}")
    print(f"prerelease={str(release.prerelease).lower()}")


if __name__ == "__main__":
    main()
