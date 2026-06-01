"""Publish exported model bundles as versioned GitHub Releases."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReleasePublishResult:
    """Result from publishing a bundle as a GitHub Release."""

    repo: str
    tag: str
    release_title: str
    assets: list[Path]
    prerelease: bool


def publish_bundle_release(
    *,
    bundle_dir: Path,
    archive_path: Path,
    repo: str,
    tag: str | None = None,
    title: str | None = None,
    notes_path: Path | None = None,
    prerelease: bool | None = None,
    dry_run: bool = False,
) -> ReleasePublishResult:
    """Create a GitHub Release for a verified model bundle using ``gh``."""
    bundle_dir = Path(bundle_dir).resolve()
    archive_path = Path(archive_path).resolve()
    manifest = load_manifest(bundle_dir)
    verify_bundle_checksums(bundle_dir)

    release_tag = tag or release_tag_from_manifest(manifest)
    release_title = title or f"Model bundle {manifest['model_id']}"
    release_prerelease = (
        not bool(manifest.get("approved_for_live_pilot"))
        if prerelease is None
        else prerelease
    )
    notes_file = notes_path or write_release_notes(bundle_dir, manifest)
    assets = [
        archive_path,
        bundle_dir / "manifest.json",
        bundle_dir / "sha256sums.txt",
    ]
    _validate_release_assets(assets)

    command = [
        "gh",
        "release",
        "create",
        release_tag,
        *[str(asset) for asset in assets],
        "--repo",
        repo,
        "--title",
        release_title,
        "--notes-file",
        str(notes_file),
    ]
    if release_prerelease:
        command.append("--prerelease")

    if not dry_run:
        subprocess.run(command, check=True)

    return ReleasePublishResult(
        repo=repo,
        tag=release_tag,
        release_title=release_title,
        assets=assets,
        prerelease=release_prerelease,
    )


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Load the bundle manifest."""
    manifest_path = Path(bundle_dir) / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def release_tag_from_manifest(manifest: dict[str, Any]) -> str:
    """Build a stable release tag from manifest model identity."""
    model_id = _required_manifest_value(manifest, "model_id")
    trading_symbol = _required_manifest_value(manifest, "trading_symbol").lower()
    model_task = _required_manifest_value(manifest, "model_task").replace("_", "-")
    return f"model/{trading_symbol}-{model_task}/{model_id}"


def write_release_notes(bundle_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write Markdown release notes derived from manifest and evaluation report."""
    report_path = Path(bundle_dir) / "evaluation_report.json"
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    notes_path = Path(bundle_dir) / "release_notes.md"
    summary = report.get("executive_summary", {})
    backtest = report.get("backtest_metrics", {})
    lines = [
        f"# Model Bundle {manifest.get('model_id')}",
        "",
        f"- Task: {manifest.get('model_task')}",
        f"- Trading symbol: {manifest.get('trading_symbol')}",
        f"- Timeframe: {manifest.get('timeframe')}",
        f"- Experiment verdict: {manifest.get('experiment_verdict')}",
        f"- Approved for live pilot: {manifest.get('approved_for_live_pilot')}",
        f"- Validation period: {_format_period(manifest.get('validation_period'))}",
        f"- Test period: {_format_period(manifest.get('test_period'))}",
        f"- Test cumulative return: {backtest.get('cumulative_return')}",
        f"- Test max drawdown: {backtest.get('max_drawdown')}",
        f"- Selected model: {summary.get('selected_model')}",
        "",
        "Assets:",
        "- model_bundle.tgz",
        "- manifest.json",
        "- sha256sums.txt",
        "",
        "The archive contains `evaluation_report.html`.",
        "",
    ]
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return notes_path


def verify_bundle_checksums(bundle_dir: Path) -> None:
    """Verify every file listed in ``sha256sums.txt``."""
    bundle_dir = Path(bundle_dir).resolve()
    checksum_file = bundle_dir / "sha256sums.txt"
    if not checksum_file.exists():
        raise ValueError(f"Missing checksum file: {checksum_file}")

    seen: set[str] = set()
    for line_number, line in enumerate(
        checksum_file.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            expected, relative_name = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(
                f"Invalid checksum line {line_number} in {checksum_file}."
            ) from exc
        path = (bundle_dir / relative_name).resolve()
        _assert_inside(bundle_dir, path)
        if not path.exists() or not path.is_file():
            raise ValueError(f"Checksum references missing file: {relative_name}")
        actual = _file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {relative_name}: expected {expected}, got {actual}."
            )
        seen.add(relative_name)

    expected_files = {
        path.relative_to(bundle_dir).as_posix()
        for path in bundle_dir.rglob("*")
        if path.is_file()
        and path.name not in {"sha256sums.txt", "release_notes.md"}
    }
    missing = sorted(expected_files - seen)
    if missing:
        raise ValueError(f"sha256sums.txt is missing bundle file(s): {missing}")


def _validate_release_assets(assets: list[Path]) -> None:
    missing = [str(asset) for asset in assets if not asset.exists() or not asset.is_file()]
    if missing:
        raise ValueError(f"Release asset(s) missing: {missing}")


def _required_manifest_value(manifest: dict[str, Any], key: str) -> str:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field {key!r} is required for release publishing.")
    return value.strip()


def _format_period(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    return f"{value.get('start')} to {value.get('end')}"


def _assert_inside(root: Path, path: Path) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"Path {path} is outside bundle directory {root}.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
