"""Tests for publishing model bundles as GitHub Releases."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.exporting.release import (
    publish_bundle_release,
    release_tag_from_manifest,
    verify_bundle_checksums,
)


def _make_bundle(path: Path) -> Path:
    path.mkdir(parents=True)
    files = {
        "manifest.json": (
            "{"
            '"model_id": "eth-binary-long-cash-1h-abc123", '
            '"model_task": "binary_long_cash", '
            '"trading_symbol": "ETH", '
            '"timeframe": "1h", '
            '"approved_for_live_pilot": false, '
            '"experiment_verdict": "PAPER_TRADE_ONLY", '
            '"validation_period": {"start": "2024-01-01", "end": "2024-02-01"}, '
            '"test_period": {"start": "2024-02-02", "end": "2024-03-01"}'
            "}\n"
        ),
        "feature_schema.json": "{}\n",
        "strategy_config.json": "{}\n",
        "evaluation_report.json": (
            "{"
            '"executive_summary": {"selected_model": "logistic"}, '
            '"backtest_metrics": {"cumulative_return": 0.12, "max_drawdown": -0.04}'
            "}\n"
        ),
        "evaluation_report.html": "<html></html>\n",
        "model_card.md": "# Card\n",
        "model.joblib": "model-bytes\n",
    }
    for relative_name, content in files.items():
        (path / relative_name).write_text(content, encoding="utf-8")

    rows = []
    for relative_name in sorted(files):
        digest = hashlib.sha256((path / relative_name).read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative_name}")
    (path / "sha256sums.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_release_tag_from_manifest_uses_model_identity() -> None:
    tag = release_tag_from_manifest(
        {
            "model_id": "eth-binary-long-cash-1h-abc123",
            "trading_symbol": "ETH",
            "model_task": "binary_long_cash",
        }
    )

    assert tag == "model/eth-binary-long-cash/eth-binary-long-cash-1h-abc123"


def test_publish_bundle_release_invokes_gh_release_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(tmp_path / "model_bundle")
    archive = tmp_path / "model_bundle.tgz"
    archive.write_text("archive\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        calls.append(command)
        assert check is True

    monkeypatch.setattr("src.exporting.release.subprocess.run", fake_run)

    result = publish_bundle_release(
        bundle_dir=bundle,
        archive_path=archive,
        repo="persegersten/cryptan",
    )

    assert result.repo == "persegersten/cryptan"
    assert result.tag == "model/eth-binary-long-cash/eth-binary-long-cash-1h-abc123"
    assert result.prerelease is True
    assert result.assets == [archive, bundle / "manifest.json", bundle / "sha256sums.txt"]
    assert calls == [
        [
            "gh",
            "release",
            "create",
            result.tag,
            str(archive),
            str(bundle / "manifest.json"),
            str(bundle / "sha256sums.txt"),
            "--repo",
            "persegersten/cryptan",
            "--title",
            "Model bundle eth-binary-long-cash-1h-abc123",
            "--notes-file",
            str(bundle / "release_notes.md"),
            "--prerelease",
        ]
    ]
    assert "evaluation_report.html" in (
        bundle / "release_notes.md"
    ).read_text(encoding="utf-8")


def test_publish_bundle_release_can_force_stable_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _make_bundle(tmp_path / "model_bundle")
    archive = tmp_path / "model_bundle.tgz"
    archive.write_text("archive\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "src.exporting.release.subprocess.run",
        lambda command, check: calls.append(command),
    )

    result = publish_bundle_release(
        bundle_dir=bundle,
        archive_path=archive,
        repo="persegersten/cryptan",
        prerelease=False,
    )

    assert result.prerelease is False
    assert "--prerelease" not in calls[0]


def test_verify_bundle_checksums_rejects_modified_file(tmp_path: Path) -> None:
    bundle = _make_bundle(tmp_path / "model_bundle")
    (bundle / "manifest.json").write_text('{"changed": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_bundle_checksums(bundle)
