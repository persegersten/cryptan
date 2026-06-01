"""Tests for the GitHub Actions CI workflow contract."""

from __future__ import annotations

from pathlib import Path

import yaml


def _load_ci_workflow() -> dict[str, object]:
    workflow_path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"
    with workflow_path.open("r", encoding="utf-8") as file_handle:
        return yaml.load(file_handle, Loader=yaml.BaseLoader)


def test_manual_ci_workflow_trains_model_after_tests() -> None:
    workflow = _load_ci_workflow()

    jobs = workflow["jobs"]
    train_job = jobs["train-model"]

    assert train_job["if"] == "github.event_name == 'workflow_dispatch'"
    assert train_job["needs"] == "tests"
    assert train_job["env"]["CRYPTAN_DATA_API_KEY"] == "${{ secrets.CRYPTAN_DATA_API_KEY }}"
    assert (
        train_job["env"]["CRYPTAN_DATA_API_SECRET"]
        == "${{ secrets.CRYPTAN_DATA_API_SECRET }}"
    )

    run_commands = [
        step.get("run", "")
        for step in train_job["steps"]
        if isinstance(step, dict)
    ]
    assert any("python -m src.pipeline.train_pipeline" in command for command in run_commands)


def test_manual_ci_workflow_uploads_model_artifacts() -> None:
    workflow = _load_ci_workflow()

    train_steps = workflow["jobs"]["train-model"]["steps"]
    upload_steps = [
        step for step in train_steps if step.get("uses") == "actions/upload-artifact@v4"
    ]

    assert len(upload_steps) == 1
    upload_config = upload_steps[0]["with"]
    assert upload_config["name"] == "training-artifacts"
    assert "artifacts/" in upload_config["path"]
    assert "${{ inputs.export_bundle_output }}/" in upload_config["path"]
    assert upload_config["if-no-files-found"] == "error"
