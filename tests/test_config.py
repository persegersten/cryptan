"""Tests for config loading and validation (Step 1 of the MVP)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.config.loader import load_config
from src.config.model import SplitConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_yaml(tmp_path: Path, content: dict) -> Path:
    """Write a dict as YAML to a temp file and return the path."""
    config_file = tmp_path / "training.yaml"
    config_file.write_text(yaml.dump(content), encoding="utf-8")
    return config_file


MINIMAL_VALID = {
    "trading_symbol": "ETH",
    "signal_symbols": ["ETH", "BNB", "SOL"],
    "timeframe": "1h",
    "start_date": "2022-01-01",
    "end_date": "2024-01-01",
}

# Real (non-placeholder) credential values used throughout the test suite.
_TEST_API_KEY = "test-api-key-abc123"
_TEST_API_SECRET = "test-api-secret-xyz789"


# ---------------------------------------------------------------------------
# Autouse fixture — supply required env vars for every test in this module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _patch_required_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject valid credentials so all tests can call load_config freely."""
    monkeypatch.setenv("CRYPTAN_DATA_API_KEY", _TEST_API_KEY)
    monkeypatch.setenv("CRYPTAN_DATA_API_SECRET", _TEST_API_SECRET)


# ---------------------------------------------------------------------------
# load_config — file system behaviour
# ---------------------------------------------------------------------------

class TestLoadConfigFileSystem:
    def test_loads_existing_yaml_file(self, tmp_path: Path) -> None:
        config_file = write_yaml(tmp_path, MINIMAL_VALID)
        config = load_config(config_file)
        assert isinstance(config, TrainingConfig)

    def test_raises_file_not_found_for_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        config_file = write_yaml(tmp_path, MINIMAL_VALID)
        config = load_config(str(config_file))
        assert isinstance(config, TrainingConfig)

    def test_raises_value_error_for_non_mapping_yaml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_config(bad_file)


# ---------------------------------------------------------------------------
# load_config — default training.yaml in the repo
# ---------------------------------------------------------------------------

class TestDefaultConfigFile:
    """The shipped config/training.yaml must load without errors."""

    def test_default_config_loads(self) -> None:
        repo_root = Path(__file__).parent.parent
        config = load_config(repo_root / "config" / "training.yaml")
        assert config.trading_symbol == "ETH"
        assert "ETH" in config.signal_symbols
        assert "BNB" in config.signal_symbols
        assert "SOL" in config.signal_symbols

    def test_default_config_split_sums_to_one(self) -> None:
        repo_root = Path(__file__).parent.parent
        config = load_config(repo_root / "config" / "training.yaml")
        total = config.split.train + config.split.validation + config.split.test
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Environment variable and credentials handling
# ---------------------------------------------------------------------------

class TestEnvVarCredentials:
    def test_credentials_injected_from_env_vars(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert config.data_api_key == _TEST_API_KEY
        assert config.data_api_secret == _TEST_API_SECRET

    def test_missing_api_key_raises_environment_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CRYPTAN_DATA_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="CRYPTAN_DATA_API_KEY"):
            load_config(write_yaml(tmp_path, MINIMAL_VALID))

    def test_missing_api_secret_raises_environment_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CRYPTAN_DATA_API_SECRET", raising=False)
        with pytest.raises(EnvironmentError, match="CRYPTAN_DATA_API_SECRET"):
            load_config(write_yaml(tmp_path, MINIMAL_VALID))

    def test_placeholder_api_key_raises_environment_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRYPTAN_DATA_API_KEY", "changeme")
        with pytest.raises(EnvironmentError, match="changeme"):
            load_config(write_yaml(tmp_path, MINIMAL_VALID))

    def test_placeholder_api_secret_raises_environment_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRYPTAN_DATA_API_SECRET", "changeme")
        with pytest.raises(EnvironmentError, match="changeme"):
            load_config(write_yaml(tmp_path, MINIMAL_VALID))

    def test_empty_api_key_raises_environment_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CRYPTAN_DATA_API_KEY", "")
        with pytest.raises(EnvironmentError, match="CRYPTAN_DATA_API_KEY"):
            load_config(write_yaml(tmp_path, MINIMAL_VALID))


# ---------------------------------------------------------------------------
# load_config — local YAML overlay
# ---------------------------------------------------------------------------

class TestLocalOverlay:
    def test_local_overlay_overrides_base_value(self, tmp_path: Path) -> None:
        base = write_yaml(tmp_path, MINIMAL_VALID)
        local = tmp_path / "local.yaml"
        local.write_text(yaml.dump({"start_date": "2023-01-01"}), encoding="utf-8")
        config = load_config(base, local_path=local)
        assert config.start_date == "2023-01-01"

    def test_local_overlay_missing_file_is_ignored(self, tmp_path: Path) -> None:
        base = write_yaml(tmp_path, MINIMAL_VALID)
        config = load_config(base, local_path=tmp_path / "nonexistent_local.yaml")
        assert config.start_date == MINIMAL_VALID["start_date"]

    def test_no_local_path_argument_uses_base_only(self, tmp_path: Path) -> None:
        base = write_yaml(tmp_path, MINIMAL_VALID)
        config = load_config(base)
        assert config.start_date == MINIMAL_VALID["start_date"]

    def test_local_overlay_cannot_inject_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Credentials in a local YAML file are overwritten by env vars."""
        base = write_yaml(tmp_path, MINIMAL_VALID)
        local = tmp_path / "local.yaml"
        local.write_text(
            yaml.dump({"data_api_key": "from-yaml-should-be-overwritten"}),
            encoding="utf-8",
        )
        config = load_config(base, local_path=local)
        # Env var must win over anything in a YAML file.
        assert config.data_api_key == _TEST_API_KEY


# ---------------------------------------------------------------------------
# TrainingConfig — symbol validation
# ---------------------------------------------------------------------------

class TestTradingSymbolValidation:
    def test_trading_symbol_is_uppercased(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "trading_symbol": "eth"}
        config = load_config(write_yaml(tmp_path, data))
        assert config.trading_symbol == "ETH"

    def test_trading_symbol_whitespace_stripped(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "trading_symbol": "  BTC  "}
        config = load_config(write_yaml(tmp_path, data))
        assert config.trading_symbol == "BTC"

    def test_empty_trading_symbol_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "trading_symbol": ""}
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, data))


class TestSignalSymbolsValidation:
    def test_signal_symbols_are_uppercased(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "signal_symbols": ["eth", "bnb"]}
        config = load_config(write_yaml(tmp_path, data))
        assert config.signal_symbols == ["ETH", "BNB"]

    def test_signal_symbols_whitespace_stripped(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "signal_symbols": [" ETH ", " SOL "]}
        config = load_config(write_yaml(tmp_path, data))
        assert config.signal_symbols == ["ETH", "SOL"]

    def test_single_signal_symbol_is_valid(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "signal_symbols": ["ETH"]}
        config = load_config(write_yaml(tmp_path, data))
        assert config.signal_symbols == ["ETH"]

    def test_empty_signal_symbols_list_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "signal_symbols": []}
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, data))

    def test_missing_signal_symbols_key_raises(self, tmp_path: Path) -> None:
        data = {k: v for k, v in MINIMAL_VALID.items() if k != "signal_symbols"}
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, data))

    def test_trading_symbol_need_not_be_in_signal_symbols(self, tmp_path: Path) -> None:
        """Trading symbol and signal symbols are independent configuration choices."""
        data = {**MINIMAL_VALID, "trading_symbol": "BTC", "signal_symbols": ["ETH", "BNB"]}
        config = load_config(write_yaml(tmp_path, data))
        assert config.trading_symbol == "BTC"
        assert "BTC" not in config.signal_symbols


# ---------------------------------------------------------------------------
# TrainingConfig — date validation
# ---------------------------------------------------------------------------

class TestDateValidation:
    def test_valid_date_range_loads(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert config.start_date == "2022-01-01"
        assert config.end_date == "2024-01-01"

    def test_end_date_before_start_date_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "start_date": "2024-01-01", "end_date": "2022-01-01"}
        with pytest.raises(ValueError, match="end_date"):
            load_config(write_yaml(tmp_path, data))

    def test_same_start_and_end_date_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "start_date": "2023-01-01", "end_date": "2023-01-01"}
        with pytest.raises(ValueError, match="end_date"):
            load_config(write_yaml(tmp_path, data))

    def test_invalid_date_format_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "start_date": "01/01/2022"}
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, data))


# ---------------------------------------------------------------------------
# TrainingConfig — labelling and horizon
# ---------------------------------------------------------------------------

class TestLabellingParams:
    def test_default_prediction_horizon(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert config.prediction_horizon_bars == 12

    def test_default_return_threshold(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert config.return_threshold == 0.01

    def test_custom_horizon(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "prediction_horizon_bars": 6}
        config = load_config(write_yaml(tmp_path, data))
        assert config.prediction_horizon_bars == 6

    def test_zero_horizon_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "prediction_horizon_bars": 0}
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, data))

    def test_negative_horizon_raises(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "prediction_horizon_bars": -1}
        with pytest.raises(ValueError):
            load_config(write_yaml(tmp_path, data))


# ---------------------------------------------------------------------------
# SplitConfig — fraction validation
# ---------------------------------------------------------------------------

class TestSplitConfig:
    def test_valid_split_fractions(self) -> None:
        split = SplitConfig(train=0.70, validation=0.15, test=0.15)
        assert abs(split.train + split.validation + split.test - 1.0) < 1e-6

    def test_fractions_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train=0.60, validation=0.20, test=0.10)

    def test_default_split_sums_to_one(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        total = config.split.train + config.split.validation + config.split.test
        assert abs(total - 1.0) < 1e-6

    def test_custom_split_via_config(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "split": {"train": 0.80, "validation": 0.10, "test": 0.10}}
        config = load_config(write_yaml(tmp_path, data))
        assert config.split.train == 0.80


# ---------------------------------------------------------------------------
# TrainingConfig — artifact dir
# ---------------------------------------------------------------------------

class TestArtifactsDir:
    def test_default_artifacts_dir_is_path(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert isinstance(config.artifacts_dir, Path)
        assert config.artifacts_dir == Path("artifacts")

    def test_custom_artifacts_dir(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "artifacts_dir": "/tmp/runs"}
        config = load_config(write_yaml(tmp_path, data))
        assert config.artifacts_dir == Path("/tmp/runs")


# ---------------------------------------------------------------------------
# TrainingConfig — model selection
# ---------------------------------------------------------------------------

class TestModelConfig:
    def test_default_model_type(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert config.model_type == "RandomForestClassifier"

    def test_custom_model_type(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "model_type": "LogisticRegression"}
        config = load_config(write_yaml(tmp_path, data))
        assert config.model_type == "LogisticRegression"

    def test_model_params_forwarded(self, tmp_path: Path) -> None:
        data = {**MINIMAL_VALID, "model_params": {"n_estimators": 200, "random_state": 7}}
        config = load_config(write_yaml(tmp_path, data))
        assert config.model_params["n_estimators"] == 200
        assert config.model_params["random_state"] == 7

    def test_default_model_params_is_empty_dict(self, tmp_path: Path) -> None:
        config = load_config(write_yaml(tmp_path, MINIMAL_VALID))
        assert config.model_params == {}
