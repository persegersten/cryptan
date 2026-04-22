"""Config loader: reads a YAML file and returns a validated TrainingConfig."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config.model import TrainingConfig

logger = logging.getLogger(__name__)

# Maps TrainingConfig field names → environment variable names.
_REQUIRED_ENV_VARS: dict[str, str] = {
    "data_api_key": "CRYPTAN_DATA_API_KEY",
    "data_api_secret": "CRYPTAN_DATA_API_SECRET",
}

_PLACEHOLDER = "changeme"


def _read_required_env_vars() -> dict[str, str]:
    """Read required environment variables and return them keyed by field name.

    Raises
    ------
    EnvironmentError
        If any required variable is absent or still holds the placeholder value.
    """
    values: dict[str, str] = {}
    for field_name, env_var in _REQUIRED_ENV_VARS.items():
        value = os.environ.get(env_var, "")
        if not value or value.strip().lower() == _PLACEHOLDER:
            raise EnvironmentError(
                f"Required environment variable {env_var!r} is not set or still uses "
                f"the placeholder value {_PLACEHOLDER!r}. "
                f"Set a real value before starting the pipeline."
            )
        values[field_name] = value
    return values


def load_config(
    path: str | Path,
    local_path: str | Path | None = None,
) -> TrainingConfig:
    """Load and validate a training configuration from a YAML file.

    The loader merges configuration in three steps:

    1. Base config from *path* (always required, safe to commit).
    2. Optional local overrides from *local_path* (gitignored, never committed).
    3. Required secrets injected from environment variables.

    Parameters
    ----------
    path:
        Path to the base YAML config file.
    local_path:
        Optional path to a local YAML override file.  Keys present here are
        merged on top of the base config before validation.  Useful for local
        development overrides such as shorter date ranges or custom artifact
        directories.  If the file does not exist it is silently ignored.

    Returns
    -------
    TrainingConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the base config file does not exist.
    EnvironmentError
        If a required environment variable is absent or holds the placeholder
        value ``changeme``.
    ValueError
        If the merged YAML content fails schema validation.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path.resolve()}"
        )

    logger.debug("Loading config from %s", config_path.resolve())

    with config_path.open("r", encoding="utf-8") as file_handle:
        raw = yaml.safe_load(file_handle)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file must contain a YAML mapping at the top level; "
            f"got {type(raw).__name__!r} from {config_path}."
        )

    # --- optional local overrides (shallow merge) ---
    if local_path is not None:
        local = Path(local_path)
        if local.exists():
            logger.debug("Applying local overrides from %s", local.resolve())
            with local.open("r", encoding="utf-8") as fh:
                local_raw = yaml.safe_load(fh)
            if isinstance(local_raw, dict):
                raw.update(local_raw)
        else:
            logger.debug(
                "Local override file not found, skipping: %s", local.resolve()
            )

    # --- inject required credentials from environment variables ---
    raw.update(_read_required_env_vars())

    try:
        config = TrainingConfig(**raw)
    except ValidationError as exc:
        raise ValueError(
            f"Config validation failed for {config_path}:\n{exc}"
        ) from exc

    logger.info(
        "Config loaded: trading_symbol=%s, signal_symbols=%s, timeframe=%s",
        config.trading_symbol,
        config.signal_symbols,
        config.timeframe,
    )

    return config
