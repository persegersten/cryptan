"""Config loader: reads a YAML file and returns a validated TrainingConfig."""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config.model import TrainingConfig

logger = logging.getLogger(__name__)


def load_config(path: str | Path) -> TrainingConfig:
    """Load and validate a training configuration from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML config file.

    Returns
    -------
    TrainingConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the YAML content fails schema validation.
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
