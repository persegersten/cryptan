"""Pydantic config model for the training pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


class SplitConfig(BaseModel):
    """Chronological split fractions for train / validation / test."""

    train: float = Field(0.70, gt=0.0, lt=1.0)
    validation: float = Field(0.15, gt=0.0, lt=1.0)
    test: float = Field(0.15, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def fractions_must_sum_to_one(self) -> "SplitConfig":
        total = round(self.train + self.validation + self.test, 10)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total:.6f} "
                f"(train={self.train}, validation={self.validation}, test={self.test})."
            )
        return self


class TrainingConfig(BaseModel):
    """Full configuration for one training pipeline run."""

    # --- symbols ---
    trading_symbol: str = Field(..., description="Target asset ticker, e.g. 'ETH'.")
    signal_symbols: list[str] = Field(
        ...,
        min_length=1,
        description="Tickers used as feature sources, e.g. ['ETH', 'BNB', 'SOL'].",
    )

    # --- data ---
    timeframe: str = Field("1h", description="OHLCV bar interval, e.g. '1h', '4h', '1d'.")
    start_date: str = Field(..., description="Historical data start date (ISO 8601, UTC).")
    end_date: str = Field(..., description="Historical data end date (ISO 8601, UTC).")

    # --- labelling ---
    prediction_horizon_bars: int = Field(
        12,
        gt=0,
        description="Number of bars ahead to predict.",
    )
    return_threshold: float = Field(
        0.01,
        gt=0.0,
        description="Minimum absolute return to classify as directional (+1/-1).",
    )

    # --- split ---
    split: SplitConfig = Field(default_factory=SplitConfig)

    # --- model ---
    model_type: str = Field(
        "RandomForestClassifier",
        description="Model identifier; must match a key in the model registry.",
    )
    model_params: dict = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the model constructor.",
    )

    # --- artifacts ---
    artifacts_dir: Path = Field(
        Path("artifacts"),
        description="Root directory for run artifacts.",
    )

    @field_validator("trading_symbol")
    @classmethod
    def trading_symbol_must_be_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("trading_symbol must not be empty.")
        return value.upper()

    @field_validator("signal_symbols", mode="before")
    @classmethod
    def signal_symbols_must_be_non_empty_strings(cls, value: list) -> list[str]:
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    f"Each signal symbol must be a non-empty string; got {item!r}."
                )
            cleaned.append(item.strip().upper())
        return cleaned

    @field_validator("timeframe")
    @classmethod
    def timeframe_must_be_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("timeframe must not be empty.")
        return value

    @field_validator("start_date", "end_date")
    @classmethod
    def date_must_be_iso_format(cls, value: str) -> str:
        try:
            datetime.date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"Date must be in ISO 8601 format (YYYY-MM-DD), got {value!r}."
            ) from exc
        return value

    @model_validator(mode="after")
    def end_date_must_be_after_start_date(self) -> "TrainingConfig":
        start = datetime.date.fromisoformat(self.start_date)
        end = datetime.date.fromisoformat(self.end_date)
        if end <= start:
            raise ValueError(
                f"end_date ({self.end_date}) must be strictly after start_date ({self.start_date})."
            )
        return self
