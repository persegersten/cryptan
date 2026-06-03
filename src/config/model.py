"""Pydantic config model for the training pipeline."""

from __future__ import annotations

import datetime
from pathlib import Path

from pydantic import BaseModel, Field, StrictInt, field_validator, model_validator


class FeatureConfig(BaseModel):
    """Configurable windows for feature engineering.

    All windows must be positive integers.  The defaults match the MVP
    feature set described in the repository instructions.
    """

    return_windows: list[int] = Field(
        default=[1, 5, 20],
        description="N-bar close return windows, e.g. [1, 5, 20].",
    )
    ma_short_window: int = Field(
        default=7,
        gt=0,
        description="Short rolling-mean window (bars).",
    )
    ma_long_window: int = Field(
        default=20,
        gt=0,
        description="Long rolling-mean window (bars).",
    )
    volatility_window: int = Field(
        default=20,
        gt=0,
        description="Rolling standard-deviation window for 1-bar return volatility.",
    )
    volume_window: int = Field(
        default=20,
        gt=0,
        description="Rolling window for volume mean and z-score.",
    )
    correlation_window: int = Field(
        default=20,
        gt=0,
        description="Rolling window for cross-asset return correlation.",
    )


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


class BacktestConfig(BaseModel):
    """Simple strategy backtest parameters."""

    portfolio_mode: str = Field(
        "all_in_long_cash",
        description="Portfolio interpretation for model signals.",
    )
    initial_position: int = Field(
        0,
        description="Initial executed position before the first in-split signal.",
    )
    transaction_fee: float = Field(
        0.001,
        ge=0.0,
        description=(
            "Fractional fee charged for each unit of position turnover, "
            "e.g. 0.001 is 10 bps."
        ),
    )
    return_buffer: float = Field(
        0.005,
        ge=0.0,
        description="Extra return required beyond round-trip fees for a long label.",
    )
    min_validation_cumulative_return: float = Field(
        0.10,
        description="Minimum validation cumulative return required for eligibility.",
    )
    min_validation_exposure_ratio: float = Field(
        0.10,
        ge=0.0,
        le=1.0,
        description="Minimum validation exposure ratio required for eligibility.",
    )
    min_validation_traded_bars: int = Field(
        100,
        ge=0,
        description="Minimum number of validation bars with long exposure.",
    )
    max_validation_drawdown: float = Field(
        -0.85,
        le=0.0,
        description="Most negative validation max drawdown allowed before rejection.",
    )
    max_validation_turnover: float = Field(
        250.0,
        ge=0.0,
        description="Maximum validation turnover allowed before rejection.",
    )
    entry_thresholds: list[float] = Field(
        default=[0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65],
        description="Probability thresholds for entering long in policy search.",
    )
    exit_thresholds: list[float] = Field(
        default=[0.35, 0.40, 0.45, 0.475, 0.50],
        description="Probability thresholds for exiting to cash in policy search.",
    )
    min_hold_bars_grid: list[int] = Field(
        default=[0, 3, 6, 12, 24],
        description="Minimum-hold bar counts for validation policy search.",
    )

    @field_validator("portfolio_mode")
    @classmethod
    def portfolio_mode_must_be_supported(cls, value: str) -> str:
        value = value.strip()
        if value != "all_in_long_cash":
            raise ValueError("Only portfolio_mode='all_in_long_cash' is supported.")
        return value

    @field_validator("initial_position")
    @classmethod
    def initial_position_must_be_cash_or_long(cls, value: int) -> int:
        if value not in (0, 1):
            raise ValueError("initial_position must be 0 (cash) or 1 (long).")
        return value


class ModelCandidateConfig(BaseModel):
    """One candidate estimator to train and score during model selection."""

    name: str | None = Field(
        default=None,
        description="Optional stable display name for this candidate.",
    )
    model_type: str = Field(
        ...,
        description="Model identifier; must match a key in the model registry.",
    )
    model_params: dict = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the model constructor.",
    )

    @field_validator("name", "model_type")
    @classmethod
    def candidate_strings_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return value
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Model candidate name/model_type must not be empty.")
        return cleaned


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
    start_date: StrictInt = Field(
        ...,
        description=(
            "Historical data start day as an integer offset from today "
            "in UTC; -1 is yesterday, 0 is today."
        ),
    )
    end_date: StrictInt = Field(
        ...,
        description=(
            "Historical data end day as an integer offset from today "
            "in UTC; -1 is yesterday, 0 is today."
        ),
    )

    # --- labelling ---
    prediction_horizon_bars: int = Field(
        12,
        gt=0,
        description="Number of bars ahead to predict.",
    )
    return_threshold: float = Field(
        0.01,
        gt=0.0,
        description="Legacy directional threshold; binary_long_cash uses backtest.return_buffer.",
    )
    model_task: str = Field(
        "binary_long_cash",
        description="Supervised target type. Only binary_long_cash is supported.",
    )

    # --- split ---
    split: SplitConfig = Field(default_factory=SplitConfig)

    # --- evaluation / backtest ---
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    # --- model ---
    model_type: str = Field(
        "RandomForestClassifier",
        description="Model identifier; must match a key in the model registry.",
    )
    model_params: dict = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the model constructor.",
    )
    model_candidates: list[ModelCandidateConfig] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional candidate estimators. If supplied, all candidates are trained "
            "and the best one is selected on validation data."
        ),
    )
    model_selection_metric: str = Field(
        "f1_macro",
        description="Validation metric used to select the best model candidate.",
    )

    # --- artifacts ---
    artifacts_dir: Path = Field(
        Path("artifacts"),
        description="Root directory for run artifacts.",
    )

    # --- features ---
    feature_config: FeatureConfig = Field(
        default_factory=FeatureConfig,
        description="Feature engineering windows and parameters.",
    )

    # --- credentials ---
    # These fields may be supplied via environment variables (CRYPTAN_DATA_API_KEY
    # and CRYPTAN_DATA_API_SECRET). They are injected by the config loader and must
    # never appear in committed YAML files.
    data_api_key: str = Field(
        "changeme",
        description="Data provider API key (from CRYPTAN_DATA_API_KEY).",
    )
    data_api_secret: str = Field(
        "changeme",
        description="Data provider API secret (from CRYPTAN_DATA_API_SECRET).",
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

    @field_validator("model_type", "model_selection_metric")
    @classmethod
    def model_strings_must_be_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model_type and model_selection_metric must not be empty.")
        return value

    @field_validator("model_task")
    @classmethod
    def model_task_must_be_supported(cls, value: str) -> str:
        value = value.strip()
        if value != "binary_long_cash":
            raise ValueError("Only model_task='binary_long_cash' is supported.")
        return value

    @property
    def min_required_future_return(self) -> float:
        """Return threshold for binary long/cash positive labels."""
        return (2.0 * self.backtest.transaction_fee) + self.backtest.return_buffer

    @field_validator("model_selection_metric")
    @classmethod
    def model_selection_metric_must_be_supported(cls, value: str) -> str:
        supported_metrics = {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
        if value not in supported_metrics:
            supported = ", ".join(sorted(supported_metrics))
            raise ValueError(
                f"Unsupported model_selection_metric {value!r}. "
                f"Supported metrics: {supported}."
            )
        return value

    @model_validator(mode="after")
    def end_date_must_be_after_start_date(self) -> "TrainingConfig":
        if self.end_date < self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be greater than or equal to "
                f"start_date ({self.start_date})."
            )
        return self

    def resolve_start_datetime(
        self,
        today: datetime.date | None = None,
    ) -> datetime.datetime:
        """Resolve ``start_date`` offset to an inclusive UTC midnight."""
        base_date = today or datetime.datetime.now(datetime.timezone.utc).date()
        start = base_date + datetime.timedelta(days=self.start_date)
        return datetime.datetime.combine(
            start,
            datetime.time.min,
            tzinfo=datetime.timezone.utc,
        )

    def resolve_end_datetime(
        self,
        today: datetime.date | None = None,
    ) -> datetime.datetime:
        """Resolve ``end_date`` offset to an exclusive UTC boundary.

        The configured integer names the final included calendar day.  The
        ingestion layer expects an exclusive end timestamp, so the resolved
        boundary is midnight after that day.
        """
        base_date = today or datetime.datetime.now(datetime.timezone.utc).date()
        end = base_date + datetime.timedelta(days=self.end_date + 1)
        return datetime.datetime.combine(
            end,
            datetime.time.min,
            tzinfo=datetime.timezone.utc,
        )
