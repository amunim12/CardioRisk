"""Typed, validated configuration loaded from YAML.

The whole pipeline is parameterised here so behaviour is reproducible and
auditable. ``load_settings`` reads a YAML file and validates it into the
``Settings`` model; invalid configs fail fast with actionable messages.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Where the data comes from and how it is split."""

    uci_id: int = 45
    raw_path: Path = Path("data/raw/heart_disease.csv")
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class TargetConfig(BaseModel):
    """How the multi-class severity label is binarised."""

    source_column: str = "num"
    name: str = "target"


class ThresholdConfig(BaseModel):
    """Decision-threshold tuning policy.

    Sensitivity is prioritised: we accept the lowest precision that is still
    clinically tolerable and then maximise recall, because a missed at-risk
    patient is the costly error.
    """

    strategy: str = "max_recall_at_min_precision"
    min_precision: float = Field(default=0.70, ge=0.0, le=1.0)


class CVConfig(BaseModel):
    """Cross-validation settings for model selection."""

    n_splits: int = Field(default=5, ge=2)
    scoring: str = "roc_auc"


class TrackingConfig(BaseModel):
    """Experiment tracking. Uses MLflow if installed, otherwise skipped."""

    enabled: bool = True
    experiment_name: str = "cardiorisk"


class Settings(BaseModel):
    """Top-level, fully validated pipeline configuration."""

    seed: int = 42
    data: DataConfig = DataConfig()
    target: TargetConfig = TargetConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    cv: CVConfig = CVConfig()
    models: list[str] = Field(default_factory=lambda: ["logistic_regression"])
    artifacts_dir: Path = Path("artifacts")
    tracking: TrackingConfig = TrackingConfig()

    @field_validator("models")
    @classmethod
    def _models_not_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("`models` must list at least one model name")
        return value


def load_settings(path: str | Path = "configs/default.yaml") -> Settings:
    """Load and validate settings from a YAML file.

    Raises:
        FileNotFoundError: if the config file does not exist.
        pydantic.ValidationError: if the contents do not match the schema.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Pass a valid path or create configs/default.yaml."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Settings.model_validate(raw)
