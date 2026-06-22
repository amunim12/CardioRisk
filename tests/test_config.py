"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cardiorisk.config import Settings, load_settings


def test_loads_default_config() -> None:
    settings = load_settings("configs/default.yaml")
    assert settings.seed == 42
    assert "logistic_regression" in settings.models
    assert settings.cv.n_splits >= 2


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_settings("configs/does_not_exist.yaml")


def test_empty_models_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(models=[])


def test_test_size_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        Settings(data={"test_size": 1.5})


def test_roundtrip_from_yaml(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("seed: 7\nmodels: [decision_tree]\n", encoding="utf-8")
    settings = load_settings(cfg)
    assert settings.seed == 7
    assert settings.models == ["decision_tree"]
