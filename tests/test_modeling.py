"""Tests for the model factory."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from cardiorisk.modeling import available_models, build_model


@pytest.mark.parametrize("name", available_models())
def test_build_returns_pipeline_for_each_model(name: str) -> None:
    model = build_model(name)
    assert isinstance(model, Pipeline)
    assert "preprocess" in model.named_steps
    assert "model" in model.named_steps


@pytest.mark.parametrize("name", available_models())
def test_each_model_fits_and_predicts(
    name: str, features_target: tuple[pd.DataFrame, pd.Series]
) -> None:
    X, y = features_target
    model = build_model(name, seed=0).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("random_forest_9000")
