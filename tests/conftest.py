"""Shared pytest fixtures.

A small synthetic dataset (with the real schema, real missingness, and a
learnable signal) lets the test suite run fast and offline, without touching
the network or the full UCI download.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cardiorisk.preprocessing import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """A 300-row frame matching the UCI schema with a learnable target."""
    rng = np.random.default_rng(42)
    n = 300
    data = {
        "age": rng.integers(29, 77, n).astype(float),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.normal(132, 18, n).clip(90, 200),
        "chol": rng.normal(245, 55, n).clip(120, 560),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.normal(150, 23, n).clip(70, 205),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.normal(1.0, 1.1, n).clip(0, 6),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    }
    df = pd.DataFrame(data)

    # A genuine signal: risk rises with age, oldpeak, exang, ca and falls with thalach.
    logit = (
        0.04 * (df["age"] - 54)
        + 0.6 * df["oldpeak"]
        + 1.1 * df["exang"]
        + 0.7 * df["ca"]
        - 0.03 * (df["thalach"] - 150)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["num"] = (rng.random(n) < prob).astype(int) * rng.integers(1, 5, n)

    # Inject realistic missingness.
    for col, frac in {"chol": 0.08, "ca": 0.15, "thal": 0.12}.items():
        mask = rng.random(n) < frac
        df.loc[mask, col] = np.nan

    df["source"] = "synthetic"
    return df


@pytest.fixture
def features_target(synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = synthetic_df[ALL_FEATURES]
    y = (synthetic_df["num"] > 0).astype(int).rename("target")
    return X, y


@pytest.fixture
def feature_lists() -> tuple[list[str], list[str], list[str]]:
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
