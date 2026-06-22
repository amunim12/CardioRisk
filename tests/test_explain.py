"""Tests for feature-importance utilities."""

from __future__ import annotations

import pandas as pd

from cardiorisk.explain import (
    coefficient_importance,
    permutation_feature_importance,
)
from cardiorisk.modeling import build_model


def test_permutation_importance_ranks_known_signal(
    features_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = features_target
    pipeline = build_model("logistic_regression", seed=0).fit(X, y)
    imp = permutation_feature_importance(pipeline, X, y, n_repeats=5, seed=0)
    assert list(imp.columns) == ["feature", "importance", "std"]
    assert len(imp) == X.shape[1]
    # Sorted descending.
    assert imp["importance"].is_monotonic_decreasing


def test_coefficients_present_for_logreg(
    features_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = features_target
    pipeline = build_model("logistic_regression", seed=0).fit(X, y)
    coefs = coefficient_importance(pipeline)
    assert coefs is not None
    assert "coefficient" in coefs.columns


def test_coefficients_none_for_tree(
    features_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = features_target
    pipeline = build_model("decision_tree", seed=0).fit(X, y)
    assert coefficient_importance(pipeline) is None
