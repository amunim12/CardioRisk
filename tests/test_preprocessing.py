"""Tests for the preprocessing transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cardiorisk.preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def test_feature_lists_are_disjoint_and_cover_all() -> None:
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(ALL_FEATURES) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)


def test_imputes_missing_values(features_target: tuple[pd.DataFrame, pd.Series]) -> None:
    X, _ = features_target
    assert X.isna().any().any(), "fixture should contain missing values"
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert not np.isnan(transformed).any(), "no NaNs should remain after imputation"


def test_output_row_count_matches_input(
    features_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = features_target
    pre = build_preprocessor()
    transformed = pre.fit_transform(X)
    assert transformed.shape[0] == len(X)


def test_unknown_category_is_handled_at_transform(
    features_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = features_target
    pre = build_preprocessor().fit(X)
    novel = X.iloc[[0]].copy()
    novel.loc[:, "thal"] = 99.0  # category unseen during fit
    transformed = pre.transform(novel)  # must not raise
    assert transformed.shape[0] == 1
