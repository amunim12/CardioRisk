"""Tests for target binarization."""

from __future__ import annotations

import pandas as pd
import pytest

from cardiorisk.target import binarize_target


def test_zero_maps_to_zero_and_positive_maps_to_one() -> None:
    severity = pd.Series([0, 1, 2, 3, 4])
    result = binarize_target(severity)
    assert result.tolist() == [0, 1, 1, 1, 1]


def test_result_is_named_and_int() -> None:
    result = binarize_target(pd.Series([0, 2]), name="risk")
    assert result.name == "risk"
    assert str(result.dtype) == "int64"


def test_missing_values_raise() -> None:
    with pytest.raises(ValueError, match="missing"):
        binarize_target(pd.Series([0, 1, None]))


def test_index_is_preserved() -> None:
    severity = pd.Series([0, 3], index=["a", "b"])
    result = binarize_target(severity)
    assert result.index.tolist() == ["a", "b"]
