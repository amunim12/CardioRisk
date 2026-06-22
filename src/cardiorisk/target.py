"""Target construction.

The UCI ``num`` column encodes heart-disease severity on a 0-4 scale. For a
binary risk model we collapse it to presence/absence of disease.
"""

from __future__ import annotations

import pandas as pd


def binarize_target(severity: pd.Series, name: str = "target") -> pd.Series:
    """Collapse the 0-4 severity label into a binary at-risk flag.

    ``0`` (no disease) maps to ``0``; any value ``> 0`` maps to ``1``.

    Args:
        severity: integer-like severity values (the UCI ``num`` column).
        name: name for the returned Series.

    Returns:
        An ``int`` Series of 0/1 aligned to the input index.

    Raises:
        ValueError: if the input contains missing values, which would make the
            label ambiguous.
    """
    if severity.isna().any():
        raise ValueError(
            "Severity column contains missing values; cannot binarize a "
            "missing label unambiguously."
        )
    binary = (severity.astype(int) > 0).astype(int)
    return binary.rename(name)
