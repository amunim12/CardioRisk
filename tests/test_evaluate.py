"""Tests for metrics, threshold tuning, and CV selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cardiorisk.evaluate import (
    compute_metrics,
    cross_validate_models,
    tune_threshold,
)


def test_compute_metrics_perfect_separation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, y_prob, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["recall"] == 1.0


def test_compute_metrics_keys_present() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.3, 0.6, 0.4, 0.7])
    m = compute_metrics(y_true, y_prob)
    assert {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"} <= m.keys()


def test_tune_threshold_respects_precision_floor() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.55, 0.45, 0.8, 0.9])
    thr = tune_threshold(y_true, y_prob, min_precision=1.0)
    # At precision 1.0, the positive at 0.45 (with a negative at 0.55) cannot be
    # captured without a false positive, so threshold must exceed 0.55.
    preds = (y_prob >= thr).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    assert fp == 0 and tp >= 1


def test_tune_threshold_falls_back_when_floor_unreachable() -> None:
    y_true = np.array([0, 1])
    y_prob = np.array([0.9, 0.1])  # ranking inverted; high precision impossible
    assert tune_threshold(y_true, y_prob, min_precision=0.99) == 0.5


def test_cross_validate_ranks_models(
    features_target: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = features_target
    results = cross_validate_models(
        ["logistic_regression", "decision_tree"], X, y, n_splits=3, seed=0
    )
    assert len(results) == 2
    # Sorted best-first.
    assert results[0].mean_score >= results[1].mean_score
    # A real signal should beat chance.
    assert results[0].mean_score > 0.5
