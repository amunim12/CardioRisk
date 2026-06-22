"""Model selection, metrics, threshold tuning, and diagnostic plots.

Plots are rendered with the non-interactive Agg backend so this module works
headless (CI, containers) without a display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from cardiorisk.modeling import build_model


@dataclass(frozen=True)
class CVResult:
    """Cross-validation summary for one model."""

    model: str
    mean_score: float
    std_score: float
    scores: list[float]


def cross_validate_models(
    model_names: list[str],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    scoring: str = "roc_auc",
    seed: int = 42,
) -> list[CVResult]:
    """Score each model with stratified k-fold CV, sorted best-first."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results: list[CVResult] = []
    for name in model_names:
        pipeline = build_model(name, seed=seed)
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring)
        results.append(
            CVResult(
                model=name,
                mean_score=float(scores.mean()),
                std_score=float(scores.std()),
                scores=[float(s) for s in scores],
            )
        )
    results.sort(key=lambda r: r.mean_score, reverse=True)
    return results


def tune_threshold(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    *,
    min_precision: float = 0.70,
) -> float:
    """Pick the decision threshold that maximises recall subject to a precision floor.

    This encodes the clinical preference for sensitivity: among all thresholds
    that keep precision at or above ``min_precision``, choose the one with the
    highest recall. Falls back to 0.5 if the floor is unattainable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1.
    precision, recall = precision[:-1], recall[:-1]
    eligible = precision >= min_precision
    if not eligible.any():
        return 0.5
    eligible_recall = np.where(eligible, recall, -np.inf)
    best_idx = int(np.argmax(eligible_recall))
    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute the full metric suite at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred),
        display_labels=["no disease", "at risk"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix")
    return _save(fig, path)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    return _save(fig, path)


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(y_true, y_prob, n_bins=10, ax=ax, name="model")
    ax.set_title("Calibration curve")
    return _save(fig, path)
