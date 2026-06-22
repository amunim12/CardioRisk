"""Feature importance / explainability.

Three complementary lenses, ordered by robustness:

1. ``permutation_importance`` - model-agnostic, always available; the primary
   signal reported for every model.
2. ``coefficient_importance`` - signed log-odds for the linear model only.
3. ``shap_summary`` - rich per-instance attributions, but SHAP is an optional
   dependency; the import is guarded so a missing/broken install never breaks
   the pipeline.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

SHAP_AVAILABLE: bool = find_spec("shap") is not None


def _feature_names(pipeline: Pipeline) -> list[str]:
    """Names emerging from the fitted preprocessor (post one-hot)."""
    return list(pipeline.named_steps["preprocess"].get_feature_names_out())


def permutation_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Model-agnostic importance on the *raw* input features.

    Permuting raw columns (rather than post-encoding columns) keeps the result
    interpretable in clinical terms.
    """
    result = permutation_importance(
        pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=seed
    )
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": result.importances_mean,
                "std": result.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def coefficient_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """Signed coefficients for a logistic-regression pipeline (else ``None``)."""
    model = pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        return None
    coefs = model.coef_.ravel()
    return (
        pd.DataFrame({"feature": _feature_names(pipeline), "coefficient": coefs})
        .assign(abs_coefficient=lambda d: d["coefficient"].abs())
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def plot_permutation_importance(importance: pd.DataFrame, path: Path, top_n: int = 15) -> Path:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top["feature"], top["importance"], xerr=top["std"], color="#2b6cb0")
    ax.set_xlabel("Mean ROC-AUC drop when permuted")
    ax.set_title("Permutation feature importance")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def shap_summary(
    pipeline: Pipeline, X: pd.DataFrame, path: Path, *, max_samples: int = 200
) -> Path | None:
    """Save a SHAP summary plot if SHAP is installed, else return ``None``.

    Uses the post-preprocessing matrix so any estimator type is supported.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed; skipping SHAP summary plot.")
        return None

    import shap  # local, guarded import

    pre = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    sample = X.sample(n=min(max_samples, len(X)), random_state=42)
    matrix = pre.transform(sample)
    names = _feature_names(pipeline)

    try:
        explainer = shap.Explainer(model, matrix, feature_names=names)
        values = explainer(matrix)
    except Exception as exc:  # SHAP can fail on some estimator combos
        logger.warning("SHAP explanation failed (%s); skipping.", exc)
        return None

    fig = plt.figure()
    shap.summary_plot(values, features=matrix, feature_names=names, show=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def importance_summary_line(importance: pd.DataFrame, top_n: int = 5) -> str:
    """A short human-readable headline of the top drivers."""
    top = importance.head(top_n)["feature"].tolist()
    return "Top predictors: " + ", ".join(top)
