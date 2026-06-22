"""Model factory.

Each model is a full ``Pipeline`` (preprocessing + estimator) so cross-
validation is honest and the fitted object can be serialised whole. The three
models span the interpretability/accuracy spectrum the assignment cares about:

- ``logistic_regression``  - the interpretable headline model (log-odds).
- ``decision_tree``        - a human-readable rule view.
- ``hist_gradient_boosting`` - the strong accuracy benchmark.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from cardiorisk.preprocessing import build_preprocessor


def _logistic_regression(seed: int) -> ClassifierMixin:
    return LogisticRegression(max_iter=1000, random_state=seed)


def _decision_tree(seed: int) -> ClassifierMixin:
    # Shallow tree: interpretable and resistant to overfitting on ~900 rows.
    return DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=seed)


def _hist_gradient_boosting(seed: int) -> ClassifierMixin:
    return HistGradientBoostingClassifier(random_state=seed)


# Registry: model name -> estimator factory.
MODEL_BUILDERS: dict[str, Callable[[int], ClassifierMixin]] = {
    "logistic_regression": _logistic_regression,
    "decision_tree": _decision_tree,
    "hist_gradient_boosting": _hist_gradient_boosting,
}


def available_models() -> list[str]:
    """Names of all registered models."""
    return list(MODEL_BUILDERS)


def build_model(name: str, seed: int = 42) -> Pipeline:
    """Build a preprocessing + estimator pipeline for ``name``.

    Raises:
        KeyError: if ``name`` is not a registered model.
    """
    if name not in MODEL_BUILDERS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_BUILDERS)}")
    estimator = MODEL_BUILDERS[name](seed)
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )
