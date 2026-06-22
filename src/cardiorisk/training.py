"""End-to-end training orchestration.

Pulls together data loading, model selection, threshold tuning, evaluation,
explainability, persistence, and tracking into one reproducible run. Kept
separate from the CLI so the whole flow is importable and testable.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from cardiorisk import __version__
from cardiorisk.config import Settings
from cardiorisk.data import feature_target_split, load_raw_data
from cardiorisk.evaluate import (
    compute_metrics,
    cross_validate_models,
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    tune_threshold,
)
from cardiorisk.explain import (
    coefficient_importance,
    importance_summary_line,
    permutation_feature_importance,
    plot_permutation_importance,
    shap_summary,
)
from cardiorisk.modeling import build_model
from cardiorisk.persistence import ModelBundle, save_bundle
from cardiorisk.preprocessing import ALL_FEATURES
from cardiorisk.target import binarize_target
from cardiorisk.tracking import ExperimentTracker

logger = logging.getLogger(__name__)


@dataclass
class TrainingReport:
    """Summary of a training run, also serialised to metrics.json."""

    best_model: str
    cv_ranking: list[dict[str, Any]]
    threshold: float
    test_metrics_tuned: dict[str, float]
    test_metrics_default: dict[str, float]
    top_features: list[str]
    n_train: int
    n_test: int
    artifacts_dir: str


def _provenance(n_train: int, best_model: str) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "cardiorisk_version": __version__,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "n_train": n_train,
        "best_model": best_model,
    }


def run_training(settings: Settings) -> TrainingReport:
    """Run the full training pipeline and write all artifacts to disk."""
    artifacts = Path(settings.artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    # 1. Load + label.
    raw = load_raw_data(settings.data.raw_path)
    features, severity = feature_target_split(raw, settings.target.source_column)
    features = features[ALL_FEATURES]  # enforce expected schema/order
    y = binarize_target(severity, settings.target.name)
    logger.info("Loaded %d rows; positive rate = %.1f%%", len(y), 100 * y.mean())

    # 2. Hold out a stratified test set.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=settings.data.test_size,
        stratify=y,
        random_state=settings.seed,
    )

    # 3. Select the best model by cross-validated ROC-AUC on the training set.
    cv_results = cross_validate_models(
        settings.models,
        X_train,
        y_train,
        n_splits=settings.cv.n_splits,
        scoring=settings.cv.scoring,
        seed=settings.seed,
    )
    best_name = cv_results[0].model
    logger.info(
        "Best model: %s (CV %s = %.3f)",
        best_name,
        settings.cv.scoring,
        cv_results[0].mean_score,
    )

    # 4. Tune the decision threshold on out-of-fold training probabilities
    #    (never on the test set) to honour the sensitivity-first policy.
    best_pipeline = build_model(best_name, seed=settings.seed)
    cv = StratifiedKFold(n_splits=settings.cv.n_splits, shuffle=True, random_state=settings.seed)
    oof_prob = cross_val_predict(best_pipeline, X_train, y_train, cv=cv, method="predict_proba")[
        :, 1
    ]
    threshold = tune_threshold(y_train, oof_prob, min_precision=settings.threshold.min_precision)
    logger.info("Tuned decision threshold = %.3f", threshold)

    # 5. Fit on the full training set; evaluate on the held-out test set.
    best_pipeline.fit(X_train, y_train)
    test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    metrics_tuned = compute_metrics(y_test, test_prob, threshold=threshold)
    metrics_default = compute_metrics(y_test, test_prob, threshold=0.5)

    # 6. Diagnostic plots.
    plot_roc_curve(y_test.to_numpy(), test_prob, artifacts / "roc_curve.png")
    plot_pr_curve(y_test.to_numpy(), test_prob, artifacts / "pr_curve.png")
    plot_calibration_curve(y_test.to_numpy(), test_prob, artifacts / "calibration_curve.png")
    y_pred = (test_prob >= threshold).astype(int)
    plot_confusion_matrix(y_test.to_numpy(), y_pred, artifacts / "confusion_matrix.png")

    # 7. Explainability.
    perm = permutation_feature_importance(
        best_pipeline, X_test, y_test, scoring=settings.cv.scoring, seed=settings.seed
    )
    perm.to_csv(artifacts / "permutation_importance.csv", index=False)
    plot_permutation_importance(perm, artifacts / "permutation_importance.png")

    coefs = coefficient_importance(best_pipeline)
    if coefs is not None:
        coefs.to_csv(artifacts / "coefficients.csv", index=False)

    shap_path = shap_summary(best_pipeline, X_test, artifacts / "shap_summary.png")
    if shap_path is None:
        logger.info("SHAP summary not produced (optional).")

    # 8. Persist the model bundle.
    bundle = ModelBundle(
        pipeline=best_pipeline,
        threshold=threshold,
        feature_names=ALL_FEATURES,
        model_name=best_name,
        metadata=_provenance(len(X_train), best_name),
    )
    save_bundle(bundle, artifacts / "model.joblib")

    # 9. Assemble + write the report.
    report = TrainingReport(
        best_model=best_name,
        cv_ranking=[
            {"model": r.model, "mean": r.mean_score, "std": r.std_score} for r in cv_results
        ],
        threshold=threshold,
        test_metrics_tuned=metrics_tuned,
        test_metrics_default=metrics_default,
        top_features=perm.head(5)["feature"].tolist(),
        n_train=len(X_train),
        n_test=len(X_test),
        artifacts_dir=str(artifacts),
    )
    (artifacts / "metrics.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info(importance_summary_line(perm))

    # 10. Optional experiment tracking.
    with ExperimentTracker(
        settings.tracking.experiment_name, enabled=settings.tracking.enabled
    ) as tracker:
        tracker.log_params(
            {
                "best_model": best_name,
                "threshold": threshold,
                "n_splits": settings.cv.n_splits,
                "test_size": settings.data.test_size,
                "seed": settings.seed,
            }
        )
        tracker.log_metrics({f"test_{k}": v for k, v in metrics_tuned.items()})
        tracker.log_artifacts(artifacts)

    return report
