"""Serialisation of the trained model and its operating metadata.

The whole inference contract is bundled together: the fitted pipeline, the
tuned decision threshold, the raw input columns the model expects, and
provenance metadata. This is what the API and CLI load at serve time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


@dataclass
class ModelBundle:
    """Everything needed to make a prediction, in one serialisable object."""

    pipeline: Pipeline
    threshold: float
    feature_names: list[str]
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: Any) -> Any:
        """Probability of the positive (at-risk) class."""
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X: Any) -> Any:
        """Binary at-risk prediction using the tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)


def save_bundle(bundle: ModelBundle, path: str | Path) -> Path:
    """Persist a :class:`ModelBundle` to disk via joblib."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return out


def load_bundle(path: str | Path) -> ModelBundle:
    """Load a :class:`ModelBundle` from disk.

    Security: joblib uses pickle, which can execute arbitrary code on load.
    Only load bundles produced by this project's own ``train`` command
    (artifacts/model.joblib). Never load a .joblib file from an untrusted
    source.

    Raises:
        FileNotFoundError: if no model has been trained yet.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"No trained model at {src}. Run `cardiorisk train` first.")
    return joblib.load(src)
