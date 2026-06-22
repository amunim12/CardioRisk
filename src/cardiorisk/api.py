"""FastAPI inference service.

Exposes ``/health`` and ``/predict``. The trained model bundle is loaded lazily
and cached; if no model has been trained yet, prediction returns a clear 503
rather than crashing the service.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from cardiorisk.persistence import ModelBundle, load_bundle
from cardiorisk.preprocessing import ALL_FEATURES

MODEL_PATH = os.environ.get("CARDIORISK_MODEL_PATH", "artifacts/model.joblib")

app = FastAPI(
    title="CardioRisk API",
    version="0.1.0",
    summary="Heart disease risk prediction service.",
)


class PatientFeatures(BaseModel):
    """One patient's clinical measurements.

    Frequently-missing fields (``chol``, ``slope``, ``ca``, ``thal``) are
    optional; the model's imputation step fills them in.
    """

    age: float = Field(ge=0, le=120, examples=[57])
    sex: int = Field(ge=0, le=1, description="1 = male, 0 = female", examples=[1])
    cp: int = Field(ge=1, le=4, description="chest pain type", examples=[4])
    trestbps: float = Field(gt=0, le=300, description="resting blood pressure", examples=[140])
    chol: float | None = Field(default=None, ge=0, le=700, examples=[260])
    fbs: int = Field(ge=0, le=1, description="fasting blood sugar > 120 mg/dl", examples=[0])
    restecg: int = Field(ge=0, le=2, examples=[1])
    thalach: float = Field(gt=0, le=300, description="max heart rate achieved", examples=[150])
    exang: int = Field(ge=0, le=1, description="exercise-induced angina", examples=[1])
    oldpeak: float = Field(ge=-5, le=10, description="ST depression", examples=[1.5])
    slope: int | None = Field(default=None, ge=1, le=3, examples=[2])
    ca: float | None = Field(default=None, ge=0, le=3, description="major vessels", examples=[1])
    thal: float | None = Field(
        default=None, description="3=normal,6=fixed,7=reversible", examples=[7]
    )

    def to_frame(self) -> pd.DataFrame:
        """Single-row DataFrame with the exact columns the model expects."""
        return pd.DataFrame([{name: getattr(self, name) for name in ALL_FEATURES}])


class PredictionResponse(BaseModel):
    probability: float = Field(description="probability of heart disease (0-1)")
    at_risk: bool = Field(description="True if probability >= tuned threshold")
    threshold: float
    risk_band: str = Field(description="low | moderate | high")
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


@lru_cache(maxsize=1)
def _cached_bundle() -> ModelBundle:
    return load_bundle(MODEL_PATH)


def get_bundle() -> ModelBundle:
    """Dependency that returns the loaded model or a 503 if unavailable."""
    try:
        return _cached_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Train it first (`cardiorisk train`).",
        ) from exc


def _risk_band(probability: float) -> str:
    if probability < 0.34:
        return "low"
    if probability < 0.67:
        return "moderate"
    return "high"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe; reports whether a trained model is loaded."""
    try:
        bundle = _cached_bundle()
    except FileNotFoundError:
        return HealthResponse(status="ok", model_loaded=False)
    return HealthResponse(status="ok", model_loaded=True, model_name=bundle.model_name)


@app.post("/predict", response_model=PredictionResponse)
def predict(
    patient: PatientFeatures,
    bundle: Annotated[ModelBundle, Depends(get_bundle)],
) -> PredictionResponse:
    """Predict heart-disease risk for a single patient."""
    probability = float(bundle.predict_proba(patient.to_frame())[0])
    return PredictionResponse(
        probability=probability,
        at_risk=probability >= bundle.threshold,
        threshold=bundle.threshold,
        risk_band=_risk_band(probability),
        model_name=bundle.model_name,
    )
