"""Tests for the FastAPI service via TestClient."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cardiorisk import api
from cardiorisk.modeling import build_model
from cardiorisk.persistence import ModelBundle, save_bundle
from cardiorisk.preprocessing import ALL_FEATURES

VALID_PATIENT = {
    "age": 60,
    "sex": 1,
    "cp": 4,
    "trestbps": 145,
    "chol": 270,
    "fbs": 0,
    "restecg": 1,
    "thalach": 140,
    "exang": 1,
    "oldpeak": 2.3,
    "slope": 2,
    "ca": 2,
    "thal": 7,
}


@pytest.fixture
def trained_model_path(tmp_path: Path, features_target: tuple[pd.DataFrame, pd.Series]) -> Path:
    X, y = features_target
    pipeline = build_model("logistic_regression", seed=0).fit(X, y)
    bundle = ModelBundle(
        pipeline=pipeline,
        threshold=0.5,
        feature_names=ALL_FEATURES,
        model_name="logistic_regression",
    )
    path = tmp_path / "model.joblib"
    save_bundle(bundle, path)
    return path


@pytest.fixture
def client_with_model(
    trained_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(api, "MODEL_PATH", str(trained_model_path))
    api._cached_bundle.cache_clear()
    yield TestClient(api.app)
    api._cached_bundle.cache_clear()


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(api, "MODEL_PATH", str(tmp_path / "missing.joblib"))
    api._cached_bundle.cache_clear()
    yield TestClient(api.app)
    api._cached_bundle.cache_clear()


def test_health_reports_model_loaded(client_with_model: TestClient) -> None:
    resp = client_with_model.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "logistic_regression",
    }


def test_health_ok_without_model(client_without_model: TestClient) -> None:
    resp = client_without_model.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is False


def test_predict_returns_valid_response(client_with_model: TestClient) -> None:
    resp = client_with_model.post("/predict", json=VALID_PATIENT)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["at_risk"], bool)
    assert body["risk_band"] in {"low", "moderate", "high"}


def test_predict_allows_missing_optional_fields(client_with_model: TestClient) -> None:
    payload = {k: v for k, v in VALID_PATIENT.items() if k not in {"ca", "thal", "chol", "slope"}}
    resp = client_with_model.post("/predict", json=payload)
    assert resp.status_code == 200


def test_predict_rejects_out_of_range(client_with_model: TestClient) -> None:
    bad = {**VALID_PATIENT, "sex": 5}
    resp = client_with_model.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_503_without_model(client_without_model: TestClient) -> None:
    resp = client_without_model.post("/predict", json=VALID_PATIENT)
    assert resp.status_code == 503
