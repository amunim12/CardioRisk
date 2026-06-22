# CardioRisk — Heart Disease Risk Prediction (Design Spec)

**Date:** 2026-06-22
**Status:** Approved (Architecture A)
**Author:** Engineering

## 1. Problem & objective

Predict whether a person is at risk of heart disease from routine clinical
measurements. This is a **binary classification** problem framed for a medical
context, where a **missed at-risk patient (false negative) is the costly
error**. The deliverable is a reproducible, production-grade ML system — not a
single notebook.

Origin: an introductory assignment ("Heart Disease Prediction") elevated to a
full MLOps showcase with high software-engineering standards.

## 2. Data

- **Source:** UCI Heart Disease dataset, combined multi-source version
  (Cleveland, Hungary, Switzerland, VA Long Beach), ~920 rows. Fetched
  programmatically via the `ucimlrepo` package (`id=45`) and cached to
  `data/raw/`. Network failure falls back to the cache with a clear message.
- **Target:** the UCI `num` field encodes disease severity 0–4. We binarize:
  `num > 0 -> 1` (disease present / at-risk), `num == 0 -> 0`. This is the
  standard binarization for this dataset.
- **Missing values:** the combined dataset has *real* missingness (notably
  `ca`, `thal`, `slope`, `chol`). Handled by a leak-free imputation step inside
  the modeling `Pipeline` (fit only on training folds), plus
  `HistGradientBoosting` which ingests NaNs natively.
- **Feature types:** numeric (`age`, `trestbps`, `chol`, `thalach`,
  `oldpeak`, ...) and categorical (`sex`, `cp`, `fbs`, `restecg`, `exang`,
  `slope`, `ca`, `thal`). Documented in `docs/DATA_DICTIONARY.md`.

## 3. Architecture (Approach A: modular package + config-driven CLI)

The entire preprocessing + estimator is a single serialized scikit-learn
`Pipeline`, eliminating train/serve skew. Configuration is YAML validated by
pydantic. A `typer` CLI is the operational entry point.

```
src/cardiorisk/
  config.py         # pydantic-settings, loads configs/*.yaml
  data.py           # fetch via ucimlrepo (id=45), cache to data/raw/
  target.py         # binarize UCI `num` (>0 -> 1)
  preprocessing.py  # ColumnTransformer: impute + scale + one-hot
  modeling.py       # build LogReg / DecisionTree / HistGB pipelines
  evaluate.py       # metrics + ROC/PR/confusion/calibration plots
  explain.py        # coefficients + permutation importance + SHAP (guarded)
  cli.py            # typer: train | evaluate | predict | serve
  api.py            # FastAPI: /health, /predict (Pydantic schema)
app/streamlit_app.py
notebooks/01_eda.ipynb
tests/
configs/default.yaml
artifacts/          # model.joblib, metrics.json, *.png (generated)
```

### Data flow

`ucimlrepo` → cache raw → binarize target → stratified train/test split →
`Pipeline(ColumnTransformer → estimator)` → 5-fold stratified CV model
selection on ROC-AUC → fit best on train → evaluate on held-out test → tune a
recall-oriented decision threshold → persist pipeline + metrics + plots →
MLflow logs every run.

## 4. Modeling

Three models, all wrapped in pipelines so CV is honest:

1. **LogisticRegression** — headline interpretable model (coefficients = log-odds).
2. **DecisionTree** — human-readable rule view.
3. **HistGradientBoostingClassifier** — accuracy benchmark; native NaN handling.

Selection by mean stratified 5-fold ROC-AUC; ties broken by recall.

## 5. Evaluation & medical framing

Saved as JSON + PNG artifacts:

- Accuracy, precision, recall, F1, **ROC-AUC**, **PR-AUC**.
- **ROC curve**, **confusion matrix**, **calibration curve**.
- **Decision-threshold tuning**: report an operating threshold chosen for high
  recall (sensitivity), not just the default 0.5, because false negatives are
  clinically costly. Discussed in `MODEL_CARD.md`.

## 6. Explainability / feature importance

Three lenses, robust to environment:

- **Model coefficients** (LogisticRegression) — direction + magnitude.
- **Permutation importance** (model-agnostic, always available) — primary.
- **SHAP** — beam plot if `shap` imports; import is guarded so a failed build
  on Python 3.14 degrades gracefully rather than breaking the pipeline.

## 7. Serving & UI

- **FastAPI** (`api.py`): `/health` and `/predict` with a Pydantic request
  schema, structured validation errors, loads the serialized pipeline.
- **Streamlit** (`app/streamlit_app.py`): interactive risk scoring + plot
  gallery.

## 8. Engineering standards

- **Packaging:** PEP 621 `pyproject.toml`, src-layout, `pip install -e .`.
  Local env is plain `venv` + `pip` on the host Python 3.14.
- **Reproducibility:** pinned `requirements.txt` / `requirements-dev.txt`;
  global `random_state`; cached raw data.
- **Quality gates:** `ruff` (lint+format), `mypy` (types), `pytest`
  (incl. FastAPI `TestClient`), wired into `pre-commit` and **GitHub Actions**.
- **Containerization:** `Dockerfile` on `python:3.12-slim` (host-independent,
  sidesteps any 3.14 wheel gaps for the deployable artifact) +
  `docker-compose.yaml` for api + ui.
- **Docs:** `README.md` (badges, mermaid architecture diagram), `MODEL_CARD.md`
  (responsible-AI: intended use, metrics, limitations, ethics), and
  `docs/DATA_DICTIONARY.md`.

## 9. Error handling

- Data fetch: network failure → cached file → clear error if neither exists.
- SHAP: guarded import; fall back to permutation importance.
- API: Pydantic validation → structured 422; missing model → clear 503.
- Config: validated on load; fail fast with actionable messages.

## 10. Testing strategy

Unit tests for: target binarization, preprocessing column routing, data schema,
metric computation, threshold tuning, and an API smoke test via `TestClient`.
TDD where it adds value (pure logic: target, metrics, threshold).

## 11. Out of scope (YAGNI)

No orchestration framework (Airflow/Kedro), no cloud deployment, no model
registry beyond MLflow's local file store, no automated retraining/monitoring
service (monitoring is documented as a future hook, not built).

## 12. Success criteria

- One command fetches data, trains, evaluates, and writes artifacts.
- ROC-AUC, confusion matrix, ROC curve, calibration, and feature-importance
  artifacts are generated and non-trivial (test ROC-AUC clearly > 0.5; target
  ~0.85+ for HistGB on this dataset).
- API and Streamlit app load the trained pipeline and return predictions.
- `ruff`, `mypy`, `pytest` pass; CI is green.
