"""Streamlit dashboard for CardioRisk.

Two tabs: an interactive risk calculator backed by the trained model bundle,
and a gallery of the evaluation/explainability artifacts produced by training.
Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from cardiorisk.persistence import load_bundle
from cardiorisk.preprocessing import ALL_FEATURES

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "model.joblib"

st.set_page_config(page_title="CardioRisk", page_icon="❤️", layout="wide")
st.title("❤️ CardioRisk — Heart Disease Risk")


@st.cache_resource
def _load():  # type: ignore[no-untyped-def]
    return load_bundle(MODEL_PATH)


def _risk_band(p: float) -> tuple[str, str]:
    if p < 0.34:
        return "Low", "🟢"
    if p < 0.67:
        return "Moderate", "🟡"
    return "High", "🔴"


calc_tab, report_tab = st.tabs(["Risk calculator", "Model report"])

with calc_tab:
    if not MODEL_PATH.exists():
        st.warning("No trained model found. Run `cardiorisk train` first.")
        st.stop()

    bundle = _load()
    st.caption(f"Model: **{bundle.model_name}** · decision threshold **{bundle.threshold:.2f}**")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 20, 90, 57)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda v: "Male" if v else "Female")
        cp = st.selectbox(
            "Chest pain type (cp)",
            [1, 2, 3, 4],
            format_func=lambda v: {
                1: "typical angina",
                2: "atypical angina",
                3: "non-anginal",
                4: "asymptomatic",
            }[v],
        )
        trestbps = st.slider("Resting BP (trestbps)", 80, 200, 140)
        chol = st.slider("Cholesterol (chol)", 100, 600, 260)
    with c2:
        fbs = st.selectbox("Fasting blood sugar > 120 (fbs)", [0, 1])
        restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
        thalach = st.slider("Max heart rate (thalach)", 70, 210, 150)
        exang = st.selectbox("Exercise-induced angina (exang)", [0, 1])
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.0, 1.5, 0.1)
    with c3:
        slope = st.selectbox("ST slope (slope)", [1, 2, 3])
        ca = st.selectbox("Major vessels (ca)", [0, 1, 2, 3])
        thal = st.selectbox(
            "Thalassemia (thal)",
            [3, 6, 7],
            format_func=lambda v: {3: "normal", 6: "fixed defect", 7: "reversible"}[v],
        )

    values = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    row = pd.DataFrame([{name: values[name] for name in ALL_FEATURES}])
    prob = float(bundle.predict_proba(row)[0])
    band, emoji = _risk_band(prob)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted probability", f"{prob:.1%}")
    m2.metric("Risk band", f"{emoji} {band}")
    m3.metric("At risk?", "Yes" if prob >= bundle.threshold else "No")
    st.progress(min(prob, 1.0))
    st.caption(
        "Decision uses the recall-tuned threshold so at-risk patients are rarely "
        "missed. This is a demonstration model, not medical advice."
    )

with report_tab:
    metrics_file = ARTIFACTS / "metrics.json"
    if metrics_file.exists():
        report = json.loads(metrics_file.read_text(encoding="utf-8"))
        st.subheader("Held-out test metrics (recall-tuned threshold)")
        st.json(report["test_metrics_tuned"])
        st.subheader("Cross-validation ranking (ROC-AUC)")
        st.dataframe(pd.DataFrame(report["cv_ranking"]), hide_index=True)
    plots = [
        ("ROC curve", "roc_curve.png"),
        ("Confusion matrix", "confusion_matrix.png"),
        ("Precision-Recall", "pr_curve.png"),
        ("Calibration", "calibration_curve.png"),
        ("Permutation importance", "permutation_importance.png"),
        ("SHAP summary", "shap_summary.png"),
    ]
    cols = st.columns(2)
    for i, (title, fname) in enumerate(plots):
        path = ARTIFACTS / fname
        if path.exists():
            with cols[i % 2]:
                st.image(str(path), caption=title, use_container_width=True)
