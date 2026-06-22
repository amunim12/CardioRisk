# Model Card — CardioRisk

A model card following the spirit of Mitchell et al. (2019). This documents what
the model is, how it was evaluated, and — importantly for a health context —
where it must **not** be trusted.

## Model details

- **Task:** binary classification — presence vs. absence of heart disease.
- **Selected model:** Logistic Regression inside a scikit-learn `Pipeline`
  (median impute + standardise numeric; most-frequent impute + one-hot
  categorical). Selected automatically by 5-fold stratified CV ROC-AUC against
  a Decision Tree and HistGradientBoosting.
- **Output:** probability of disease in [0, 1] plus a binary decision using a
  tuned threshold.
- **Version:** 0.1.0.

## Intended use

- **Intended:** education and demonstration of an end-to-end, interpretable
  clinical-style ML workflow; a teaching example of recall-aware thresholding
  and feature attribution.
- **Out of scope:** real clinical decision-making, triage, or diagnosis. This
  model is trained on small, decades-old, multi-site research data and is **not
  a medical device.**

## Training data

- **Source:** UCI Heart Disease (combined Cleveland + Hungary + Switzerland +
  VA Long Beach), 920 rows. Label `num` (0-4 severity) binarised to `num > 0`.
- **Split:** 736 train / 184 test, stratified, seed 42.
- **Missingness:** real and substantial in `ca`, `thal`, `slope`, `chol`;
  handled by in-pipeline imputation (see [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md)).

## Decision threshold

A missed at-risk patient (false negative) is the costly error, so the operating
threshold is **not** 0.5. We tune it on out-of-fold training predictions to
maximise recall subject to precision ≥ 0.70, yielding a threshold ≈ **0.20**.
The threshold is chosen without ever touching the test set.

## Evaluation (held-out test set, n = 184)

| Metric    | Default thr 0.50 | Recall-tuned thr 0.20 |
|-----------|------------------|-----------------------|
| ROC-AUC   | 0.909            | 0.909                 |
| PR-AUC    | 0.920            | 0.920                 |
| Accuracy  | 0.842            | 0.788                 |
| Precision | 0.848            | 0.727                 |
| Recall    | 0.873            | **0.990**             |
| F1        | 0.860            | 0.838                 |

Cross-validation ROC-AUC (model selection): Logistic Regression 0.885 ± 0.022,
HistGradientBoosting 0.853 ± 0.030, Decision Tree 0.815 ± 0.035.

ROC-AUC/PR-AUC are threshold-independent and unchanged; tuning trades precision
for near-total recall. Diagnostic plots are in `artifacts/`.

## Most influential features

Permutation importance and SHAP agree on the top drivers: chest-pain type (`cp`),
ST depression (`oldpeak`), cholesterol (`chol`), sex, exercise-induced angina
(`exang`), number of major vessels (`ca`), and max heart rate (`thalach`) —
all clinically plausible.

## Limitations & ethical considerations

- **Population:** four hospitals, largely older patients, collected in the late
  1980s–90s. Performance will not transfer to modern or different populations.
- **Sex imbalance:** the source data skews male; calibration and error rates may
  differ by sex and were not formally audited for fairness here.
- **Calibration:** see `artifacts/calibration_curve.png`; probabilities are
  approximate and should not be read as clinical risk percentages.
- **Imputation:** heavy missingness in some cohorts means imputed values carry
  uncertainty not reflected in the point prediction.
- **No causal claims:** feature importances are associational, not causal.

## How to reproduce

`docker run --rm -v "$PWD/artifacts:/app/artifacts" cardiorisk cardiorisk train`
(seed 42). Metrics are written to `artifacts/metrics.json`.
