# Data Dictionary — UCI Heart Disease (combined)

Source: UCI Machine Learning Repository, "Heart Disease" (id 45), combined from
four databases: Cleveland, Hungary, Switzerland, and VA Long Beach (~920 rows).
The four `processed.*.data` files are downloaded and concatenated by
[`src/cardiorisk/data.py`](../src/cardiorisk/data.py); `?` is parsed as missing.

## Features

| Column     | Type        | Description | Encoding / units | Notes |
|------------|-------------|-------------|------------------|-------|
| `age`      | numeric     | Age | years | |
| `sex`      | categorical | Biological sex | 1 = male, 0 = female | |
| `cp`       | categorical | Chest pain type | 1 = typical angina, 2 = atypical, 3 = non-anginal, 4 = asymptomatic | strong predictor |
| `trestbps` | numeric     | Resting blood pressure | mm Hg (on admission) | |
| `chol`     | numeric     | Serum cholesterol | mg/dl | some missing |
| `fbs`      | categorical | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false | |
| `restecg`  | categorical | Resting ECG results | 0 = normal, 1 = ST-T abnormality, 2 = LV hypertrophy | |
| `thalach`  | numeric     | Maximum heart rate achieved | bpm | |
| `exang`    | categorical | Exercise-induced angina | 1 = yes, 0 = no | |
| `oldpeak`  | numeric     | ST depression induced by exercise relative to rest | mm | strong predictor |
| `slope`    | categorical | Slope of peak exercise ST segment | 1 = upsloping, 2 = flat, 3 = downsloping | heavily missing |
| `ca`       | categorical | Number of major vessels coloured by fluoroscopy | 0–3 | heavily missing |
| `thal`     | categorical | Thalassemia | 3 = normal, 6 = fixed defect, 7 = reversible defect | heavily missing |

## Label

| Column   | Type    | Description |
|----------|---------|-------------|
| `num`    | 0–4     | Raw disease severity (angiographic vessel narrowing class). |
| `target` | 0/1     | **Derived**: `1` if `num > 0` (disease present), else `0`. Created by `cardiorisk.target.binarize_target`. |

## Bookkeeping

| Column   | Description |
|----------|-------------|
| `source` | Originating database (`cleveland`, `hungarian`, `switzerland`, `va`). Dropped before modelling so it cannot leak. |

## Modelling treatment

- **Numeric** (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`): median
  imputation → standardisation.
- **Categorical** (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`,
  `thal`): most-frequent imputation → one-hot encoding (`handle_unknown=ignore`).
  `ca` and `thal` are treated as categorical (codes), not continuous counts, to
  avoid assuming a linear effect.

All preprocessing is fitted only on training folds, inside the model pipeline.
