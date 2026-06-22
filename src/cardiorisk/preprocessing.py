"""Feature preprocessing.

All preprocessing lives inside a scikit-learn ``ColumnTransformer`` so it is
fitted only on training folds during cross-validation (no leakage) and travels
with the model as a single serialisable object (no train/serve skew).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Continuous measurements: median-imputed and standardised (scaling matters
# for the linear model and is harmless for the tree-based ones).
NUMERIC_FEATURES: list[str] = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Coded clinical categories (including ``ca`` vessel count and ``thal``):
# mode-imputed and one-hot encoded. Treating these as categorical avoids
# assuming a linear effect of arbitrary integer codes.
CATEGORICAL_FEATURES: list[str] = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build the leak-free preprocessing transformer.

    - numeric: median imputation -> standardisation
    - categorical: most-frequent imputation -> one-hot (unknowns ignored)
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
