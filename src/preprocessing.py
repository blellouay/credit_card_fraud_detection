# src/preprocessing.py

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, StandardScaler


# =========================
# Cleaning
# =========================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# =========================
# Feature functions
# =========================

def log_transform(x):
    return np.log1p(x)


def hour_sin_transform(x):
    hour = (x // 3600) % 24
    return np.sin(2 * np.pi * hour / 24)


def hour_cos_transform(x):
    hour = (x // 3600) % 24
    return np.cos(2 * np.pi * hour / 24)


# =========================
# Transformers
# =========================

log_amount = FunctionTransformer(log_transform, validate=False)

hour_sin = FunctionTransformer(hour_sin_transform, validate=False)
hour_cos = FunctionTransformer(hour_cos_transform, validate=False)

scaled_log_amount = Pipeline([
    ("log", FunctionTransformer(log_transform, validate=False)),
    ("scale", StandardScaler())
])


# =========================
# Preprocessing pipelines
# =========================

# 1. Raw features: no preprocessing
raw_pipeline = ColumnTransformer(
    transformers=[],
    remainder="passthrough"
)


# 2. Tree-based models:
# Log amount + cyclic time, no scaling
tree_time_cyclic_pipeline = ColumnTransformer(
    transformers=[
        ("amount_log", log_amount, ["Amount"]),
        ("hour_sin", hour_sin, ["Time"]),
        ("hour_cos", hour_cos, ["Time"]),
    ],
    remainder="passthrough"
)


# 3. Linear models:
# Log amount + scaled amount + scaled Time
linear_scaled_pipeline = ColumnTransformer(
    transformers=[
        ("amount_log_scaled", scaled_log_amount, ["Amount"]),
        ("time_scaled", StandardScaler(), ["Time"]),
    ],
    remainder="passthrough"
)


# 4. Linear models with cyclic time:
# Log amount scaled + cyclic time + scaled remaining features
linear_time_cyclic_scaled_pipeline = ColumnTransformer(
    transformers=[
        ("amount_log_scaled", scaled_log_amount, ["Amount"]),
        ("hour_sin", hour_sin, ["Time"]),
        ("hour_cos", hour_cos, ["Time"]),
        ("v_features_scaled", StandardScaler(), [f"V{i}" for i in range(1, 29)]),
    ],
    remainder="drop"
)


# =========================
# Registry
# =========================

PREPROCESSING_PIPELINES = {
    "raw": raw_pipeline,

    # use for Random Forest / XGBoost
    "tree_time_cyclic": tree_time_cyclic_pipeline,

    # use for Logistic Regression / SVM / KNN
    "linear_scaled": linear_scaled_pipeline,
    "linear_time_cyclic_scaled": linear_time_cyclic_scaled_pipeline,
}