"""sklearn pipelines for the two model heads.

Each head is a full ``Pipeline``: coerce the feature frame to model dtypes ->
one-hot the compound -> LightGBM. Bundling preprocessing into the estimator means
the exact same transformations run at training and at serving time, so the API
can't silently disagree with training (a classic production bug).

LightGBM handles missing values natively, so numeric NaNs (e.g. an undefined
degradation slope early in a stint) are passed through rather than imputed.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from tire_strategy.config import SEED
from tire_strategy.data.features import (
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)


def _num(series: pd.Series) -> pd.Series:
    """Coerce a (possibly nullable) column to plain float64, NA -> NaN."""
    numeric = pd.to_numeric(cast(pd.Series, series).astype("object"), errors="coerce")
    return cast(pd.Series, numeric).astype("float64")


def _to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the feature columns and coerce them to model-ready dtypes."""
    X = cast(pd.DataFrame, df.loc[:, FEATURE_COLUMNS]).copy()
    for col in NUMERIC_FEATURES + BOOLEAN_FEATURES:
        X[col] = _num(cast(pd.Series, X[col]))
    for col in CATEGORICAL_FEATURES:
        X[col] = cast(pd.Series, X[col]).astype("object")
    return X


def _preprocessor() -> ColumnTransformer:
    """One-hot the compound; pass numeric/boolean features through unchanged."""
    transformer = ColumnTransformer(
        transformers=[
            (
                "compound",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # Emit a named DataFrame so the model keeps feature names (readable importances,
    # no name-mismatch warnings between fit and predict).
    return transformer.set_output(transform="pandas")


def _coerce_step() -> FunctionTransformer:
    return FunctionTransformer(_to_model_frame, feature_names_out="one-to-one")


def _pipeline(clf: LGBMClassifier) -> Pipeline:
    return Pipeline([("coerce", _coerce_step()), ("pre", _preprocessor()), ("clf", clf)])


def build_pit_pipeline(scale_pos_weight: float = 1.0, **overrides: object) -> Pipeline:
    """Calibratable LightGBM pipeline for the binary pit-timing head.

    ``scale_pos_weight`` counters the ~30:1 stay-out imbalance without resampling
    (which would distort calibration).
    """
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    if overrides:
        clf.set_params(**overrides)
    return _pipeline(clf)


def build_compound_pipeline(**overrides: object) -> Pipeline:
    """LightGBM pipeline for the conditional compound-choice head (slicks only)."""
    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    if overrides:
        clf.set_params(**overrides)
    return _pipeline(clf)
