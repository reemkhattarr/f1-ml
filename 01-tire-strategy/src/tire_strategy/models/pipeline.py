"""sklearn pipelines for the two model heads.

Each head is a full ``Pipeline``: coerce the feature frame to model dtypes ->
one-hot the categorical features -> LightGBM. Bundling preprocessing into the
estimator means the exact same transformations run at training and serving time,
so the API can't silently disagree with training (a classic production bug).

The feature set is described by a :class:`FeatureSpec`, so experiments and
ablations can add/remove features without editing the pipeline internals.
LightGBM handles missing values natively, so numeric NaNs (e.g. an undefined
degradation slope early in a stint) are passed through rather than imputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
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
    NUMERIC_FEATURES,
)


@dataclass(frozen=True)
class FeatureSpec:
    """The columns that feed a model, grouped by how they are preprocessed."""

    numeric: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    boolean: tuple[str, ...] = tuple(BOOLEAN_FEATURES)

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.boolean]


DEFAULT_FEATURES = FeatureSpec()

# The compound head additionally uses the constructor (`team`): which compound a
# team fits is mildly team-flavoured (a measured 5/5-fold accuracy win), whereas
# *when* to pit is not — so the pit head deliberately stays on DEFAULT_FEATURES.
COMPOUND_FEATURES = FeatureSpec(categorical=(*CATEGORICAL_FEATURES, "team"))


def _num(series: pd.Series) -> pd.Series:
    """Coerce a (possibly nullable) column to plain float64, NA -> NaN."""
    numeric = pd.to_numeric(cast(pd.Series, series).astype("object"), errors="coerce")
    return cast(pd.Series, numeric).astype("float64")


def _to_model_frame(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Select the spec's columns and coerce them to model-ready dtypes."""
    X = cast(pd.DataFrame, df.loc[:, spec.columns]).copy()
    for col in (*spec.numeric, *spec.boolean):
        X[col] = _num(cast(pd.Series, X[col]))
    for col in spec.categorical:
        X[col] = cast(pd.Series, X[col]).astype("object")
    return X


def _preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    """One-hot the categorical features; pass numeric/boolean through unchanged."""
    transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(spec.categorical),
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    # Emit a named DataFrame so the model keeps feature names (readable importances,
    # no name-mismatch warnings between fit and predict).
    return transformer.set_output(transform="pandas")


def _coerce_step(spec: FeatureSpec) -> FunctionTransformer:
    # Selects a subset of columns (not one-to-one), returning a named DataFrame so
    # the downstream ColumnTransformer can still select categoricals by name.
    return FunctionTransformer(partial(_to_model_frame, spec=spec))


def _pipeline(clf: LGBMClassifier, spec: FeatureSpec) -> Pipeline:
    return Pipeline([("coerce", _coerce_step(spec)), ("pre", _preprocessor(spec)), ("clf", clf)])


def build_pit_pipeline(
    scale_pos_weight: float = 1.0,
    features: FeatureSpec = DEFAULT_FEATURES,
    **overrides: object,
) -> Pipeline:
    """Calibratable LightGBM pipeline for the binary pit-timing head.

    ``scale_pos_weight`` counters the stay-out imbalance without resampling (which
    would distort calibration).
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
    return _pipeline(clf, features)


def build_compound_pipeline(
    features: FeatureSpec = DEFAULT_FEATURES, **overrides: object
) -> Pipeline:
    """LightGBM pipeline for the conditional compound-choice head (slicks only)."""
    clf = LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight="balanced",  # lift minority SOFT/MEDIUM (HARD is ~53% of pits)
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    if overrides:
        clf.set_params(**overrides)
    return _pipeline(clf, features)
