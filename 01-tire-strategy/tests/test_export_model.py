"""The JSON model export must reproduce the real models' predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from tire_strategy.data.features import BOOLEAN_FEATURES, NUMERIC_FEATURES
from tire_strategy.export_model import export_models, predict_compound, predict_pit
from tire_strategy.models import build_compound_pipeline, build_pit_pipeline
from tire_strategy.models.pipeline import COMPOUND_FEATURES


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(20, 8, n) for c in NUMERIC_FEATURES})
    for c in BOOLEAN_FEATURES:
        df[c] = rng.random(n) < 0.2
    df["compound"] = rng.choice(["SOFT", "MEDIUM", "HARD"], n)
    df["circuit"] = rng.choice(["Monza", "Mexico City", "São Paulo"], n)
    df["team"] = rng.choice(["Ferrari", "McLaren"], n)
    df.loc[rng.random(n) < 0.15, "gap_ahead_s"] = np.nan  # missing values
    return df


@pytest.fixture(scope="module")
def models():
    train, cal = _frame(1500, 1), _frame(600, 2)
    y = ((train["tyre_age"] > 24) ^ (train["compound"] == "SOFT")).astype(int)
    pipe = build_pit_pipeline(n_estimators=40)
    pipe.fit(train, y)
    yc = ((cal["tyre_age"] > 24) ^ (cal["compound"] == "SOFT")).astype(int)
    pit = CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic").fit(cal, yc)
    comp = build_compound_pipeline(features=COMPOUND_FEATURES, n_estimators=30)
    labels = np.where(
        train["track_temp"] > 25, "HARD", np.where(train["tyre_age"] > 20, "MEDIUM", "SOFT")
    )
    comp.fit(train, labels)
    return pit, comp, export_models(pit, comp, threshold=0.3)


def test_pit_probabilities_match(models) -> None:
    pit, _, exported = models
    test = _frame(200, 3)
    expected = pit.predict_proba(test)[:, 1]
    got = [predict_pit(exported, row) for row in test.to_dict("records")]
    assert np.allclose(got, expected, atol=1e-5)


def test_compound_probabilities_match(models) -> None:
    _, comp, exported = models
    test = _frame(200, 4)
    expected = comp.predict_proba(test)
    got = [predict_compound(exported, row) for row in test.to_dict("records")]
    classes = list(comp.named_steps["clf"].classes_)
    arr = np.array([[g[c] for c in classes] for g in got])
    assert np.allclose(arr, expected, atol=1e-5)
