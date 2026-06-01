"""Tests for baselines and the model pipelines (construction + fit/predict)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tire_strategy.data.features import FEATURE_COLUMNS, SLICK_COMPOUNDS
from tire_strategy.models import (
    MajorityBaseline,
    RuleBasedBaseline,
    build_compound_pipeline,
    build_pit_pipeline,
)


def _feature_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    age = rng.integers(1, 40, n)
    df = pd.DataFrame(
        {
            "lap_number": rng.integers(1, 50, n),
            "laps_remaining": rng.integers(0, 50, n),
            "tyre_age": age,
            "stint_position": age,
            "pit_count_so_far": rng.integers(0, 3, n),
            "track_temp": rng.uniform(20, 50, n),
            "air_temp": rng.uniform(15, 35, n),
            "gap_ahead_s": rng.uniform(0, 20, n),
            "gap_behind_s": rng.uniform(0, 20, n),
            "last_clean_lap_time_s": rng.uniform(80, 95, n),
            "deg_slope_s_per_lap": rng.uniform(-0.3, 0.5, n),
            "relative_pace_s": rng.uniform(-1.5, 2.5, n),
            "compound": rng.choice(SLICK_COMPOUNDS, n),
            "circuit": rng.choice(["Monza", "Monaco", "Spa"], n),
            "fresh_tyre": rng.integers(0, 2, n).astype(bool),
            "under_caution": rng.integers(0, 2, n).astype(bool),
            "flag_sc": rng.integers(0, 2, n).astype(bool),
            "flag_vsc": np.zeros(n, dtype=bool),
            "flag_yellow": rng.integers(0, 2, n).astype(bool),
            "used_second_compound": rng.integers(0, 2, n).astype(bool),
            "rainfall": np.zeros(n, dtype=bool),
            "sc_just_deployed": rng.integers(0, 2, n).astype(bool),
        }
    )
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    return df


def test_majority_baseline_always_stays_out() -> None:
    X = _feature_frame()
    preds = MajorityBaseline().predict(X)
    assert preds.shape == (len(X),)
    assert (preds == 0).all()


def test_rule_based_baseline_pits_old_tyres() -> None:
    X = pd.DataFrame(
        {
            "compound": ["SOFT", "SOFT", "HARD"],
            "tyre_age": [3, 30, 5],  # only the 30-lap soft is past its age
            "under_caution": [False, False, False],
        }
    )
    preds = RuleBasedBaseline().predict(X)
    assert preds.tolist() == [0, 1, 0]

    # A worn tyre under a safety car is a cheap stop, even below the age limit.
    X_sc = pd.DataFrame({"compound": ["HARD"], "tyre_age": [12], "under_caution": [True]})
    assert RuleBasedBaseline().predict(X_sc).tolist() == [1]


def test_pit_pipeline_fits_and_predicts_proba() -> None:
    X = _feature_frame()
    y = (X["tyre_age"] > 22).astype(int)  # a learnable, balanced-ish target
    model = build_pit_pipeline(scale_pos_weight=2.0, n_estimators=40)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_compound_pipeline_fits_three_classes() -> None:
    X = _feature_frame()
    y = X["compound"]  # predict the (synthetic) compound
    model = build_compound_pipeline(n_estimators=40)
    model.fit(X, y)
    preds = model.predict(X)
    assert set(preds).issubset(set(SLICK_COMPOUNDS))


def test_cross_validate_pit_runs_and_reports_baselines() -> None:
    from tire_strategy.models import cross_validate_pit

    rng = np.random.default_rng(1)
    X = _feature_frame(180)
    X["year"] = 2024
    X["round"] = rng.integers(1, 7, len(X))  # 6 race-weekend groups
    X["label_pit_window"] = (X["tyre_age"] > 20).astype(int)

    result = cross_validate_pit(X, n_splits=3, params={"n_estimators": 40})
    assert len(result["folds"]) == 3
    assert {"ap_mean", "model_cost_mean", "majority_cost_mean", "rule_cost_mean"} <= set(
        result["agg"]
    )


def test_cross_validate_compound_reports_macro_f1_and_baseline() -> None:
    from tire_strategy.models import cross_validate_compound

    rng = np.random.default_rng(2)
    X = _feature_frame(180)
    X["year"] = 2024
    X["round"] = rng.integers(1, 7, len(X))
    X["label_pit"] = 1  # all rows are pit-laps for this test
    X["label_compound"] = X["compound"]  # learnable target

    result = cross_validate_compound(X, n_splits=3)
    assert {"accuracy_mean", "macro_f1_mean", "baseline_accuracy_mean", "n_pits"} <= set(result)
    assert 0.0 <= result["macro_f1_mean"] <= 1.0
    assert result["n_pits"] == len(X)
