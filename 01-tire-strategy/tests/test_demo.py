"""Tests for the Gradio demo's data/prediction logic (not the UI wiring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tire_strategy.data.features import SLICK_COMPOUNDS
from tire_strategy.models import build_compound_pipeline, build_pit_pipeline
from tire_strategy.serving.demo import drivers_in, lap_prediction, probability_curve
from tire_strategy.serving.models import ModelBundle


def _synthetic_bundle() -> ModelBundle:
    """A tiny bundle trained on synthetic data — no dependency on trained artifacts."""
    rng = np.random.default_rng(0)
    age = rng.integers(1, 40, 120)
    X = pd.DataFrame(
        {
            "lap_number": rng.integers(1, 50, 120),
            "laps_remaining": rng.integers(0, 50, 120),
            "tyre_age": age,
            "stint_position": age,
            "pit_count_so_far": rng.integers(0, 3, 120),
            "track_temp": rng.uniform(20, 50, 120),
            "air_temp": rng.uniform(15, 35, 120),
            "gap_ahead_s": rng.uniform(0, 20, 120),
            "gap_behind_s": rng.uniform(0, 20, 120),
            "last_clean_lap_time_s": rng.uniform(80, 95, 120),
            "deg_slope_s_per_lap": rng.uniform(-0.3, 0.5, 120),
            "relative_pace_s": rng.uniform(-1.5, 2.5, 120),
            "compound": rng.choice(SLICK_COMPOUNDS, 120),
            "circuit": rng.choice(["Monza", "Monaco", "Spa"], 120),
            "fresh_tyre": rng.integers(0, 2, 120).astype(bool),
            "under_caution": rng.integers(0, 2, 120).astype(bool),
            "flag_sc": rng.integers(0, 2, 120).astype(bool),
            "flag_vsc": np.zeros(120, dtype=bool),
            "flag_yellow": rng.integers(0, 2, 120).astype(bool),
            "used_second_compound": rng.integers(0, 2, 120).astype(bool),
            "rainfall": np.zeros(120, dtype=bool),
        }
    )
    pit = build_pit_pipeline(scale_pos_weight=2.0, n_estimators=30).fit(
        X, (X["tyre_age"] > 22).astype(int)
    )
    compound = build_compound_pipeline(n_estimators=30).fit(X, X["compound"])
    return ModelBundle(pit, compound, 0.3, list(SLICK_COMPOUNDS))


@pytest.mark.integration
def test_demo_recommendation_on_monza_2024() -> None:
    bundle = _synthetic_bundle()
    assert "LEC" in drivers_in(2024, "Monza")

    pred = lap_prediction(bundle, 2024, "Monza", "LEC", 14)
    assert 0.0 <= pred["pit_probability"] <= 1.0
    assert pred["recommended_compound"] in set(SLICK_COMPOUNDS)
    assert isinstance(pred["actual_pit"], bool)
    # Leclerc was on mediums on lap 14 (pre his lap-15 stop).
    assert pred["compound"] == "MEDIUM"

    fig = probability_curve(bundle, 2024, "Monza", "LEC")
    assert fig is not None

    # An out-of-range lap yields no data.
    assert lap_prediction(bundle, 2024, "Monza", "LEC", 999) == {}
