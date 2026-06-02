"""Tests for the FastAPI serving layer.

A fixture injects lightweight models trained on synthetic data via FastAPI's
dependency override, so these run without the on-disk trained artifacts.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tire_strategy.data.features import DEFAULT_WET_COMPOUND, FEATURE_COLUMNS, SLICK_COMPOUNDS
from tire_strategy.models import build_compound_pipeline, build_pit_pipeline
from tire_strategy.serving.app import app, get_bundle
from tire_strategy.serving.models import ModelBundle

_VALID_STATE = {
    "lap_number": 14,
    "laps_remaining": 39,
    "tyre_age": 14,
    "stint_position": 14,
    "pit_count_so_far": 0,
    "track_temp": 48.0,
    "air_temp": 33.0,
    "compound": "MEDIUM",
    "circuit": "Monza",
    "gap_ahead_s": 2.9,
    "gap_behind_s": 1.0,
    "last_clean_lap_time_s": 85.5,
    "deg_slope_s_per_lap": 0.28,
    "relative_pace_s": 0.4,
}


def _synthetic_features(n: int = 120) -> pd.DataFrame:
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


@pytest.fixture
def client() -> Iterator[TestClient]:
    X = _synthetic_features()
    pit_model = build_pit_pipeline(scale_pos_weight=2.0, n_estimators=40)
    pit_model.fit(X, (X["tyre_age"] > 22).astype(int))
    compound_model = build_compound_pipeline(n_estimators=40)
    compound_model.fit(X, X["compound"])
    bundle = ModelBundle(pit_model, compound_model, 0.5, list(SLICK_COMPOUNDS))

    app.dependency_overrides[get_bundle] = lambda: bundle
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["models_loaded"], bool)


def test_predict_returns_valid_recommendation(client: TestClient) -> None:
    resp = client.post("/predict", json=_VALID_STATE)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["pit_probability"] <= 1.0
    assert isinstance(body["pit_recommended"], bool)
    assert body["compound_source"] == "model"
    assert body["recommended_compound"] in set(SLICK_COMPOUNDS)
    assert set(body["compound_probabilities"]) == set(SLICK_COMPOUNDS)
    assert body["compound_probabilities"][body["recommended_compound"]] == max(
        body["compound_probabilities"].values()
    )


def test_predict_gates_to_wet_tyre_on_cool_rainy_track(client: TestClient) -> None:
    """Raining + cool track -> rule-based wet tyre, regardless of the slick head."""
    resp = client.post("/predict", json={**_VALID_STATE, "rainfall": True, "track_temp": 18.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["compound_source"] == "wet-rule"
    assert body["recommended_compound"] == DEFAULT_WET_COMPOUND
    assert body["recommended_compound"] not in set(SLICK_COMPOUNDS)
    assert body["compound_probabilities"] == {DEFAULT_WET_COMPOUND: 1.0}
    # pit timing is unaffected by the gate — Head 1 still runs as normal.
    assert 0.0 <= body["pit_probability"] <= 1.0


def test_predict_defers_to_slick_head_on_warm_rainy_track(client: TestClient) -> None:
    """Raining but warm track (likely light/drying) -> defer to the slick head."""
    resp = client.post("/predict", json={**_VALID_STATE, "rainfall": True, "track_temp": 40.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["compound_source"] == "model"
    assert body["recommended_compound"] in set(SLICK_COMPOUNDS)
    assert set(body["compound_probabilities"]) == set(SLICK_COMPOUNDS)


def test_predict_rejects_invalid_input(client: TestClient) -> None:
    bad = {**_VALID_STATE, "lap_number": 0}  # violates ge=1
    assert client.post("/predict", json=bad).status_code == 422

    missing = {k: v for k, v in _VALID_STATE.items() if k != "compound"}
    assert client.post("/predict", json=missing).status_code == 422
