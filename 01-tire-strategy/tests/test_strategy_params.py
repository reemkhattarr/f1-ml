"""Fitting the strategy parameters recovers the tyre behaviour races were built with."""

from __future__ import annotations

import json

import pytest

from tire_strategy.strategy.params import fit_strategy_params

from .strategy_fixtures import TRUE_DEG, TRUE_OFFSET, TRUE_PIT_LOSS, synthetic_race_laps


@pytest.fixture(scope="module")
def params() -> dict:
    return fit_strategy_params(synthetic_race_laps(sc_laps=(31, 32, 33)))


def test_degradation_recovered(params) -> None:
    deg = params["circuits"]["Testville"]["degradation"]
    for compound, true in TRUE_DEG.items():
        assert deg[compound] == pytest.approx(true, abs=0.012)


def test_compound_offsets_recovered(params) -> None:
    off = params["circuits"]["Testville"]["offsets"]
    for compound, true in TRUE_OFFSET.items():
        assert off[compound] == pytest.approx(true, abs=0.15)
    assert params["track_evolution_s_per_lap"] < 0


def test_pit_loss_and_safety_car(params) -> None:
    c = params["circuits"]["Testville"]
    assert c["pit_loss"] == pytest.approx(TRUE_PIT_LOSS, abs=1.5)
    assert c["race_laps"] == 50
    assert 0 < c["p_safety_car"] <= 1
    assert c["tyre_life"]["HARD"] >= 28


def test_params_are_json(params) -> None:
    assert json.loads(json.dumps(params))["circuits"]["Testville"]["races"] == 3
