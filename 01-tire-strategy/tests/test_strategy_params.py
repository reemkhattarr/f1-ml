"""Fitting the strategy parameters: wear recovered, physical limits always hold."""

from __future__ import annotations

import json

import pytest

from tire_strategy.strategy.params import (
    OFFSET_DATA_WEIGHT,
    STANDARD_OFFSETS,
    check_params,
    fit_strategy_params,
)

from .strategy_fixtures import TRUE_DEG, TRUE_OFFSET, TRUE_PIT_LOSS, synthetic_race_laps


@pytest.fixture(scope="module")
def params() -> dict:
    return fit_strategy_params(synthetic_race_laps(sc_laps=(31, 32, 33)))


def test_degradation_recovered(params) -> None:
    deg = params["circuits"]["Testville"]["degradation"]
    for compound, true in TRUE_DEG.items():
        assert deg[compound] == pytest.approx(true, abs=0.015)


def test_offsets_are_standard_moved_toward_data(params) -> None:
    off = params["circuits"]["Testville"]["offsets"]
    w = OFFSET_DATA_WEIGHT
    for compound in ("SOFT", "HARD"):
        expected = (1 - w) * STANDARD_OFFSETS[compound] + w * TRUE_OFFSET[compound]
        assert off[compound] == pytest.approx(expected, abs=0.08)
    assert params["pace"]["measured_offsets"]["SOFT"] == pytest.approx(-0.6, abs=0.15)


def test_pit_loss_and_safety_car(params) -> None:
    c = params["circuits"]["Testville"]
    assert c["pit_loss"] == pytest.approx(TRUE_PIT_LOSS, abs=1.5)
    assert c["race_laps"] == 50
    assert 0 < c["p_safety_car"] <= 1
    assert c["tyre_life"]["HARD"] >= 28


def test_physical_limits_hold(params) -> None:
    assert check_params(params) == []
    life = params["circuits"]["Testville"]["tyre_life"]
    assert life["SOFT"] <= life["MEDIUM"] <= life["HARD"]


def test_sanity_check_catches_impossible_values(params) -> None:
    bad = json.loads(json.dumps(params))
    c = bad["circuits"]["Testville"]
    c["offsets"]["SOFT"] = 0.2
    c["degradation"]["HARD"] = -0.01
    c["tyre_life"]["SOFT"] = 60
    problems = check_params(bad)
    assert len(problems) == 3
    assert all(p.startswith("Testville") for p in problems)


def test_reversed_data_is_pulled_back_to_physics() -> None:
    """If races suggest hards are faster and wear faster, the limits still hold."""
    laps = synthetic_race_laps()
    hard = laps["compound"] == "HARD"
    laps.loc[hard, "lap_time_s"] -= 1.0 + 0.05 * laps.loc[hard, "tyre_life"].astype(float)
    p = fit_strategy_params(laps)
    assert check_params(p) == []
    assert "hard pace" in p["pace"]["limits_applied"] or p["default"]["offsets"]["HARD"] > 0


def test_lusail_gets_tyre_rule_note() -> None:
    p = fit_strategy_params(synthetic_race_laps(circuit="Lusail"))
    assert "Pirelli" in p["circuits"]["Lusail"]["note"]


def test_params_are_json(params) -> None:
    assert json.loads(json.dumps(params))["circuits"]["Testville"]["races"] == 3
