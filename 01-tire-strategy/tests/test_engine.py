"""Planner and advisor behaviour on simple, hand-checkable parameters."""

from __future__ import annotations

import itertools

import pytest

from tire_strategy.strategy.engine import advise, circuit_params, plan, race_plan, stint_cost

PARAMS = {
    "default": {
        "degradation": {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.04},
        "offsets": {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.4},
        "tyre_life": {"SOFT": 20, "MEDIUM": 30, "HARD": 40},
        "pit_loss": 22.0, "p_caution": 0.5, "p_safety_car": 0.3, "caution_per_lap": 0.01,
    },
    "circuits": {"Fastville": {"pit_loss": 15.0, "race_laps": 40}},
    "sc_pit_factor": 0.5,
    "vsc_pit_factor": 0.65,
}  # fmt: skip
P = circuit_params(PARAMS, None)


def _brute(p, total, seq):
    best = None
    for cut in itertools.combinations(range(1, total), len(seq) - 1):
        bounds = (0, *cut, total)
        lengths = [b - a for a, b in itertools.pairwise(bounds)]
        if any(n > p["tyre_life"][c] for c, n in zip(seq, lengths, strict=True)):
            continue
        cost = sum(stint_cost(p, c, n) for c, n in zip(seq, lengths, strict=True))
        best = cost if best is None or cost < best else best
    return best + (len(seq) - 1) * p["pit_loss"]


def test_planner_is_exact_against_brute_force() -> None:
    options = plan(P, 45, max_stops=2)
    for o in options[:6]:
        seq = tuple(s["compound"] for s in o["stints"])
        assert o["cost"] == pytest.approx(_brute(P, 45, seq), abs=0.01)


def test_two_compound_rule_and_tyre_life() -> None:
    for o in plan(P, 50):
        compounds = {s["compound"] for s in o["stints"]}
        assert len(compounds) >= 2
        assert all(s["laps"] <= P["tyre_life"][s["compound"]] for s in o["stints"])
        assert sum(s["laps"] for s in o["stints"]) == 50
    no_rule = plan(P, 15, two_compound_rule=False)
    assert no_rule[0]["stops"] == 0


def test_cheaper_stops_mean_more_stops() -> None:
    expensive = race_plan(circuit_params(PARAMS, None), 40)["best"]
    cheap = race_plan(circuit_params(PARAMS, "Fastville"), 40)["best"]
    assert cheap["stops"] >= expensive["stops"]
    assert all(
        lo <= lap <= hi for lap, (lo, hi) in zip(cheap["pit_laps"], cheap["windows"], strict=True)
    )


def test_best_is_the_cheapest_plan() -> None:
    for pit in (8.0, 15.0, 30.0):
        p = dict(P, pit_loss=pit)
        result = race_plan(p, 50)
        assert result["best"]["cost"] == min(o["cost"] for o in result["by_stops"])
        assert result["best"]["delta"] == 0 and "windows" in result["best"]


def test_advisor_boxes_on_dead_tyres_and_under_safety_car() -> None:
    dead = advise(P, total_laps=50, lap=31, compound="MEDIUM", tyre_age=30)
    assert dead["decision"] == "box"
    fresh = advise(P, total_laps=50, lap=6, compound="MEDIUM", tyre_age=5, used=["SOFT"])
    assert fresh["decision"] == "stay"
    green = advise(P, total_laps=50, lap=14, compound="MEDIUM", tyre_age=13)
    sc = advise(P, total_laps=50, lap=14, compound="MEDIUM", tyre_age=13, caution="sc")
    assert sc["delta"] > green["delta"]  # a safety car makes boxing relatively better
    assert sc["pit_loss_now"] == 11.0


def test_advisor_rain_rule() -> None:
    wet = advise(P, total_laps=50, lap=20, compound="SOFT", tyre_age=4, rain=True)
    assert (wet["decision"], wet["compound"], wet["source"]) == ("box", "INTERMEDIATE", "wet rule")


def test_finished_race_is_an_error() -> None:
    with pytest.raises(ValueError):
        advise(P, total_laps=50, lap=51, compound="SOFT", tyre_age=4)
