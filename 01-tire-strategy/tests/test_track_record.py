"""Scoring plans against what the top ten did, and the live prediction log."""

from __future__ import annotations

import pandas as pd

from tire_strategy.strategy import track_record as tr
from tire_strategy.strategy.params import fit_strategy_params

from .strategy_fixtures import synthetic_race_laps


def _race(sc_laps=()) -> pd.DataFrame:
    laps = synthetic_race_laps(races=((2026, 2),), sc_laps=sc_laps)
    return laps


def test_actual_strategies_counts_stops() -> None:
    actual = tr.actual_strategies(_race())
    assert actual is not None and len(actual) == 10
    assert {a["stops"] for a in actual} <= {1, 2}
    assert all(len(a["pit_laps"]) == a["stops"] for a in actual)


def test_wet_race_is_not_scored() -> None:
    race = _race()
    race["rainfall"] = True
    assert tr.actual_strategies(race) is None


def test_score_and_caution_flag() -> None:
    actual = tr.actual_strategies(_race(sc_laps=(13, 14, 15)))
    assert actual is not None
    plan = {"label": "M-H", "stops": 1, "pit_laps": [22], "windows": [[20, 25]]}
    s = tr.score(plan, actual)
    assert s["caution_shaped"] is True  # several stopped on laps 14-15 under the SC
    assert s["predicted_first_stop"] == 22
    assert s["actual_stops"] in (1, 2)
    assert isinstance(s["stops_match"], bool)


def test_summary_rates() -> None:
    rows = [
        {"caution_shaped": False, "stops_match": True, "baseline_one_stop": False,
         "baseline_last_time": None, "in_window": True, "first_stop_error": 2.0},
        {"caution_shaped": True, "stops_match": False, "baseline_one_stop": True,
         "baseline_last_time": True, "in_window": False, "first_stop_error": -6.0},
    ]  # fmt: skip
    s = tr.summarise(rows)
    assert s["races"] == 2 and s["stops_match"] == 0.5
    assert s["baseline_last_time"] == 1.0
    assert s["first_stop_error_median"] == 4.0
    assert s["clean"]["races"] == 1 and s["clean"]["stops_match"] == 1.0


def test_backtest_uses_only_earlier_races() -> None:
    races = tuple((2024, r) for r in range(1, 12))
    laps = synthetic_race_laps(races=races, drivers=12)
    result = tr.backtest(laps, since=2024)
    # The first eight races have too little history to plan from.
    assert [r["round"] for r in result["races"]] == [9, 10, 11]
    assert result["summary"]["races"] == 3


def test_prediction_log_adds_once_then_scores() -> None:
    laps = synthetic_race_laps()
    params = fit_strategy_params(laps)
    calendar = {"next": {"year": 2026, "round": 9, "circuit": "Testville",
                         "date": "2026-11-29T16:00:00Z"}}  # fmt: skip
    log = tr.update_predictions([], params, calendar, laps, "2026-11-24T08:00")
    assert len(log) == 1 and "result" not in log[0]
    again = tr.update_predictions(log, params, calendar, laps, "2026-11-25T08:00")
    assert len(again) == 1 and again[0]["made_at"] == "2026-11-24T08:00"
    raced = synthetic_race_laps(races=((2026, 9),))
    scored = tr.update_predictions(again, params, None, raced, "2026-12-01T08:00")
    assert "stops_match" in scored[0]["result"]
