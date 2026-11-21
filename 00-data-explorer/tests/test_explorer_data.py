"""Budget/skip behaviour of the explorer builder, with FastF1 faked out.

No network: ``fastf1.get_event_schedule`` and ``_extract_session`` are replaced
by fakes that count calls, so the tests check *which* sessions get API calls.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastf1.exceptions import RateLimitExceededError

from f1_foundation import explorer_data as ed

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=UTC)


def _schedule(year: int, rounds: dict[int, datetime]) -> pd.DataFrame:
    """One 'Race' session per round, starting at the given time."""
    return pd.DataFrame(
        [
            {
                "RoundNumber": rnd,
                "Session1": "Race",
                "Session1DateUtc": pd.Timestamp(start).tz_convert(None),
            }
            for rnd, start in rounds.items()
        ]
    )


def _laps(year: int, rnd: int) -> dict[str, pd.DataFrame]:
    meta = {"year": year, "round": rnd, "circuit": "X", "session": "R"}
    return {"laps": pd.DataFrame([{"LapNumber": 1, **meta}])}


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "raw_path": tmp_path / "raw_laps.parquet",
        "weather_path": tmp_path / "weather.parquet",
        "results_path": tmp_path / "results.parquet",
        "unavailable_path": tmp_path / "explorer_unavailable.json",
    }


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ed, "enable_cache", lambda: None)


def _fake(monkeypatch, schedules, behaviour, budget=None):
    """behaviour[(year, rnd)] -> 'ok' | 'missing'. Returns the list of calls made."""
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        ed.fastf1, "get_event_schedule", lambda y, include_testing=False: schedules[y]
    )

    def extract(year, rnd, _name):
        if budget is not None and len(calls) >= budget:
            raise RateLimitExceededError("any API: 500 calls/h")
        calls.append((year, rnd))
        if behaviour[(year, rnd)] == "ok":
            return _laps(year, rnd)
        raise ValueError("no data")

    monkeypatch.setattr(ed, "_extract_session", extract)
    return calls


def test_new_rounds_first_and_future_sessions_cost_nothing(monkeypatch, paths) -> None:
    old = NOW - timedelta(days=200)
    schedules = {
        2025: _schedule(2025, {1: old, 2: old}),
        2026: _schedule(2026, {1: NOW - timedelta(days=3), 2: NOW + timedelta(days=5)}),
    }
    behaviour = {(2025, 1): "ok", (2025, 2): "ok", (2026, 1): "ok", (2026, 2): "ok"}
    calls = _fake(monkeypatch, schedules, behaviour, budget=1)

    res = ed.build_explorer_data(seasons=(2025, 2026), now=NOW, **paths)

    assert calls == [(2026, 1)]  # newest completed round first; future round never called
    assert res.added == 1 and res.rate_limited


def test_dead_sessions_are_remembered_and_never_retried(monkeypatch, paths) -> None:
    old = NOW - timedelta(days=400)
    schedules = {2025: _schedule(2025, {1: old, 2: old, 3: NOW - timedelta(days=2)})}
    behaviour = {(2025, 1): "missing", (2025, 2): "ok", (2025, 3): "missing"}

    calls = _fake(monkeypatch, schedules, behaviour)
    first = ed.build_explorer_data(seasons=(2025,), now=NOW, **paths)
    assert first.added == 1 and first.unavailable == 1  # round 3 is too recent to give up on
    assert json.loads(paths["unavailable_path"].read_text()) == [[2025, 1, "R"]]
    assert sorted(calls) == [(2025, 1), (2025, 2), (2025, 3)]

    calls = _fake(monkeypatch, schedules, behaviour)
    second = ed.build_explorer_data(seasons=(2025,), now=NOW, **paths)
    assert calls == [(2025, 3)]  # built and dead sessions cost nothing; recent one retried
    assert second.added == 0 and second.unavailable == 0 and not second.rate_limited


def test_result_line_is_parseable() -> None:
    line = ed.BuildResult(added=2, unavailable=1, rate_limited=True).line()
    assert line == "RESULT added=2 unavailable=1 rate_limited=1"
