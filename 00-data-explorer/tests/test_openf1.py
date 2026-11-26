"""OpenF1 payloads -> the same race table FastF1 ingestion produces (offline)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from f1_foundation.circuits import canonical_circuit
from f1_foundation.openf1 import calendar_from_sessions, race_table, results_table

T0 = datetime(2026, 11, 8, 17, 0, tzinfo=UTC)


def _laps() -> list[dict]:
    out = []
    for number, pace in ((1, 90.0), (4, 90.5)):
        t = T0
        for lap in range(1, 9):
            duration = pace + (22 if lap == 4 and number == 1 else 0) + (30 if lap in (6, 7) else 0)
            out.append(
                {
                    "driver_number": number,
                    "lap_number": lap,
                    "date_start": t.isoformat(),
                    "lap_duration": None if lap == 1 else duration,
                    "is_pit_out_lap": number == 1 and lap == 5,
                }
            )
            t += timedelta(seconds=duration)
    return out


PAYLOAD = {
    "stints": [
        {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 4,
         "compound": "MEDIUM", "tyre_age_at_start": 0},
        {"driver_number": 1, "stint_number": 2, "lap_start": 5, "lap_end": None,
         "compound": "HARD", "tyre_age_at_start": 3},
        {"driver_number": 4, "stint_number": 1, "lap_start": 1, "lap_end": 8,
         "compound": "SOFT", "tyre_age_at_start": 0},
    ],
    "pits": [{"driver_number": 1, "lap_number": 4, "pit_duration": 23.1}],
    "race_control": [
        {"date": "2026-11-08T17:09:00", "category": "SafetyCar", "lap_number": 6,
         "message": "SAFETY CAR DEPLOYED"},
        {"date": "2026-11-08T17:12:00", "category": "SafetyCar", "lap_number": 7,
         "message": "SAFETY CAR IN THIS LAP"},
        {"date": "2026-11-08T17:02:00", "category": "Flag", "flag": "YELLOW", "lap_number": 2,
         "message": "YELLOW IN TRACK SECTOR 4"},
    ],
    "weather": [
        {"date": "2026-11-08T16:59:00+00:00", "air_temperature": 20.1,
         "track_temperature": 41.0, "rainfall": 0, "humidity": 60, "pressure": 930,
         "wind_speed": 1.0, "wind_direction": 100},
    ],
    "drivers": [
        {"driver_number": 1, "name_acronym": "VER", "team_name": "Red Bull Racing"},
        {"driver_number": 4, "name_acronym": "NOR", "team_name": "McLaren"},
    ],
}  # fmt: skip


def _table() -> pd.DataFrame:
    return race_table(_laps(), **PAYLOAD, year=2026, round_number=21, circuit="São Paulo")


def test_race_table_tyres_pits_and_flags() -> None:
    df = _table()
    ver = df[df["driver"] == "VER"].set_index("lap_number")
    assert ver.loc[4, "is_in_lap"] and ver.loc[5, "is_out_lap"]
    assert ver.loc[4, "compound"] == "MEDIUM" and ver.loc[5, "compound"] == "HARD"
    assert ver.loc[1, "tyre_life"] == 1 and ver.loc[5, "tyre_life"] == 4  # used hards
    assert not bool(ver.loc[5, "fresh_tyre"])
    assert ver.loc[6, "flag_sc"] and ver.loc[7, "under_caution"] and not ver.loc[8, "flag_sc"]
    assert ver.loc[2, "flag_yellow"]
    assert ver.loc[3, "track_temp"] == 41.0
    assert df["total_laps"].iloc[0] == 8 and ver.loc[8, "laps_remaining"] == 0
    assert pd.isna(ver.loc[1, "lap_time_s"]) and ver.loc[2, "lap_time_s"] == 90.0


def test_laps_without_start_time_are_kept() -> None:
    laps = _laps()
    laps[2]["date_start"] = None  # VER lap 3: no start time (seen in real OpenF1 data)
    laps[0]["date_start"] = None  # VER lap 1: nothing before it to infer from
    df = race_table(laps, **PAYLOAD, year=2026, round_number=21, circuit="São Paulo")
    ver = df[df["driver"] == "VER"].set_index("lap_number")
    assert len(ver) == 8
    assert ver.loc[3, "track_temp"] == 41.0  # start time inferred from lap 2
    assert ver.loc[4, "is_in_lap"]


def test_positions_follow_line_crossings() -> None:
    df = _table()
    lap3 = df[df["lap_number"] == 3].set_index("driver")["position"]
    assert lap3["VER"] == 1 and lap3["NOR"] == 2
    lap5 = df[df["lap_number"] == 5].set_index("driver")["position"]
    assert lap5["NOR"] == 1  # VER lost ~22 s in the pits


def test_table_feeds_labels_and_features() -> None:
    from tire_strategy.data.features import build_features
    from tire_strategy.data.labels import build_labels

    feats = build_features(build_labels(_table(), retired=set()))
    pit = feats[(feats["driver"] == "VER") & (feats["label_pit"] == 1)]
    assert pit["lap_number"].tolist() == [4]
    assert pit["label_compound"].tolist() == ["HARD"]


def test_calendar_skips_cancelled_races_and_maps_names() -> None:
    sessions = [
        {"session_name": "Race", "session_key": 1, "location": "Melbourne",
         "date_start": "2026-03-08T04:00:00+00:00"},
        {"session_name": "Race", "session_key": 2, "location": "Sakhir", "is_cancelled": True,
         "date_start": "2026-04-12T15:00:00+00:00"},
        {"session_name": "Race", "session_key": 3, "location": "Miami Gardens",
         "date_start": "2026-05-03T20:00:00+00:00"},
        {"session_name": "Sprint", "session_key": 4, "location": "Miami Gardens",
         "date_start": "2026-05-02T16:00:00+00:00"},
    ]  # fmt: skip
    cal = calendar_from_sessions(sessions, 2026)
    assert [(r.round, r.circuit) for r in cal] == [(1, "Melbourne"), (2, "Miami")]


def test_results_mark_retirements() -> None:
    res = results_table(
        [{"driver_number": 1, "position": 1}, {"driver_number": 4, "position": None, "dnf": True}],
        PAYLOAD["drivers"],
    )
    assert res.set_index("driver")["classified"].to_dict() == {"VER": True, "NOR": False}


def test_canonical_names() -> None:
    assert canonical_circuit("Yas Marina") == "Yas Island" == canonical_circuit("Yas Island")
    assert canonical_circuit("Monza") == "Monza"
