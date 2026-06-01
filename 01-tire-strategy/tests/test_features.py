"""Tests for leak-free feature engineering.

Pure tests build tiny tables so each rule is checked deterministically; the
integration test confirms the leakage discipline on a real race.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tire_strategy.data import build_features, load_labelled_race

_BOOL_COLS = [
    "is_in_lap",
    "is_out_lap",
    "under_caution",
    "is_accurate",
    "deleted",
    "fresh_tyre",
    "flag_sc",
    "flag_vsc",
    "flag_yellow",
]


def _make(records: list[dict]) -> pd.DataFrame:
    """Build a labelled-race-shaped frame, filling sane defaults for omitted cols."""
    df = pd.DataFrame(records)
    defaults: dict = {
        "year": 2024,
        "round": 1,
        "circuit": "Test",
        "stint": 1,
        "compound": "MEDIUM",
        "is_in_lap": False,
        "is_out_lap": False,
        "under_caution": False,
        "is_accurate": True,
        "deleted": False,
        "rainfall": False,
        "lap_time_s": 90.0,
        "track_temp": 40.0,
        "air_temp": 25.0,
        "position": 1,
        "fresh_tyre": False,
        "flag_sc": False,
        "flag_vsc": False,
        "flag_yellow": False,
        "label_pit": 0,
        "label_compound": pd.NA,
        "train_eligible": True,
        "exclude_reason": pd.NA,
    }
    for key, val in defaults.items():
        if key not in df.columns:
            df[key] = val
    if "time_s" not in df.columns:
        df["time_s"] = df["lap_number"] * 90.0
    if "tyre_life" not in df.columns:
        df["tyre_life"] = df["lap_number"]
    if "laps_remaining" not in df.columns:
        df["laps_remaining"] = df["lap_number"].max() - df["lap_number"]
    for col in _BOOL_COLS:
        df[col] = df[col].astype("boolean")
    df["compound"] = df["compound"].astype("string")
    df["driver"] = df["driver"].astype("string")
    df["stint"] = df["stint"].astype("Int64")
    df["tyre_life"] = df["tyre_life"].astype("Int64")
    return df


def test_stint_counters_and_second_compound_flag() -> None:
    df = _make(
        [
            {"driver": "A", "lap_number": 1, "stint": 1, "compound": "MEDIUM"},
            {"driver": "A", "lap_number": 2, "stint": 1, "compound": "MEDIUM"},
            {"driver": "A", "lap_number": 3, "stint": 2, "compound": "HARD"},
            {"driver": "A", "lap_number": 4, "stint": 2, "compound": "HARD"},
        ]
    )
    out = build_features(df).sort_values("lap_number")
    assert out["stint_position"].tolist() == [1, 2, 1, 2]
    assert out["pit_count_so_far"].tolist() == [0, 0, 1, 1]
    # Two distinct compounds are only "used" once HARD has actually been run.
    assert [bool(x) for x in out["used_second_compound"]] == [False, False, False, True]


def test_gaps_are_reconstructed_and_lagged_one_lap() -> None:
    # A leads B; B is 1s, 2s, 3s behind at the line on laps 1, 2, 3.
    df = _make(
        [
            {"driver": "A", "lap_number": 1, "position": 1, "time_s": 100.0},
            {"driver": "A", "lap_number": 2, "position": 1, "time_s": 200.0},
            {"driver": "A", "lap_number": 3, "position": 1, "time_s": 300.0},
            {"driver": "B", "lap_number": 1, "position": 2, "time_s": 101.0},
            {"driver": "B", "lap_number": 2, "position": 2, "time_s": 202.0},
            {"driver": "B", "lap_number": 3, "position": 2, "time_s": 303.0},
        ]
    )
    out = build_features(df)
    b = out[out["driver"] == "B"].sort_values("lap_number")
    # Lagged: lap 1 unknown; lap 2 = lap-1 gap (1.0); lap 3 = lap-2 gap (2.0).
    assert pd.isna(b["gap_ahead_s"].iloc[0])
    assert b["gap_ahead_s"].iloc[1] == pytest.approx(1.0)
    assert b["gap_ahead_s"].iloc[2] == pytest.approx(2.0)
    # The leader never has a car ahead.
    a = out[out["driver"] == "A"]
    assert a["gap_ahead_s"].isna().all()


def test_pace_and_degradation_use_only_past_laps() -> None:
    # A blatant outlier on the final lap must not leak into that lap's features.
    df = _make(
        [
            {"driver": "A", "lap_number": 1, "lap_time_s": 90.0},
            {"driver": "A", "lap_number": 2, "lap_time_s": 91.0},
            {"driver": "A", "lap_number": 3, "lap_time_s": 92.0},
            {"driver": "A", "lap_number": 4, "lap_time_s": 200.0},  # outlier (e.g. in-lap)
        ]
    )
    out = build_features(df).sort_values("lap_number").reset_index(drop=True)
    # Last clean lap time on lap 4 is lap 3's 92s, never the current 200s.
    assert out.loc[3, "last_clean_lap_time_s"] == pytest.approx(92.0)
    # Degradation slope on lap 4 is fitted on laps 1-3 (+1.0 s/lap), ignoring 200.
    assert out.loc[3, "deg_slope_s_per_lap"] == pytest.approx(1.0, abs=1e-6)
    # Not enough history on lap 1 -> undefined.
    assert pd.isna(out.loc[0, "deg_slope_s_per_lap"])


@pytest.mark.integration
def test_features_are_leak_free_on_monza_2024() -> None:
    df = build_features(load_labelled_race(2024, "Monza"))
    lec = df[df["driver"] == "LEC"].sort_values("lap_number")

    # Leclerc's in-lap (lap 15) is ~90s, but its pace feature must be a prior
    # green lap (~85s), proving current-lap timing never leaks in.
    lap15 = lec[lec["lap_number"] == 15].iloc[0]
    assert lap15["last_clean_lap_time_s"] < 87.0

    # First lap of a new stint has no in-stint history for the slope yet.
    out_lap = lec[lec["lap_number"] == 16].iloc[0]
    assert pd.isna(out_lap["deg_slope_s_per_lap"])
