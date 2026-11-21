"""Tests for single-race ingestion.

Pure-helper tests run anywhere. The ``integration`` test loads a real session
(cached after first run) and pins it to known historical facts, which also acts
as a regression guard on FastF1 data shape.
"""

from __future__ import annotations

import pandas as pd
import pytest

from f1_foundation.ingest import _contains, _seconds, load_race


def test_contains_detects_concatenated_track_status_codes() -> None:
    is_sc = _contains("4")
    assert is_sc("14") is True  # codes 1 and 4 both occurred this lap
    assert is_sc("1") is False
    assert is_sc("46") is True
    # Non-strings (e.g. pandas NA) must never raise and must be False.
    assert is_sc(pd.NA) is False
    assert is_sc(None) is False


def test_seconds_converts_timedeltas_and_preserves_nan() -> None:
    s = pd.Series(pd.to_timedelta(["0:01:28.179", None]))
    out = _seconds(s)
    assert out.iloc[0] == pytest.approx(88.179, abs=1e-3)
    assert pd.isna(out.iloc[1])


@pytest.mark.integration
def test_load_race_monza_2024_matches_known_facts() -> None:
    df = load_race(2024, "Monza")

    # Shape / coverage.
    assert df["driver"].nunique() == 20
    assert len(df) > 900
    assert df["circuit"].iloc[0] == "Monza"

    # Required columns the downstream stages depend on.
    for col in (
        "lap_number",
        "compound",
        "tyre_life",
        "is_in_lap",
        "under_caution",
        "laps_remaining",
        "lap_time_s",
    ):
        assert col in df.columns

    # Leclerc's winning one-stop: a single in-lap, on lap 15, off mediums.
    lec_pits = df.loc[(df["driver"] == "LEC") & df["is_in_lap"]]
    assert len(lec_pits) == 1
    assert int(lec_pits["lap_number"].iloc[0]) == 15
    assert lec_pits["compound"].iloc[0] == "MEDIUM"

    # Monza 2024 ran green flag-to-flag.
    assert int(df["flag_red"].sum()) == 0
    assert int(df["under_caution"].sum()) == 0

    # Leakage canary: an in-lap is mechanically slower than the green lap before
    # it (pit-lane time). This is *why* current-lap timing must never be a
    # feature — the test documents the trap as much as it guards it.
    lec = df.loc[df["driver"] == "LEC"].sort_values("lap_number")
    in_lap = lec.loc[lec["is_in_lap"]].iloc[0]
    prior = lec.loc[lec["lap_number"] == in_lap["lap_number"] - 1].iloc[0]
    assert in_lap["lap_time_s"] > prior["lap_time_s"] + 2.0
