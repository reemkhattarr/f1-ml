"""Tests for strategy label construction.

Pure tests run on a hand-built toy table so the labelling rules are verified
without the network. The integration test pins the rules to real Monza 2024 facts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tire_strategy.data import build_labels, load_labelled_race


def _toy() -> pd.DataFrame:
    """A minimal table exercising every labelling branch.

    - A: clean one-stop (in-lap on lap 3, fits HARD on the out-lap).
    - B: in-lap under red flag (a free change — must not count as a strategic pit).
    - C: a retiree (its final lap must be excluded).
    """
    rows = [
        # driver, lap, is_in_lap, flag_red, compound
        ("A", 1, False, False, "MEDIUM"),
        ("A", 2, False, False, "MEDIUM"),
        ("A", 3, True, False, "MEDIUM"),  # strategic pit -> fits HARD next lap
        ("A", 4, False, False, "HARD"),  # out-lap
        ("A", 5, False, False, "HARD"),
        ("B", 1, False, False, "SOFT"),
        ("B", 2, True, True, "SOFT"),  # red-flag change -> excluded, not a pit
        ("B", 3, False, False, "MEDIUM"),
        ("C", 1, False, False, "HARD"),
        ("C", 2, False, False, "HARD"),  # final lap of a retiree -> excluded
    ]
    df = pd.DataFrame(rows, columns=["driver", "lap_number", "is_in_lap", "flag_red", "compound"])
    df["is_in_lap"] = df["is_in_lap"].astype(bool)
    df["flag_red"] = df["flag_red"].astype("boolean")
    df["compound"] = df["compound"].astype("string")
    return df


def test_strategic_pit_and_fitted_compound() -> None:
    out = build_labels(_toy(), retired={"C"})
    a3 = out[(out["driver"] == "A") & (out["lap_number"] == 3)].iloc[0]
    assert a3["label_pit"] == 1
    assert a3["label_compound"] == "HARD"  # the tyre fitted on the out-lap
    # Every non-pit A lap is a "stay out".
    a_other = out[(out["driver"] == "A") & (out["lap_number"] != 3)]
    assert (a_other["label_pit"] == 0).all()
    assert a_other["label_compound"].isna().all()


def test_red_flag_change_is_not_a_strategic_pit() -> None:
    out = build_labels(_toy(), retired={"C"})
    b2 = out[(out["driver"] == "B") & (out["lap_number"] == 2)].iloc[0]
    assert b2["label_pit"] == 0
    assert pd.isna(b2["label_compound"])
    assert b2["train_eligible"] is False or b2["train_eligible"] == False  # noqa: E712
    assert b2["exclude_reason"] == "red_flag_change"


def test_retiree_final_lap_excluded() -> None:
    out = build_labels(_toy(), retired={"C"})
    c = out[out["driver"] == "C"].sort_values("lap_number")
    assert c.iloc[-1]["exclude_reason"] == "retirement"
    assert bool(c.iloc[-1]["train_eligible"]) is False
    # The earlier C lap is still a valid stay-out decision.
    assert bool(c.iloc[0]["train_eligible"]) is True


@pytest.mark.integration
def test_labelled_monza_2024_matches_known_facts() -> None:
    df = load_labelled_race(2024, "Monza")

    # Leclerc's winning one-stop: a single strategic pit, lap 15, fitting hards.
    lec_pits = df[(df["driver"] == "LEC") & (df["label_pit"] == 1)]
    assert len(lec_pits) == 1
    assert int(lec_pits["lap_number"].iloc[0]) == 15
    assert lec_pits["label_compound"].iloc[0] == "HARD"

    # Tsunoda retired -> his final lap is the only retirement exclusion type present.
    assert "retirement" in set(df["exclude_reason"].dropna())
    tsu_excluded = df[(df["driver"] == "TSU") & (df["exclude_reason"] == "retirement")]
    assert len(tsu_excluded) == 1
