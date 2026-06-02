"""Strategy label construction.

Turns the cleaned per-lap table from ``ingest.py`` into the two training targets
defined in ``docs/DESIGN.md`` §3, plus a training-eligibility mask:

- ``label_pit``     — 1 if the team made a *strategic* pit stop at the end of this
                      lap, else 0. This is Head 1 (the calibrated pit-timing model).
- ``label_compound``— the compound fitted at that stop (Head 2), only defined on
                      strategic pit laps; ``<NA>`` otherwise.
- ``train_eligible``/``exclude_reason`` — rows that are not honest decisions
                      (free tyre changes under red flag, a retiree's final lap)
                      are flagged out rather than silently corrupting the labels.

Everything here is a **pure** transformation of an ingested DataFrame, so the
labelling rules are unit-testable without touching the network.
"""

from __future__ import annotations

import pandas as pd

from f1_foundation import load_race, load_results, retired_drivers


def add_pit_and_compound_labels(laps: pd.DataFrame) -> pd.DataFrame:
    """Add ``label_pit`` and ``label_compound`` to a per-lap table.

    A *strategic* pit is an in-lap that is **not** a free change under red flag.
    The fitted compound is read from the very next lap (the out-lap on the new
    tyre), looked up per driver in lap order.
    """
    df = laps.sort_values(["driver", "lap_number"]).copy()

    is_in_lap = df["is_in_lap"].astype(bool)
    under_red = df["flag_red"].fillna(False).astype(bool)
    strategic_pit = is_in_lap & ~under_red

    df["label_pit"] = strategic_pit.astype("Int8")

    next_compound = df.groupby("driver", observed=True)["compound"].shift(-1)
    df["label_compound"] = next_compound.where(strategic_pit).astype("string")

    return df


def add_train_eligibility(laps: pd.DataFrame, retired: set[str] | None = None) -> pd.DataFrame:
    """Flag rows that are not honest strategy decisions.

    Excluded:
    - **red_flag_change** — an in-lap under red flag (tyres can be changed for
      free; not a strategic call).
    - **retirement** — the final recorded lap of a driver who did not finish
      (the car stopped; it was never a "stay out" decision).
    """
    df = laps.copy()
    df["train_eligible"] = True
    df["exclude_reason"] = pd.Series(pd.NA, index=df.index, dtype="string")

    def _exclude(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False).astype(bool)
        df.loc[mask, "train_eligible"] = False
        df.loc[mask, "exclude_reason"] = reason

    _exclude(
        df["is_in_lap"].astype(bool) & df["flag_red"].fillna(False).astype(bool),
        "red_flag_change",
    )

    if retired:
        last_lap = df.groupby("driver")["lap_number"].transform("max")
        is_final_lap = df["lap_number"] == last_lap
        _exclude(df["driver"].isin(list(retired)) & is_final_lap, "retirement")

    return df


def add_pit_window_label(
    df: pd.DataFrame,
    horizon: int = 3,
    group_cols: tuple[str, ...] = ("year", "round", "driver"),
) -> pd.DataFrame:
    """Add ``label_pit_window``: 1 if a strategic pit happens on this lap or within
    the next ``horizon - 1`` laps for the same car, else 0.

    A strategy decision has a *window* — boxing on lap 22 vs 24 is essentially the
    same call. The exact-lap ``label_pit`` is so sparse (~3%) that "never pit" is
    nearly unbeatable; predicting a window is both more learnable and closer to
    what a strategist actually wants ("the pit window is opening").

    This target is **forward-looking by design**. That is fine: it is the thing we
    predict, not a feature — features remain strictly leak-safe (see features.py).
    Grouping by race + driver prevents the window from spilling across races.
    """
    out = df.sort_values([*group_cols, "lap_number"]).copy()
    grouped = out.groupby(list(group_cols))["label_pit"]
    window = out["label_pit"].fillna(0).astype(int).astype(bool)
    for k in range(1, horizon):
        window = window | grouped.shift(-k).fillna(0).astype(int).astype(bool)
    out["label_pit_window"] = window.astype("Int8")
    return out


def build_labels(laps: pd.DataFrame, retired: set[str] | None = None) -> pd.DataFrame:
    """Compose label construction and eligibility flagging."""
    df = add_pit_and_compound_labels(laps)
    df = add_train_eligibility(df, retired=retired)
    return df.reset_index(drop=True)


def load_labelled_race(year: int, grand_prix: int | str, session_id: str = "R") -> pd.DataFrame:
    """Convenience: ingest one race, attach results-based DNF info, build labels."""
    laps = load_race(year, grand_prix, session_id)
    results = load_results(year, grand_prix, session_id)
    return build_labels(laps, retired=retired_drivers(results))
