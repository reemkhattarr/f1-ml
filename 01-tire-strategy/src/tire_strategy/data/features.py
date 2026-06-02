"""Leak-free feature engineering.

Builds the model inputs from ``docs/DESIGN.md`` §5 out of a labelled per-race
table. The single most important rule (the leakage register, §4) is enforced
*structurally* here:

- **State known at the start of the decision lap** (tyre age, compound, laps
  remaining, temperatures, caution status) uses the current lap's values.
- **Anything derived from lap timing** (gaps, recent pace, degradation slope) is
  computed from *completed* laps only and **lagged by one lap**, because the
  decision lap's own time is contaminated — most obviously on an in-lap, where it
  includes pit-lane time.

Features are built per race (gaps and degradation need within-race context). At
serving time these same quantities arrive in the request JSON, so the model
itself only ever sees the per-row feature vector.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

# Window (in clean laps) used to estimate the in-stint degradation slope.
_DEG_WINDOW = 5

# Approximate fuel effect: a car gets ~this many seconds/lap faster per lap of
# fuel burned (heavier car = slower). Lap times therefore fall through a stint as
# fuel burns, masking tyre degradation; we subtract this before fitting the
# degradation slope so it reflects tyre wear, not fuel. A deliberate domain
# constant (~0.05 s/kg x ~1.6 kg/lap), not a fitted value.
FUEL_S_PER_LAP = 0.055

# Dry-weather slick compounds. The compound-choice model is restricted to these:
# wet-weather tyre changes (INTERMEDIATE/WET) are driven by rain, not strategy.
SLICK_COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

# Wet-weather tyres. The compound head deliberately does NOT model these. In the
# wet the tyre choice is not a strategic slick decision but a rule (raining -> fit
# a wet tyre), handled by a deterministic rain gate at serving time (see
# serving/recommend.py). We cannot learn inter-vs-full-wet either: that split is
# decided by standing water / rain intensity, which our boolean ``rainfall``
# feature cannot express, and full WET appears at only ~13 strategic stops across
# 2018-2026 (vs ~327 INTERMEDIATE). So the gate defaults to INTERMEDIATE.
WET_COMPOUNDS: tuple[str, ...] = ("INTERMEDIATE", "WET")
DEFAULT_WET_COMPOUND: str = "INTERMEDIATE"

# Track temperature (deg C) below which a rainy pit is treated as a genuine wet
# stop (-> INTERMEDIATE); at or above it the rain is more likely light/drying and
# teams fit slicks, so the gate defers to the slick head. Track temp is the single
# strongest *available* discriminator of wet-vs-slick at a rainy pit (warm track
# sheds water): adding it lifts the gate's wet-regime accuracy ~0.61 -> ~0.70
# (leak-free nested CV) and halves its variance — measured, see docs §9.2 / §12 #20.
# A documented domain constant: accuracy is flat across ~24-28 C, so 26 is not a
# load-bearing knife-edge (cf. FUEL_S_PER_LAP).
WET_TRACK_TEMP_C: float = 26.0

NUMERIC_FEATURES: list[str] = [
    "lap_number",
    "laps_remaining",
    "tyre_age",
    "stint_position",
    "pit_count_so_far",
    "track_temp",
    "air_temp",
    "gap_ahead_s",
    "gap_behind_s",
    "last_clean_lap_time_s",
    "deg_slope_s_per_lap",
    "relative_pace_s",
]
CATEGORICAL_FEATURES: list[str] = ["compound", "circuit"]
BOOLEAN_FEATURES: list[str] = [
    "fresh_tyre",
    "under_caution",
    "flag_sc",
    "flag_vsc",
    "flag_yellow",
    "used_second_compound",
    "rainfall",
    "sc_just_deployed",
]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

# Columns carried alongside the features for splitting, evaluation and auditing.
# `team` is carried so the compound head can use the constructor (a measured 5/5-fold
# accuracy win for compound choice; see docs/IMPLEMENTATION.md §12 #16).
_CARRY: list[str] = [
    "year",
    "round",
    "circuit",
    "driver",
    "team",
    "lap_number",
    "stint",
    "label_pit",
    "label_compound",
    "train_eligible",
    "exclude_reason",
]


def _bool(df: pd.DataFrame, name: str) -> pd.Series:
    """Read a nullable-boolean column as plain bool with NA treated as False."""
    return cast(pd.Series, df[name]).fillna(False).astype(bool)


def _clean_lap_mask(df: pd.DataFrame) -> pd.Series:
    """Laps whose pace is representative (usable for pace/degradation features)."""
    return (
        ~_bool(df, "is_in_lap")
        & ~_bool(df, "is_out_lap")
        & ~_bool(df, "under_caution")
        & _bool(df, "is_accurate")
        & ~_bool(df, "deleted")
        & cast(pd.Series, df["lap_time_s"]).notna()
    )


def _add_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Within-stint counters and the mandatory-compound flag (all lap-start safe)."""
    df = df.copy()
    df["tyre_age"] = df["tyre_life"]
    df["stint_position"] = df.groupby(["driver", "stint"]).cumcount() + 1
    df["pit_count_so_far"] = (df["stint"].astype("Int64") - 1).clip(lower=0)

    # used_second_compound: have >= 2 distinct compounds been run *before* this lap?
    first_use = ~df.duplicated(["driver", "compound"])
    distinct_incl = first_use.groupby(df["driver"]).cumsum()
    distinct_before = distinct_incl - first_use.astype(int)
    df["used_second_compound"] = (distinct_before >= 2).astype("boolean")
    return df


def _add_gap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct gap to car ahead/behind at the line, then lag by one lap.

    For each lap, cars are ordered by classified position; the gap is the
    difference in their line-crossing session time. Lapped traffic / missing
    neighbours fall out as NaN. The result is lagged one lap so the decision lap's
    (possibly pit-contaminated) own time is never used.
    """
    df = df.sort_values(["lap_number", "position"]).copy()
    by_lap = df.groupby("lap_number")["time_s"]
    gap_ahead_raw = by_lap.diff()  # time to the car one position ahead
    gap_behind_raw = -by_lap.diff(-1)  # time to the car one position behind

    df["_gap_ahead_raw"] = gap_ahead_raw
    df["_gap_behind_raw"] = gap_behind_raw

    df = df.sort_values(["driver", "lap_number"])
    df["gap_ahead_s"] = df.groupby("driver")["_gap_ahead_raw"].shift(1)
    df["gap_behind_s"] = df.groupby("driver")["_gap_behind_raw"].shift(1)
    return df.drop(columns=["_gap_ahead_raw", "_gap_behind_raw"])


def _add_pace_features(df: pd.DataFrame, window: int = _DEG_WINDOW) -> pd.DataFrame:
    """Most-recent clean lap time and in-stint degradation slope (past laps only)."""
    df = df.sort_values(["driver", "lap_number"]).copy()
    clean = _clean_lap_mask(df)

    # Last representative lap time strictly before the current lap.
    clean_lt = df["lap_time_s"].where(clean)
    df["_clean_lt"] = clean_lt
    shifted = df.groupby("driver")["_clean_lt"].shift(1)
    df["last_clean_lap_time_s"] = shifted.groupby(df["driver"]).ffill()

    # Field-relative pace: the car's most recent clean lap minus the field's clean
    # median that lap (positive = slower than rivals -> a pit signal). Built from
    # the field median per lap, then lagged per driver so it stays leak-safe.
    # (build_features runs per race, so grouping by lap/driver stays within-race.)
    field_median = df.groupby("lap_number")["_clean_lt"].transform("median")
    df["_rel"] = df["_clean_lt"] - field_median
    rel_shift = df.groupby("driver")["_rel"].shift(1)
    df["relative_pace_s"] = rel_shift.groupby(df["driver"]).ffill()

    # Degradation slope: fit fuel-corrected lap time ~ tyre_age over the last
    # `window` clean laps of the current stint, using only laps before the current
    # one. Fuel correction (see FUEL_S_PER_LAP) removes the stint-long speed-up
    # from fuel burn so the slope reflects tyre wear rather than a lightening car.
    df["_clean"] = clean
    slope = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, grp in df.groupby(["driver", "stint"], sort=False):
        xs: list[float] = []
        ys: list[float] = []
        for idx in grp.index:
            if len(xs) >= 2:
                x = xs[-window:]
                y = ys[-window:]
                slope.loc[idx] = float(np.polyfit(x, y, 1)[0])
            is_usable = (
                bool(grp.loc[idx, "_clean"])
                and pd.notna(grp.loc[idx, "tyre_life"])
                and pd.notna(grp.loc[idx, "laps_remaining"])
            )
            if is_usable:
                corrected = float(grp.loc[idx, "lap_time_s"]) - FUEL_S_PER_LAP * float(
                    grp.loc[idx, "laps_remaining"]
                )
                xs.append(float(grp.loc[idx, "tyre_life"]))
                ys.append(corrected)
    df["deg_slope_s_per_lap"] = slope
    return df.drop(columns=["_clean_lt", "_clean", "_rel"])


def _add_safety_car_features(df: pd.DataFrame) -> pd.DataFrame:
    """The safety-car *deployment moment* — the real pit trigger. A static "is there
    a caution" flag is weaker than "the caution came out *this* lap", which is when
    the cheap-stop window opens (pit rate ~22% vs ~3% otherwise). Computed per driver
    within the race; it describes the live state at the decision lap, so leak-safe.
    """
    df = df.sort_values(["driver", "lap_number"]).copy()
    caution = _bool(df, "under_caution")
    prev = _bool(df.assign(_p=df.groupby("driver")["under_caution"].shift(1)), "_p")
    df["sc_just_deployed"] = (caution & ~prev).astype("boolean")
    return df


def build_features(labelled: pd.DataFrame) -> pd.DataFrame:
    """Build the §5 feature table from a single labelled race.

    Returns the feature columns (``FEATURE_COLUMNS``) plus carried identifiers and
    labels. One row per lap; leak-safe by construction.
    """
    df = labelled.sort_values(["driver", "lap_number"]).copy()
    df = _add_stint_features(df)
    df = _add_gap_features(df)
    df = _add_pace_features(df)
    df = _add_safety_car_features(df)

    # `team` (constructor) is carried for the compound head; real ingest always
    # provides it, but default it so partial inputs (e.g. tests) still build.
    if "team" not in df.columns:
        df["team"] = "UNKNOWN"

    # lap_number is both an identifier and a feature; dedupe while keeping order.
    keep = list(dict.fromkeys(_CARRY + FEATURE_COLUMNS))
    out = cast(pd.DataFrame, df[keep]).reset_index(drop=True)

    # Normalise feature dtypes for downstream modelling.
    out["tyre_age"] = out["tyre_age"].astype("Int64")
    out["stint_position"] = out["stint_position"].astype("Int64")
    out["pit_count_so_far"] = out["pit_count_so_far"].astype("Int64")
    for col in BOOLEAN_FEATURES:
        out[col] = out[col].astype("boolean")
    return out
