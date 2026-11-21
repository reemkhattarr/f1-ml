"""Leakage-safe dataset splitting.

The atomic unit is the **race weekend** (``year`` + ``round``): every lap from a
race goes entirely into one split. Splitting by lap or by driver-within-race
would let the model see part of a race's degradation curve in training and the
rest at test time — a leak (see ``docs/DESIGN.md`` §7).

The primary split is **temporal** — train on past seasons, validate on the early
part of the most recent season, test on its later part — because that mirrors how
the model would actually be deployed: predicting future races from past ones.
"""

from __future__ import annotations

from collections import Counter
from typing import cast

import pandas as pd

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def race_group_key(df: pd.DataFrame) -> pd.Series:
    """Return a per-row race-weekend key (``"<year>_<round>"``) for grouping/CV."""
    return df["year"].astype(str) + "_" + df["round"].astype(str)


def assign_temporal_split(
    df: pd.DataFrame,
    train_seasons: tuple[int, ...] = (2022, 2023, 2024),
    holdout_season: int = 2025,
    val_round_max: int = 12,
) -> pd.DataFrame:
    """Add a ``split`` column with values from :data:`SPLIT_NAMES`.

    - ``train``: any lap from a ``train_seasons`` race.
    - ``val``: ``holdout_season`` races up to and including ``val_round_max``.
    - ``test``: ``holdout_season`` races after ``val_round_max``.

    Rows from other seasons are left as ``<NA>`` so callers can spot anything that
    falls outside the intended design rather than silently training on it.
    """
    out = df.copy()
    split = pd.Series(pd.NA, index=out.index, dtype="string")

    is_train = out["year"].isin(list(train_seasons))
    is_holdout = out["year"] == holdout_season
    is_val = is_holdout & (out["round"] <= val_round_max)
    is_test = is_holdout & (out["round"] > val_round_max)

    split[is_train] = "train"
    split[is_val] = "val"
    split[is_test] = "test"
    out["split"] = split
    return out


# The regulation era the model is trained on starts here (ground-effect cars).
TRAIN_START_SEASON: int = 2022
# A season needs at least this many races before it can be the holdout season;
# until then the previous season stays the holdout (avoids a tiny, noisy test set).
MIN_HOLDOUT_RACES: int = 6


def default_split_seasons(df: pd.DataFrame) -> tuple[tuple[int, ...], int, int]:
    """Pick the split from the data instead of hard-coding seasons.

    Returns ``(train_seasons, holdout_season, val_round_max)``:

    - the **holdout** is the latest season with at least :data:`MIN_HOLDOUT_RACES`
      races (so a new season such as 2026 becomes the test bed for the new era as
      soon as enough of it has been raced);
    - **train** is every season from :data:`TRAIN_START_SEASON` up to the holdout;
    - **val/test** split the holdout races in half by round (early half calibrates
      the model and picks the threshold, late half is the untouched test set).

    Raises ``ValueError`` when the data cannot support a split (fewer than two
    seasons), rather than training on an empty set.
    """
    pairs = cast(pd.DataFrame, df[["year", "round"]]).dropna().drop_duplicates()
    races = sorted(
        (int(y), int(r)) for y, r in pairs.itertuples(index=False) if int(y) >= TRAIN_START_SEASON
    )
    counts = Counter(year for year, _ in races)
    eligible = sorted(y for y, n in counts.items() if n >= MIN_HOLDOUT_RACES)
    if not eligible:
        raise ValueError("No season has enough races to hold out; build the dataset first.")
    holdout = eligible[-1]
    train = tuple(sorted(y for y in counts if y < holdout))
    if not train:
        raise ValueError(f"No training seasons before the holdout season {holdout}.")
    rounds = [r for y, r in races if y == holdout]
    val_round_max = int(rounds[len(rounds) // 2 - 1])
    return train, holdout, val_round_max
