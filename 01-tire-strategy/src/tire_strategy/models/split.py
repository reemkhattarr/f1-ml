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

import pandas as pd

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")


def race_group_key(df: pd.DataFrame) -> pd.Series:
    """Return a per-row race-weekend key (``"<year>_<round>"``) for grouping/CV."""
    return df["year"].astype(str) + "_" + df["round"].astype(str)


def assign_temporal_split(
    df: pd.DataFrame,
    train_seasons: tuple[int, ...] = (2022, 2023),
    holdout_season: int = 2024,
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
