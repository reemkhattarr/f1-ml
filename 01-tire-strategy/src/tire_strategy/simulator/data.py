"""Clean stint-lap dataset for the simulator.

The main modelling dataset deliberately drops raw current-lap times (leakage), so
the degradation model is fitted on a separate table of **clean slick laps with
their lap times**, assembled from the cached raw FastF1 data. "Clean" = green-flag,
accurate, non-deleted, not an in/out lap — i.e. laps whose pace reflects the tyre.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

import pandas as pd

from f1_foundation import load_race
from tire_strategy.config import DATA_DIR, enable_cache
from tire_strategy.data.dataset import iter_calendar
from tire_strategy.data.features import SLICK_COMPOUNDS

logger = logging.getLogger(__name__)

STINT_PATH: Path = DATA_DIR / "processed" / "stint_laps.parquet"
_COLUMNS = [
    "year",
    "round",
    "circuit",
    "driver",
    "stint",
    "compound",
    "tyre_age",
    "lap_time_s",
    "laps_remaining",
    "track_temp",
    "air_temp",
]


def _bool(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name]).fillna(False).astype(bool)


def clean_slick_mask(race: pd.DataFrame) -> pd.Series:
    """True for green-flag, timed, accurate slick laps that are not pit laps."""
    return (
        ~_bool(race, "is_in_lap")
        & ~_bool(race, "is_out_lap")
        & ~_bool(race, "under_caution")
        & _bool(race, "is_accurate")
        & ~_bool(race, "deleted")
        & cast(pd.Series, race["lap_time_s"]).notna()
        & cast(pd.Series, race["compound"]).isin(SLICK_COMPOUNDS)
    )


def clean_slick_laps(race: pd.DataFrame) -> pd.DataFrame:
    """Clean slick laps from one ingested race, with the columns the model needs."""
    mask = clean_slick_mask(race)
    out = cast(pd.DataFrame, race.loc[mask]).rename(columns={"tyre_life": "tyre_age"})
    return cast(pd.DataFrame, out[_COLUMNS]).reset_index(drop=True)


def build_stint_dataset(
    seasons: tuple[int, ...] = (2022, 2023, 2024), out_path: Path | None = STINT_PATH
) -> pd.DataFrame:
    """Assemble clean slick laps across all races into one parquet."""
    enable_cache()
    frames: list[pd.DataFrame] = []
    for year, rnd in iter_calendar(seasons):
        try:
            frames.append(clean_slick_laps(load_race(year, rnd)))
        except Exception as exc:
            logger.warning("Skipping %s round %s: %s", year, rnd, exc)

    if not frames:
        raise RuntimeError("No stint laps could be built.")
    data = pd.concat(frames, ignore_index=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
    return data


def load_stint_laps(path: Path = STINT_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = build_stint_dataset()
    print(f"Stint laps: {len(df):,} clean slick laps -> {STINT_PATH}")
