"""Assemble the full multi-season modelling dataset.

Iterates the FastF1 race calendar for the configured seasons, builds the
labelled, leak-free feature table for each race (``labels`` + ``features``), and
concatenates them into one parquet file. Individual races that fail to load are
logged and skipped rather than aborting the whole build — silent gaps would
misrepresent coverage, so skipped races are reported explicitly.

Run as a module::

    uv run python -m tire_strategy.data.dataset            # 2022-2024
    uv run python -m tire_strategy.data.dataset 2024       # one season
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import fastf1
import pandas as pd

from f1_foundation import retired_drivers
from f1_foundation.circuits import canonical_circuit
from f1_foundation.openf1 import (
    OpenF1Client,
    completed_races,
    load_race_openf1,
    load_results_openf1,
)
from tire_strategy.config import DATA_DIR, enable_cache
from tire_strategy.data.features import build_features
from tire_strategy.data.labels import build_labels, load_labelled_race

logger = logging.getLogger(__name__)

DEFAULT_SEASONS: tuple[int, ...] = (2022, 2023, 2024)
DATASET_PATH: Path = DATA_DIR / "processed" / "tire_strategy_dataset.parquet"
# Raw per-lap race tables (with lap times), the input of the strategy engine.
RACE_LAPS_PATH: Path = DATA_DIR / "processed" / "race_laps.parquet"
# OpenF1 coverage starts with the 2023 season.
OPENF1_FIRST_SEASON = 2023


def iter_calendar(seasons: tuple[int, ...]) -> list[tuple[int, int]]:
    """Return ``(season, round_number)`` pairs for every race in the seasons."""
    races: list[tuple[int, int]] = []
    for season in seasons:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
        for round_number in schedule["RoundNumber"]:
            if int(round_number) >= 1:
                races.append((season, int(round_number)))
    return races


def build_one(season: int, race: int | str) -> pd.DataFrame | None:
    """Build the feature table for a single race, or ``None`` if it fails."""
    try:
        labelled = load_labelled_race(season, race)
        return build_features(labelled)
    except Exception as exc:  # one bad race must not abort the whole build
        logger.warning("Skipping %s round %s: %s", season, race, exc)
        return None


def assemble(
    races: Sequence[tuple[int, int | str]], out_path: Path | None = DATASET_PATH
) -> pd.DataFrame:
    """Build and concatenate the given races, optionally writing parquet."""
    enable_cache()
    frames: list[pd.DataFrame] = []
    skipped = 0
    for season, race in races:
        df = build_one(season, race)
        if df is None:
            skipped += 1
            continue
        frames.append(df)
        logger.info("Built %s round %s (%d laps)", season, race, len(df))

    if not frames:
        raise RuntimeError("No races could be built; dataset is empty.")

    data = pd.concat(frames, ignore_index=True)
    logger.info(
        "Assembled %d laps from %d/%d races (%d skipped).",
        len(data),
        len(frames),
        len(races),
        skipped,
    )

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
        logger.info("Wrote dataset -> %s", out_path)
    return data


def build_dataset(
    seasons: tuple[int, ...] = DEFAULT_SEASONS, out_path: Path | None = DATASET_PATH
) -> pd.DataFrame:
    """Assemble the full dataset for the given seasons."""
    return assemble(iter_calendar(seasons), out_path=out_path)


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load a previously built dataset (circuit names normalised across sources)."""
    df = pd.read_parquet(path)
    df["circuit"] = df["circuit"].map(canonical_circuit)
    return df


def _keys(df: pd.DataFrame | None) -> set[tuple[int, int]]:
    if df is None or df.empty:
        return set()
    pairs = df[["year", "round"]].drop_duplicates().to_numpy()
    return {(int(y), int(r)) for y, r in pairs}


def update_from_openf1(
    seasons: tuple[int, ...],
    dataset_path: Path = DATASET_PATH,
    laps_path: Path = RACE_LAPS_PATH,
    client: OpenF1Client | None = None,
) -> tuple[int, int]:
    """Add every finished race that is missing, using OpenF1 (works from the cloud).

    Keeps two files current: the modelling dataset (labels + features, one row per
    lap) and the raw race laps the strategy engine fits on. Races already present
    are not fetched again. Returns ``(new_dataset_races, new_lap_races)``.
    """
    client = client or OpenF1Client()
    seasons = tuple(s for s in seasons if s >= OPENF1_FIRST_SEASON)
    dataset = pd.read_parquet(dataset_path) if dataset_path.exists() else None
    laps = pd.read_parquet(laps_path) if laps_path.exists() else None
    have_ds, have_laps = _keys(dataset), _keys(laps)

    new_ds: list[pd.DataFrame] = []
    new_laps: list[pd.DataFrame] = []
    for race in completed_races(seasons, client):
        key = (race.year, race.round)
        if key in have_ds and key in have_laps:
            continue
        try:
            raw = load_race_openf1(race.year, race, client)
            if key not in have_laps:
                new_laps.append(raw)
            if key not in have_ds:
                results = load_results_openf1(race.year, race, client)
                new_ds.append(build_features(build_labels(raw, retired=retired_drivers(results))))
            logger.info("OpenF1: %s round %s (%s)", race.year, race.round, race.circuit)
        except Exception as exc:  # one bad race must not stop the update
            logger.warning("Skipping %s round %s: %s", race.year, race.round, exc)

    for existing, new, path in ((dataset, new_ds, dataset_path), (laps, new_laps, laps_path)):
        if new:
            parts = ([existing] if existing is not None else []) + new
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    logger.info("Added %d races to the dataset, %d to the race laps.", len(new_ds), len(new_laps))
    return len(new_ds), len(new_laps)


def update_dataset(
    seasons: tuple[int, ...] = DEFAULT_SEASONS, out_path: Path = DATASET_PATH
) -> pd.DataFrame:
    """Incrementally add only the races not already in the dataset, then save.

    Serves both backfilling history and picking up newly-completed races with the
    same code: races already present are skipped (no API calls), missing ones are
    built and appended. Races that fail (e.g. rate-limited, or a future race not yet
    run) are simply left for a later run.
    """
    enable_cache()
    existing = pd.read_parquet(out_path) if out_path.exists() else None
    have: set[tuple[int, int]] = set()
    if existing is not None:
        have = {
            (int(y), int(r)) for y, r in existing[["year", "round"]].drop_duplicates().to_numpy()
        }

    new_frames: list[pd.DataFrame] = []
    for year, rnd in iter_calendar(seasons):
        if (year, rnd) in have:
            continue
        df = build_one(year, rnd)
        if df is not None:
            new_frames.append(df)
            logger.info("Added %s round %s (%d laps)", year, rnd, len(df))

    if not new_frames:
        logger.info("Dataset already up to date for the requested seasons.")
        return existing if existing is not None else pd.DataFrame()

    parts = ([existing] if existing is not None else []) + new_frames
    combined = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    n_races = combined[["year", "round"]].drop_duplicates().shape[0]
    logger.info("Dataset now covers %d races (%d laps).", n_races, len(combined))
    return combined


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the tire-strategy dataset.")
    parser.add_argument(
        "seasons",
        nargs="*",
        type=int,
        default=list(DEFAULT_SEASONS),
        help="Seasons to include (default: 2022 2023 2024).",
    )
    parser.add_argument("--out", type=Path, default=DATASET_PATH, help="Output parquet path.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incrementally add only missing races (backfill / new races).",
    )
    parser.add_argument(
        "--source",
        choices=("fastf1", "openf1"),
        default="fastf1",
        help="Where to fetch races: FastF1 (local machines) or OpenF1 (works in the cloud).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seasons = tuple(args.seasons)
    if args.source == "openf1":
        n_ds, n_laps = update_from_openf1(seasons, dataset_path=args.out)
        print(f"OpenF1 update: {n_ds} new races in the dataset, {n_laps} in the race laps.")
        return 0
    data = (
        update_dataset(seasons, out_path=args.out)
        if args.update
        else build_dataset(seasons, out_path=args.out)
    )
    print(f"Dataset: {len(data):,} laps, {data['driver'].nunique()} drivers, -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
