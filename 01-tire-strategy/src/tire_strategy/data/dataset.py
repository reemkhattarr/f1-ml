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

from tire_strategy.config import DATA_DIR, enable_cache
from tire_strategy.data.features import build_features
from tire_strategy.data.labels import load_labelled_race

logger = logging.getLogger(__name__)

DEFAULT_SEASONS: tuple[int, ...] = (2022, 2023, 2024)
DATASET_PATH: Path = DATA_DIR / "processed" / "tire_strategy_dataset.parquet"


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
    """Load a previously built dataset."""
    return pd.read_parquet(path)


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
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seasons = tuple(args.seasons)
    data = (
        update_dataset(seasons, out_path=args.out)
        if args.update
        else build_dataset(seasons, out_path=args.out)
    )
    print(f"Dataset: {len(data):,} laps, {data['driver'].nunique()} drivers, -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
