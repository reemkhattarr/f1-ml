"""Build rich "browse everything" datasets for the data explorer.

From the already-cached race sessions, extract the full FastF1 data we otherwise
drop: complete per-lap timing (lap + sector times, speed-trap speeds, positions,
pit timestamps), full weather, and session results. Timedeltas are converted to
seconds and leftover objects to strings so everything is parquet/display friendly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from tire_strategy.config import DATA_DIR, enable_cache
from tire_strategy.data.dataset import load_dataset

logger = logging.getLogger(__name__)

RAW_LAPS_PATH: Path = DATA_DIR / "processed" / "raw_laps.parquet"
WEATHER_PATH: Path = DATA_DIR / "processed" / "weather.parquet"
RESULTS_PATH: Path = DATA_DIR / "processed" / "results.parquet"


def _display_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Timedeltas -> seconds; datetimes/objects -> strings; round floats."""
    out = df.copy()
    for col in out.columns:
        kind = str(out[col].dtype)
        if kind.startswith("timedelta"):
            out[col] = out[col].dt.total_seconds().round(3)
        elif kind.startswith("datetime") or kind == "object":
            out[col] = out[col].astype(str)
        elif kind.startswith("float"):
            out[col] = out[col].round(3)
    return out


def build_explorer_data(
    raw_path: Path = RAW_LAPS_PATH,
    weather_path: Path = WEATHER_PATH,
    results_path: Path = RESULTS_PATH,
) -> None:
    """Assemble raw laps / weather / results parquets from all cached races."""
    enable_cache()
    races = load_dataset()[["year", "round"]].drop_duplicates()
    laps_all: list[pd.DataFrame] = []
    weather_all: list[pd.DataFrame] = []
    results_all: list[pd.DataFrame] = []

    for year, rnd in races.itertuples(index=False):
        try:
            session = fastf1.get_session(int(year), int(rnd), "R")
            session.load(laps=True, weather=True, telemetry=False, messages=False)
            meta = {"year": int(year), "round": int(rnd), "circuit": str(session.event["Location"])}
            laps_all.append(_display_safe(session.laps).assign(**meta))
            weather_all.append(_display_safe(session.weather_data).assign(**meta))
            results_all.append(_display_safe(session.results).assign(**meta))
            logger.info("done %s round %s", year, rnd)
        except Exception as exc:  # noqa: BLE001 - one bad race must not abort the build
            logger.warning("skip %s round %s: %s", year, rnd, exc)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    for frames, path in ((laps_all, raw_path), (weather_all, weather_path), (results_all, results_path)):
        combined = pd.concat(frames, ignore_index=True)
        for col in combined.columns:  # coerce leftover objects so pyarrow never chokes
            if combined[col].dtype == "object":
                combined[col] = combined[col].astype(str)
        combined.to_parquet(path, index=False)
        logger.info("Wrote %s (%d rows)", path.name, len(combined))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_explorer_data()
    print(f"Built: {RAW_LAPS_PATH.name}, {WEATHER_PATH.name}, {RESULTS_PATH.name}")
