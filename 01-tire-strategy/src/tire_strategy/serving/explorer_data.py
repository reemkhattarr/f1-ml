"""Build rich "browse everything" datasets for the data explorer.

Walks the FastF1 event schedule for every season and extracts the full data for
**every session** of each weekend — free practice, qualifying, sprint and race —
not just the race. For each session we keep complete per-lap timing (lap + sector
times, speed-trap speeds, positions, pit timestamps, deleted-lap flags), full
weather, and the session results (so e.g. Qualifying rows carry Q1/Q2/Q3).

Each row is tagged with `year`, `round`, `circuit` and `session` so the explorer
can filter by any of them. Timedeltas become seconds and leftover objects become
strings so everything is parquet/display friendly.

The build is **resumable**: FastF1's on-disk cache persists every download, so an
interrupted run continues cheaply (cached sessions re-read instantly). Each session
is loaded defensively — a missing/unavailable session is skipped, never fatal.

    uv run python -m tire_strategy.serving.explorer_data            # all seasons
    uv run python -m tire_strategy.serving.explorer_data 2025 2026  # only these
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

from tire_strategy.config import DATA_DIR, enable_cache

logger = logging.getLogger(__name__)

RAW_LAPS_PATH: Path = DATA_DIR / "processed" / "raw_laps.parquet"
WEATHER_PATH: Path = DATA_DIR / "processed" / "weather.parquet"
RESULTS_PATH: Path = DATA_DIR / "processed" / "results.parquet"

# Seasons FastF1 covers well. 2026 is partial (only completed rounds load).
DEFAULT_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026)

# Map FastF1's full session names to short, stable codes for the `session` column.
_SESSION_CODE: dict[str, str] = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Sprint": "Sprint",
    "Race": "R",
}


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


def _event_sessions(event: pd.Series) -> list[str]:
    """The session names actually scheduled for one event (Session1..Session5)."""
    names = []
    for i in range(1, 6):
        val = event.get(f"Session{i}")
        if isinstance(val, str) and val and val.lower() != "none":
            names.append(val)
    return names


def _extract_session(year: int, rnd: int, session_name: str) -> dict[str, pd.DataFrame] | None:
    """Load one session and return display-safe laps/weather/results, or None."""
    session = fastf1.get_session(year, rnd, session_name)
    session.load(laps=True, weather=True, telemetry=False, messages=True)
    if session.laps is None or len(session.laps) == 0:
        return None
    meta = {
        "year": year,
        "round": rnd,
        "circuit": str(session.event["Location"]),
        "session": _SESSION_CODE.get(session_name, session_name),
    }
    out: dict[str, pd.DataFrame] = {"laps": _display_safe(session.laps).assign(**meta)}
    if session.weather_data is not None and len(session.weather_data):
        out["weather"] = _display_safe(session.weather_data).assign(**meta)
    if session.results is not None and len(session.results):
        out["results"] = _display_safe(session.results).assign(**meta)
    return out


def _load_existing(path: Path) -> pd.DataFrame | None:
    """Prior output for resume. Pre-`session` parquets are treated as race-only."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "session" not in df.columns:
        df = df.assign(session="R")  # legacy race-only build
    return df


def _done_keys(*frames: pd.DataFrame | None) -> set[tuple[int, int, str]]:
    """The (year, round, session) triples already present, so we never refetch."""
    done: set[tuple[int, int, str]] = set()
    for df in frames:
        if df is None or not {"year", "round", "session"} <= set(df.columns):
            continue
        for y, r, s in df[["year", "round", "session"]].drop_duplicates().itertuples(index=False):
            done.add((int(y), int(r), str(s)))
    return done


def _write(existing: pd.DataFrame | None, new: list[pd.DataFrame], path: Path) -> None:
    frames = [f for f in [existing, *new] if f is not None and len(f)]
    if not frames:
        return
    combined = pd.concat(frames, ignore_index=True)
    for col in combined.columns:  # coerce leftover objects so pyarrow never chokes
        if combined[col].dtype == "object":
            combined[col] = combined[col].astype(str)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path.name, len(combined))


def build_explorer_data(
    seasons: tuple[int, ...] = DEFAULT_SEASONS,
    raw_path: Path = RAW_LAPS_PATH,
    weather_path: Path = WEATHER_PATH,
    results_path: Path = RESULTS_PATH,
) -> None:
    """Assemble laps / weather / results parquets across all sessions and seasons.

    **Resumable and rate-limit-aware.** FastF1's API allows ~500 calls/hour, far
    fewer than a cold all-sessions download needs, so this skips sessions already
    in the output and — when the rate limit is hit — saves everything gathered so
    far and stops cleanly. Re-run later (the cache holds prior downloads) to
    continue; progress accumulates until coverage is complete.
    """
    enable_cache()
    existing_laps = _load_existing(raw_path)
    existing_weather = _load_existing(weather_path)
    existing_results = _load_existing(results_path)
    done = _done_keys(existing_laps, existing_weather, existing_results)
    logger.info("Resuming: %d session(s) already built", len(done))

    laps_all: list[pd.DataFrame] = []
    weather_all: list[pd.DataFrame] = []
    results_all: list[pd.DataFrame] = []
    n_new = 0
    rate_limited = False

    for year in seasons:
        if rate_limited:
            break
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:  # pragma: no cover - network/availability
            logger.warning("skip season %s schedule: %s", year, exc)
            continue

        for event in schedule.to_dict("records"):
            if rate_limited:
                break
            rnd = int(event.get("RoundNumber", 0) or 0)
            if rnd <= 0:
                continue
            for session_name in _event_sessions(pd.Series(event)):
                code = _SESSION_CODE.get(session_name, session_name)
                if (year, rnd, code) in done:
                    continue
                try:
                    extracted = _extract_session(year, rnd, session_name)
                except RateLimitExceededError:
                    # Hard cap reached — stop and persist progress; resume later.
                    logger.warning(
                        "FastF1 rate limit reached after %d new session(s); "
                        "saving progress. Re-run to continue.",
                        n_new,
                    )
                    rate_limited = True
                    break
                except Exception as exc:
                    # Future/cancelled/unavailable sessions are expected; debug only.
                    logger.debug("skip %s R%s %s: %s", year, rnd, session_name, exc)
                    continue
                if extracted is None:
                    continue
                laps_all.append(extracted["laps"])
                if "weather" in extracted:
                    weather_all.append(extracted["weather"])
                if "results" in extracted:
                    results_all.append(extracted["results"])
                n_new += 1
                logger.info("done %s R%s %s", year, rnd, session_name)

    logger.info("Added %d new session(s) this run", n_new)
    _write(existing_laps, laps_all, raw_path)
    _write(existing_weather, weather_all, weather_path)
    _write(existing_results, results_all, results_path)


def _parse_seasons(argv: list[str]) -> tuple[int, ...]:
    years = tuple(int(a) for a in argv if a.isdigit())
    return years or DEFAULT_SEASONS


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_explorer_data(seasons=_parse_seasons(sys.argv[1:]))
    print(f"Built: {RAW_LAPS_PATH.name}, {WEATHER_PATH.name}, {RESULTS_PATH.name}")
