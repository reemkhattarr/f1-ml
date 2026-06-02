"""Per-lap telemetry features (tyre-load / driving-aggression).

The lap-level dataset captures *that* tyres degrade; telemetry captures *how hard
they are being worked* — braking intensity, time at full throttle, speed, and
throttle modulation (aggression). These are aggregated per lap from the ~10 Hz
car-data trace.

This is a heavy data pull (telemetry is ~1000x the volume of lap times), so it is
built into its own parquet and merged into the main dataset only if it proves out.
Whether it helps the pit model is an open question — much of the per-lap signal is
circuit-driven (already captured by `circuit`); the new part is driver/ lap-to-lap
aggression. Measured, not assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import fastf1
import pandas as pd

from tire_strategy.config import DATA_DIR, enable_cache
from tire_strategy.data.dataset import iter_calendar

logger = logging.getLogger(__name__)

TELEMETRY_PATH: Path = DATA_DIR / "processed" / "telemetry_features.parquet"


def _lap_features(lap) -> dict | None:
    """Aggregate one lap's car-data trace into tyre-load / aggression features."""
    try:
        tel = lap.get_car_data()
    except Exception:
        return None
    if tel.empty or len(tel) < 10:
        return None

    speed = tel["Speed"]
    throttle = tel["Throttle"]
    brake = tel["Brake"]
    brake_pct = float(brake.mean()) if brake.dtype == bool else float((brake > 0).mean())

    return {
        "full_throttle_pct": float((throttle >= 99).mean()),
        "brake_pct": brake_pct,
        "mean_speed": float(speed.mean()),
        "long_load": float(speed.diff().abs().mean()),  # accel/braking intensity proxy
        "throttle_modulation": float(throttle.diff().abs().mean()),  # on/off aggression
    }


def race_telemetry(year: int, race: int | str) -> pd.DataFrame:
    """Per-lap telemetry features for one race, keyed for merging into the dataset."""
    enable_cache()
    session = fastf1.get_session(year, race, "R")
    session.load(telemetry=True, weather=False, messages=False)
    round_number = int(session.event["RoundNumber"])

    rows: list[dict] = []
    for _, lap in session.laps.iterlaps():
        feats = _lap_features(lap)
        if feats is None or bool(pd.isna(lap["LapNumber"])):
            continue
        feats.update(
            {
                "year": int(year),
                "round": round_number,
                "driver": str(lap["Driver"]),
                "lap_number": int(lap["LapNumber"]),
            }
        )
        rows.append(feats)
    return pd.DataFrame(rows)


def build_telemetry_dataset(
    seasons: tuple[int, ...] = (2024,), out_path: Path | None = TELEMETRY_PATH
) -> pd.DataFrame:
    """Assemble per-lap telemetry features across races (heavy — downloads telemetry)."""
    frames: list[pd.DataFrame] = []
    for year, rnd in iter_calendar(seasons):
        try:
            frames.append(race_telemetry(year, rnd))
            logger.info("Telemetry: %s round %s done", year, rnd)
        except Exception as exc:
            logger.warning("Telemetry skip %s round %s: %s", year, rnd, exc)

    if not frames:
        raise RuntimeError("No telemetry could be built.")
    data = pd.concat(frames, ignore_index=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out_path, index=False)
    return data


TELEMETRY_FEATURES = [
    "full_throttle_pct",
    "brake_pct",
    "mean_speed",
    "long_load",
    "throttle_modulation",
]


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    seasons = tuple(int(a) for a in sys.argv[1:]) or (2024,)
    df = build_telemetry_dataset(seasons)
    print(f"Telemetry features: {len(df):,} laps across seasons {seasons} -> {TELEMETRY_PATH}")
