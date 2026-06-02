"""Per-circuit pit-loss estimate (seconds), from data.

Pit-loss is what a stop costs versus staying out. We estimate it per circuit as the
median, over real pit stops, of the in-lap + out-lap time *in excess of* the
driver's green pace that race: ``(in_lap - green) + (out_lap - green)``. That
captures the slow in-lap, the pit-lane traversal, and the cold out-lap — i.e. the
time the stop actually costs on track.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from f1_foundation import load_race
from tire_strategy.config import DATA_DIR, enable_cache
from tire_strategy.data.dataset import iter_calendar

PITLOSS_PATH: Path = DATA_DIR / "processed" / "pit_losses.json"
DEFAULT_PIT_LOSS = 22.0  # seconds — fallback for circuits without an estimate
_SANE = (8.0, 45.0)  # plausible pit-loss range (s) for filtering outliers


def _bool(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name]).fillna(False).astype(bool)


def race_pit_losses(race: pd.DataFrame) -> list[float]:
    """Estimated pit-loss (s) for every stop in one ingested race."""
    losses: list[float] = []
    for _, group in race.groupby("driver"):
        laps = cast(pd.DataFrame, group).sort_values("lap_number")
        clean = (
            ~_bool(laps, "is_in_lap")
            & ~_bool(laps, "is_out_lap")
            & ~_bool(laps, "under_caution")
            & _bool(laps, "is_accurate")
            & cast(pd.Series, laps["lap_time_s"]).notna()
        )
        green = laps.loc[clean, "lap_time_s"].median()
        if not np.isfinite(green):
            continue
        for _, in_lap in laps[_bool(laps, "is_in_lap")].iterrows():
            out_lap = laps[
                (laps["lap_number"] == in_lap["lap_number"] + 1) & _bool(laps, "is_out_lap")
            ]
            if out_lap.empty:
                continue
            in_t = in_lap["lap_time_s"]
            out_t = cast(pd.Series, out_lap["lap_time_s"]).iloc[0]
            if not (np.isfinite(in_t) and np.isfinite(out_t)):
                continue
            loss = float((in_t - green) + (out_t - green))
            if _SANE[0] < loss < _SANE[1]:
                losses.append(loss)
    return losses


@dataclass
class PitLossModel:
    """Per-circuit pit-loss (seconds), with a default fallback."""

    per_circuit: dict[str, float] = field(default_factory=dict)
    default: float = DEFAULT_PIT_LOSS

    def loss(self, circuit: str) -> float:
        return self.per_circuit.get(circuit, self.default)


def fit_pit_losses(
    seasons: tuple[int, ...] = (2022, 2023, 2024), min_stops: int = 5
) -> PitLossModel:
    """Estimate per-circuit pit-loss from cached race data."""
    enable_cache()
    by_circuit: dict[str, list[float]] = defaultdict(list)
    for year, rnd in iter_calendar(seasons):
        try:
            race = load_race(year, rnd)
        except Exception:
            continue
        circuit = str(race["circuit"].iloc[0])
        by_circuit[circuit].extend(race_pit_losses(race))

    per_circuit = {
        circuit: float(np.median(vals))
        for circuit, vals in by_circuit.items()
        if len(vals) >= min_stops
    }
    return PitLossModel(per_circuit=per_circuit)


def save_pit_losses(model: PitLossModel, path: Path = PITLOSS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"per_circuit": model.per_circuit, "default": model.default}, indent=2)
    )


def load_pit_losses(path: Path = PITLOSS_PATH) -> PitLossModel:
    data = json.loads(path.read_text())
    return PitLossModel(per_circuit=data["per_circuit"], default=data["default"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    model = fit_pit_losses()
    save_pit_losses(model)
    print(f"Pit-loss estimates for {len(model.per_circuit)} circuits -> {PITLOSS_PATH}")
