"""Synthetic races with known tyre behaviour, for testing the strategy engine."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_DEG = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.04}
TRUE_OFFSET = {"SOFT": -0.6, "MEDIUM": 0.0, "HARD": 0.4}
TRUE_PIT_LOSS = 22.0
EVOLUTION = -0.02
FUEL = 0.055
STRATEGIES = [
    [("MEDIUM", 22), ("HARD", 28)],
    [("SOFT", 15), ("MEDIUM", 35)],
    [("SOFT", 14), ("HARD", 36)],
    [("MEDIUM", 18), ("HARD", 18), ("MEDIUM", 14)],
    [("HARD", 30), ("SOFT", 20)],
    [("SOFT", 12), ("MEDIUM", 20), ("SOFT", 18)],
]


def synthetic_race_laps(
    races: tuple[tuple[int, int], ...] = ((2025, 3), (2026, 2), (2026, 7)),
    circuit: str = "Testville",
    drivers: int = 18,
    laps: int = 50,
    seed: int = 0,
    sc_laps: tuple[int, ...] = (),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year, rnd in races:
        for d in range(drivers):
            pace = rng.normal(0, 0.6)
            strategy = STRATEGIES[d % len(STRATEGIES)]
            lap = 1
            for stint_no, (compound, length) in enumerate(strategy, start=1):
                for age in range(1, length + 1):
                    if lap > laps:
                        break
                    t = (90 + pace + TRUE_OFFSET[compound] + TRUE_DEG[compound] * (age - 1)
                         + FUEL * (laps - lap) + EVOLUTION * lap + rng.normal(0, 0.12))  # fmt: skip
                    in_lap = age == length and lap < laps
                    out_lap = age == 1 and stint_no > 1
                    if in_lap:
                        t += TRUE_PIT_LOSS * 0.55
                    if out_lap:
                        t += TRUE_PIT_LOSS * 0.45
                    if lap == 1:
                        t += 6
                    caution = lap in sc_laps
                    rows.append(
                        {
                            "year": year, "round": rnd, "circuit": circuit,
                            "driver": f"D{d:02d}", "team": f"T{d // 2}",
                            "lap_number": lap, "stint": stint_no, "compound": compound,
                            "tyre_life": age, "laps_remaining": laps - lap,
                            "lap_time_s": t + (30 if caution else 0), "track_temp": 40.0,
                            "air_temp": 25.0, "is_in_lap": in_lap, "is_out_lap": out_lap,
                            "under_caution": caution, "flag_sc": caution, "flag_vsc": False,
                            "is_accurate": True, "deleted": False, "rainfall": False,
                        }
                    )  # fmt: skip
                    lap += 1
    df = pd.DataFrame(rows)
    df["time_s"] = df.groupby(["year", "round", "driver"])["lap_time_s"].cumsum()
    df["position"] = df.groupby(["year", "round", "lap_number"])["time_s"].rank(method="first")
    df["position"] = df["position"].astype("Int64")
    for col in ("is_in_lap", "is_out_lap", "under_caution", "flag_sc", "flag_vsc",
                "is_accurate", "deleted", "rainfall"):  # fmt: skip
        df[col] = df[col].astype("boolean")
    df["compound"] = df["compound"].astype("string")
    df["tyre_life"] = df["tyre_life"].astype("Int64")
    df["laps_remaining"] = df["laps_remaining"].astype("Int64")
    return df
