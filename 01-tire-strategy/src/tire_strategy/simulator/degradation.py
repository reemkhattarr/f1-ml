"""Data-driven tyre degradation model (within-stint method).

**Why within-stint?** Pooling lap times across stints to compare compounds is
confounded: "soft pace" is dominated by whichever cars happened to run softs (often
backmarkers, late, in traffic), not by the compound. Measured directly, that pooled
approach recovers the soft-faster-than-hard ordering on only ~1/17 circuits.

Within a single stint the car is held constant, so the slope of fuel-corrected lap
time vs tyre age is **pure degradation**. Aggregating those within-stint slopes
recovers the correct physics: SOFT degrades fastest, HARD slowest (measured:
~0.091 / 0.061 / 0.054 s/lap). The slope is fitted through per-age medians, which
trims traffic-slowed laps.

**Honest limitation:** this models the degradation *rate*, which we can measure.
Absolute compound *offsets* (how much faster a fresh soft is than a fresh hard) are
small and not reliably estimable from public pooled data — see docs/SIMULATOR.md.
The simulator anchors offsets per-race from the driver's own observed pace instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from tire_strategy.data.features import FUEL_S_PER_LAP, SLICK_COMPOUNDS

_MIN_STINTS = 20  # min within-stint slopes to trust a (circuit, compound) rate
_MIN_AGES = 5  # min distinct tyre ages within a stint to fit its slope
_MIN_LAPS = 6  # min laps within a stint
_SANE = (-1.0, 2.0)  # plausible degradation-rate range (s/lap) for filtering


def fuel_corrected(df: pd.DataFrame) -> pd.Series:
    """Lap time with the fuel effect removed (isolates tyre + circuit pace)."""
    return df["lap_time_s"] - FUEL_S_PER_LAP * df["laps_remaining"]


def _stint_slope(stint: pd.DataFrame) -> float | None:
    """Robust degradation slope for one stint: fit through per-age medians."""
    usable = cast(pd.DataFrame, stint[stint["tyre_age"] >= 2])  # drop the out-lap warm-up
    if usable["tyre_age"].nunique() < _MIN_AGES or len(usable) < _MIN_LAPS:
        return None
    medians = usable.groupby("tyre_age")["_corrected"].median()
    if len(medians) < _MIN_AGES:
        return None
    slope = float(np.polyfit(medians.index.to_numpy(float), medians.to_numpy(float), 1)[0])
    return slope if _SANE[0] < slope < _SANE[1] else None


@dataclass
class DegradationModel:
    """Per-compound (and per-circuit) degradation rates, in seconds/lap."""

    rate_per_circuit: dict[tuple[str, str], float]
    rate_per_compound: dict[str, float]

    def rate(self, circuit: str, compound: str) -> float:
        """Degradation rate (s/lap) for a compound, circuit-specific if available."""
        circuit_rate = self.rate_per_circuit.get((circuit, compound))
        if circuit_rate is not None:
            return circuit_rate
        return self.rate_per_compound.get(compound, float("nan"))

    def tyre_penalty(self, circuit: str, compound: str, age: int) -> float:
        """Cumulative time lost to degradation by tyre `age` (s, vs a fresh tyre)."""
        return self.rate(circuit, compound) * max(age, 0)


def fit_degradation(stint_df: pd.DataFrame, min_stints: int = _MIN_STINTS) -> DegradationModel:
    """Fit degradation rates from within-stint slopes (car held constant)."""
    df = stint_df.copy()
    df["_corrected"] = fuel_corrected(df)
    df = df[df["compound"].isin(SLICK_COMPOUNDS) & np.isfinite(df["_corrected"])]

    records: list[dict] = []
    for _, group in df.groupby(["year", "round", "driver", "stint"], sort=False):
        stint = cast(pd.DataFrame, group)
        slope = _stint_slope(stint)
        if slope is not None:
            records.append(
                {
                    "circuit": stint["circuit"].iloc[0],
                    "compound": stint["compound"].iloc[0],
                    "slope": slope,
                }
            )
    slopes = pd.DataFrame(records)

    rate_per_compound = {str(c): float(g["slope"].median()) for c, g in slopes.groupby("compound")}
    rate_per_circuit: dict[tuple[str, str], float] = {}
    for key, group in slopes.groupby(["circuit", "compound"]):
        if len(group) >= min_stints:
            circuit, compound = cast(tuple[str, str], key)
            rate_per_circuit[(str(circuit), str(compound))] = float(group["slope"].median())
    return DegradationModel(rate_per_circuit=rate_per_circuit, rate_per_compound=rate_per_compound)
