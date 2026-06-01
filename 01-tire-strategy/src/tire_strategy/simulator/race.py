"""Free-air race simulator: the explicit pit-loss vs degradation trade-off.

Two strategies differ only in (a) how much time their tyres lose to degradation and
(b) how much they lose in the pits (fuel cancels — same for every strategy). This
module puts both on the same clock:

    strategy time-cost = cumulative tyre degradation + pit-loss per stop

and finds the time-optimal number of stops (and stint lengths) for a circuit and
compound. More stops = more pit-loss but shorter, fresher stints (less degradation);
fewer stops = less pit-loss but more degradation. The optimum is the crossover.

Honest scope: this is degradation-driven. Compound *offsets* (a fresh soft being
faster than a fresh hard) are not modelled — they are not reliably estimable from
public data (see degradation.py) — so this optimises stop *timing/number* for a
given compound, not the compound choice itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from tire_strategy.simulator.degradation import DegradationModel
from tire_strategy.simulator.pitloss import PitLossModel


@dataclass
class Stint:
    compound: str
    laps: int


def stint_degradation_cost(deg: DegradationModel, circuit: str, compound: str, laps: int) -> float:
    """Time (s) a stint loses to tyre degradation = sum of per-lap penalties."""
    return float(sum(deg.tyre_penalty(circuit, compound, age) for age in range(1, laps + 1)))


def strategy_cost(
    deg: DegradationModel, pit: PitLossModel, circuit: str, stints: list[Stint]
) -> dict[str, float]:
    """Total time-cost of a strategy, split into tyre and pit components (s)."""
    tyre = sum(stint_degradation_cost(deg, circuit, s.compound, s.laps) for s in stints)
    pit_cost = (len(stints) - 1) * pit.loss(circuit)
    return {"total": tyre + pit_cost, "tyre_cost": tyre, "pit_cost": pit_cost}


def _equal_stints(total_laps: int, n_stints: int) -> list[int]:
    """Split a race into as-equal-as-possible stint lengths (optimal for convex deg)."""
    base, remainder = divmod(total_laps, n_stints)
    return [base + (1 if i < remainder else 0) for i in range(n_stints)]


def compare_stop_counts(
    deg: DegradationModel,
    pit: PitLossModel,
    circuit: str,
    compound: str,
    total_laps: int,
    max_stops: int = 3,
) -> list[dict]:
    """Time-cost of 0..max_stops strategies for one compound — the trade-off table."""
    rows: list[dict] = []
    for n_stops in range(max_stops + 1):
        lengths = _equal_stints(total_laps, n_stops + 1)
        cost = strategy_cost(deg, pit, circuit, [Stint(compound, n) for n in lengths])
        rows.append(
            {
                "n_stops": n_stops,
                "stint_laps": lengths,
                "tyre_cost": round(cost["tyre_cost"], 1),
                "pit_cost": round(cost["pit_cost"], 1),
                "total_cost": round(cost["total"], 1),
            }
        )
    return rows


def optimal_strategy(
    deg: DegradationModel,
    pit: PitLossModel,
    circuit: str,
    compound: str,
    total_laps: int,
    max_stops: int = 3,
) -> dict:
    """The minimum-time-cost stop count + stint plan for a circuit and compound."""
    options = compare_stop_counts(deg, pit, circuit, compound, total_laps, max_stops)
    return min(options, key=lambda r: r["total_cost"])
