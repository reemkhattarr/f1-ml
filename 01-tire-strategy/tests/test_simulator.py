"""Tests for the race-strategy simulator's degradation model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tire_strategy.simulator.degradation import (
    DegradationModel,
    _stint_slope,
    fit_degradation,
)


def test_stint_slope_recovers_known_degradation() -> None:
    # Seven ages with a clean +0.10 s/lap trend in fuel-corrected time.
    stint = pd.DataFrame(
        {"tyre_age": range(2, 9), "_corrected": [90.0 + 0.10 * a for a in range(2, 9)]}
    )
    assert _stint_slope(stint) == pytest.approx(0.10, abs=1e-6)

    # Too few laps -> undefined.
    assert _stint_slope(stint.head(3)) is None


def test_model_rate_lookup_and_penalty() -> None:
    model = DegradationModel(
        rate_per_circuit={("Monza", "HARD"): 0.06},
        rate_per_compound={"HARD": 0.05},
    )
    assert model.rate("Monza", "HARD") == 0.06  # circuit-specific
    assert model.rate("Spa", "HARD") == 0.05  # falls back to per-compound
    assert np.isnan(model.rate("Spa", "SOFT"))  # unknown
    assert model.tyre_penalty("Monza", "HARD", 10) == pytest.approx(0.6)
    assert model.tyre_penalty("Monza", "HARD", 0) == 0.0


def test_strategy_cost_splits_tyre_and_pit() -> None:
    from tire_strategy.simulator.pitloss import PitLossModel
    from tire_strategy.simulator.race import Stint, stint_degradation_cost, strategy_cost

    deg = DegradationModel(rate_per_circuit={}, rate_per_compound={"HARD": 0.06})
    pit = PitLossModel(per_circuit={"Test": 25.0})
    # 10-lap stint degradation = 0.06 * (1+...+10) = 0.06*55 = 3.3
    assert stint_degradation_cost(deg, "Test", "HARD", 10) == pytest.approx(3.3)
    # A 1-stop = two stints + one pit-loss.
    cost = strategy_cost(deg, pit, "Test", [Stint("HARD", 25), Stint("HARD", 25)])
    assert cost["pit_cost"] == pytest.approx(25.0)
    assert cost["total"] == pytest.approx(cost["tyre_cost"] + cost["pit_cost"])


def test_optimiser_trades_pit_loss_against_degradation() -> None:
    from tire_strategy.simulator.pitloss import PitLossModel
    from tire_strategy.simulator.race import optimal_strategy

    deg = DegradationModel(rate_per_circuit={}, rate_per_compound={"HARD": 0.06})
    # Cheap pits + high degradation over a long race -> more stops are worth it.
    cheap = optimal_strategy(deg, PitLossModel(default=10.0), "X", "HARD", 60)
    dear = optimal_strategy(deg, PitLossModel(default=40.0), "X", "HARD", 60)
    assert cheap["n_stops"] > dear["n_stops"]  # cheaper pits -> stop more


def test_pit_loss_estimate_from_a_synthetic_race() -> None:
    from tire_strategy.simulator.pitloss import race_pit_losses

    # Driver runs ~90s green laps; the in-lap (lap 10) and out-lap (lap 11) are slow.
    rows = []
    for lap in range(1, 16):
        is_in = lap == 10
        is_out = lap == 11
        t = 90.0 + (12 if is_in else 0) + (11 if is_out else 0)
        rows.append(
            {
                "driver": "A",
                "lap_number": lap,
                "lap_time_s": t,
                "is_in_lap": is_in,
                "is_out_lap": is_out,
                "under_caution": False,
                "is_accurate": True,
            }
        )
    race = pd.DataFrame(rows)
    losses = race_pit_losses(race)
    assert len(losses) == 1
    assert losses[0] == pytest.approx(23.0)  # (102-90) + (101-90)


@pytest.mark.integration
def test_degradation_ordering_is_physical() -> None:
    from tire_strategy.simulator.data import load_stint_laps

    model = fit_degradation(load_stint_laps())
    rates = model.rate_per_compound
    # The core physics: softer compounds degrade faster.
    assert rates["SOFT"] > rates["MEDIUM"] > rates["HARD"]
    # Sensible magnitudes (seconds per lap).
    assert 0.0 < rates["HARD"] < 0.2
    assert 0.0 < rates["SOFT"] < 0.3


@pytest.mark.integration
def test_simulator_recovers_monza_one_stop() -> None:
    from tire_strategy.simulator.data import load_stint_laps
    from tire_strategy.simulator.pitloss import fit_pit_losses
    from tire_strategy.simulator.race import optimal_strategy

    deg = fit_degradation(load_stint_laps())
    pit = fit_pit_losses((2024,))
    best = optimal_strategy(deg, pit, "Monza", "HARD", total_laps=53)
    # Monza 2024 was a one-stop on hards (Leclerc's winning strategy).
    assert best["n_stops"] == 1
