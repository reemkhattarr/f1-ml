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
