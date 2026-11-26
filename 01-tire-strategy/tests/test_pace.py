"""The fixed-effects pace fit recovers wear and keeps the physical limits."""

from __future__ import annotations

import numpy as np
import pytest

from tire_strategy.strategy.pace import MIN_STEP, MIN_WEAR, fit_pace
from tire_strategy.strategy.params import clean_laps

from .strategy_fixtures import TRUE_DEG, synthetic_race_laps


@pytest.fixture(scope="module")
def laps():
    return clean_laps(synthetic_race_laps())


def test_wear_recovered(laps) -> None:
    fit = fit_pace(laps)
    for c, true in TRUE_DEG.items():
        assert fit.wear[c] == pytest.approx(true, abs=0.015)
    assert fit.drivers == 54
    assert fit.multi_compound > 0


def test_strong_prior_wins_over_few_laps(laps) -> None:
    prior = np.array([-0.05, -0.4, 0.3, 0.2, 0.15, 0.1])
    fit = fit_pace(laps.head(200), prior=prior, prior_laps=1e9)
    assert fit.wear["SOFT"] == pytest.approx(0.2, abs=1e-3)
    assert fit.offsets["HARD"] == pytest.approx(0.3, abs=1e-3)


def test_too_few_laps_returns_the_prior(laps) -> None:
    prior = np.array([-0.05, -0.4, 0.3, 0.09, 0.06, 0.045])
    fit = fit_pace(laps.head(10), prior=prior)
    assert fit.wear["MEDIUM"] == pytest.approx(0.06)


def test_limits_on_reversed_data(laps) -> None:
    bad = laps.copy()
    soft = bad["compound"] == "SOFT"
    # Make softs look slower and wear-free.
    bad.loc[soft, "lap_time_s"] += 1.0 - 0.1 * bad.loc[soft, "tyre_age"].astype(float)
    fit = fit_pace(bad)
    assert fit.offsets["SOFT"] <= -MIN_STEP
    assert min(fit.wear.values()) >= MIN_WEAR
    assert fit.wear["SOFT"] >= fit.wear["MEDIUM"] >= fit.wear["HARD"]
    assert fit.clamped
