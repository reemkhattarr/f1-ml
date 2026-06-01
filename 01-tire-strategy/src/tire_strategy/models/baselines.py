"""Baselines for the pit-decision model.

Two reference points the learned model must beat (DESIGN.md §2):

- **Majority class** — always "stay out". Scores ~98% accuracy and ~0 useful
  recall; its job is to show why accuracy is the wrong metric.
- **Rule-based strategy** — a transparent domain heuristic: pit when the tyre is
  past a compound-specific age, or pit under a safety car once the tyre is
  reasonably worn (the classic "cheap stop" opportunity). This is the bar a real
  strategy engineer would recognise, so beating it is a meaningful result.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

# Rough compound-specific age (laps) past which a stop becomes likely. Deliberate
# domain heuristics, not fitted values — this is a baseline.
DEFAULT_AGE_THRESHOLDS: dict[str, int] = {"SOFT": 16, "MEDIUM": 26, "HARD": 36}
# Under a safety car, a stop is "cheap"; take it once the tyre is this old.
DEFAULT_SC_MIN_AGE: int = 8


class MajorityBaseline:
    """Always predict 'stay out' (class 0)."""

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int_]:
        return np.zeros(len(X), dtype=int)


class MostFrequentCompound:
    """Compound-head baseline: always predict the most common training compound.

    The bar the learned compound model must clear — if it can't beat "always guess
    HARD", it has learned nothing useful.
    """

    def __init__(self) -> None:
        self.compound_: str = "HARD"

    def fit(self, y: pd.Series) -> MostFrequentCompound:
        self.compound_ = str(y.astype("string").mode().iloc[0])
        return self

    def predict(self, X: pd.DataFrame) -> npt.NDArray:
        return np.array([self.compound_] * len(X), dtype=object)


class RuleBasedBaseline:
    """Pit when the tyre is past its compound age, or worn under a safety car."""

    def __init__(
        self,
        age_thresholds: dict[str, int] | None = None,
        sc_min_age: int = DEFAULT_SC_MIN_AGE,
    ) -> None:
        self.age_thresholds = age_thresholds or DEFAULT_AGE_THRESHOLDS
        self.sc_min_age = sc_min_age

    def predict(self, X: pd.DataFrame) -> npt.NDArray[np.int_]:
        compound = cast(pd.Series, X["compound"]).astype("object")
        age = np.asarray(
            pd.to_numeric(cast(pd.Series, X["tyre_age"]), errors="coerce"), dtype=float
        )
        caution_obj = cast(pd.Series, X["under_caution"]).astype("object")
        caution = np.nan_to_num(
            np.asarray(pd.to_numeric(caution_obj, errors="coerce"), dtype=float), nan=0.0
        )

        # Compound-specific age threshold; unknown (e.g. wet) compounds -> NaN, so
        # the age rule never fires for them (only the safety-car rule can).
        thresholds = np.array([self.age_thresholds.get(c, np.nan) for c in compound], dtype=float)
        past_age = age >= thresholds  # NaN comparisons are False
        cheap_stop = (caution > 0) & (age >= self.sc_min_age)
        return (past_age | cheap_stop).astype(int)
