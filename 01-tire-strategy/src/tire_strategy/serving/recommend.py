"""Compound recommendation with a deterministic wet-weather gate.

The compound head is trained on slick compounds only (SOFT / MEDIUM / HARD); see
``docs/IMPLEMENTATION.md`` §9.2. That makes its prediction meaningless in the
wet, where the tyre choice is not a strategic slick trade-off but a rule: if it
is raining, fit a wet-weather tyre. So at serving time we **gate** on rainfall
rather than always calling the slick head.

The gate is deterministic, not a second ML model, on measured grounds (§12 #20):
ML with ``rainfall`` as a feature was tried two ways and neither reliably beat the
rule. We also tested refining the rule with **track temperature** (predict slicks
on a warm-but-raining pit). It looked good on the pooled 2018-2025 data (0.59 ->
0.71) but that win came entirely from the older tyre era — on the **2022-2025
regulation era we actually train/serve on, it was net-negative** (0.66 vs 0.74 for
the flat rule, because modern teams fit wets at ~74% of rainy pits). So the gate is
the flat rule: **raining -> INTERMEDIATE**. Full WET is never predicted (only ~13
stops, and inter-vs-full-wet needs rain intensity we lack); WET is collapsed into
INTERMEDIATE everywhere.

The gate only chooses the *compound* given a pit; **whether/when to pit is the
pit-timing head's job** (rain-aware, and trained on a 3-lap window so the lap or
two teams wait after rain starts is absorbed) — so the gate never forces a
premature wet stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from tire_strategy.data.features import DEFAULT_WET_COMPOUND

CompoundSource = Literal["model", "wet-rule"]


@dataclass(frozen=True)
class CompoundRecommendation:
    """A compound choice plus how it was reached.

    ``source`` is ``"model"`` when the slick compound head decided, or
    ``"wet-rule"`` when the deterministic rain gate did.
    """

    compound: str
    probabilities: dict[str, float]
    source: CompoundSource


def recommend_compound(
    compound_model: Any, features: pd.DataFrame, *, rainfall: bool
) -> CompoundRecommendation:
    """Recommend a compound for one race state, gating on rainfall.

    In the rain, return the rule-based wet tyre (``DEFAULT_WET_COMPOUND``) at
    probability 1.0; otherwise defer to the slick compound head. ``features`` must
    be a single-row frame of the model's feature columns.
    """
    if rainfall:
        return CompoundRecommendation(
            compound=DEFAULT_WET_COMPOUND,
            probabilities={DEFAULT_WET_COMPOUND: 1.0},
            source="wet-rule",
        )

    proba = compound_model.predict_proba(features)[0]
    classes = [str(c) for c in compound_model.classes_]
    probabilities = {cls: float(p) for cls, p in zip(classes, proba, strict=True)}
    best = max(probabilities, key=lambda c: probabilities[c])
    return CompoundRecommendation(compound=best, probabilities=probabilities, source="model")
