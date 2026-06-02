"""Compound recommendation with a deterministic wet-weather gate.

The compound head is trained on slick compounds only (SOFT / MEDIUM / HARD); see
``docs/IMPLEMENTATION.md`` §9.2. That makes its prediction meaningless in the
wet, where the tyre choice is not a strategic slick trade-off but a rule: if it
is raining, fit a wet-weather tyre. So at serving time we **gate** on rainfall
rather than always calling the slick head.

Why the wet branch is a rule rather than a second ML model: ML with ``rainfall``
as a feature was tried two ways (a 4-class head; a binary wet-vs-slick model) and
neither reliably beat the rule — the true discriminator is rain intensity /
standing water, absent from our features, and with ~12 wet seasons the proxy
signal can't beat a majority-class rule with stability (docs §12 #20).

The rule is **not** flat "always INTERMEDIATE", though. **Track temperature** is
the strongest available wet-vs-slick discriminator at a rainy pit (a warm track
sheds water, so the rain is light/drying and teams fit slicks; a cool track holds
it). So the gate fits INTERMEDIATE only when it is raining *and* the track is cool
(< ``WET_TRACK_TEMP_C``); on a warm-but-raining pit it defers to the slick head.
This lifts wet-regime accuracy ~0.61 -> ~0.70 (leak-free nested CV) and halves the
variance. Full WET is never predicted (only ~13 stops, and inter-vs-wet needs the
intensity we lack); WET is collapsed into INTERMEDIATE everywhere.

The *timing* of wet<->dry switches (rain onset, dry-out crossover) is a separate,
learnable problem handled by the pit-timing head's rain features — not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from tire_strategy.data.features import DEFAULT_WET_COMPOUND, WET_TRACK_TEMP_C

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
    compound_model: Any, features: pd.DataFrame, *, rainfall: bool, track_temp: float
) -> CompoundRecommendation:
    """Recommend a compound for one race state, gating on rainfall + track temp.

    On a cool-track rainy pit (``track_temp < WET_TRACK_TEMP_C``) return the
    rule-based wet tyre (``DEFAULT_WET_COMPOUND``) at probability 1.0. Otherwise —
    dry, or warm-and-raining (likely light/drying) — defer to the slick compound
    head. ``features`` must be a single-row frame of the model's feature columns.
    """
    if rainfall and track_temp < WET_TRACK_TEMP_C:
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
