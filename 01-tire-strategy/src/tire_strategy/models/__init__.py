"""Modelling layer for the tire-strategy project: splitting, metrics, models."""

from tire_strategy.models.baselines import MajorityBaseline, RuleBasedBaseline
from tire_strategy.models.metrics import asymmetric_pit_cost, pit_cost_matrix
from tire_strategy.models.pipeline import build_compound_pipeline, build_pit_pipeline
from tire_strategy.models.split import (
    SPLIT_NAMES,
    assign_temporal_split,
    race_group_key,
)

__all__ = [
    "SPLIT_NAMES",
    "MajorityBaseline",
    "RuleBasedBaseline",
    "assign_temporal_split",
    "asymmetric_pit_cost",
    "build_compound_pipeline",
    "build_pit_pipeline",
    "pit_cost_matrix",
    "race_group_key",
]
