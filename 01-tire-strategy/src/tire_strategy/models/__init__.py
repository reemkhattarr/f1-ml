"""Modelling layer for the tire-strategy project: splitting, metrics, models."""

from tire_strategy.models.baselines import MajorityBaseline, RuleBasedBaseline
from tire_strategy.models.cv import (
    cost_ratio_sweep,
    cross_validate_compound,
    cross_validate_pit,
    evaluate_wet_gate,
    tune_pit,
)
from tire_strategy.models.metrics import asymmetric_pit_cost, pit_cost_matrix
from tire_strategy.models.pipeline import (
    DEFAULT_FEATURES,
    FeatureSpec,
    build_compound_pipeline,
    build_pit_pipeline,
)
from tire_strategy.models.split import (
    SPLIT_NAMES,
    assign_temporal_split,
    default_split_seasons,
    race_group_key,
)

__all__ = [
    "DEFAULT_FEATURES",
    "SPLIT_NAMES",
    "FeatureSpec",
    "MajorityBaseline",
    "RuleBasedBaseline",
    "assign_temporal_split",
    "asymmetric_pit_cost",
    "build_compound_pipeline",
    "build_pit_pipeline",
    "cost_ratio_sweep",
    "cross_validate_compound",
    "cross_validate_pit",
    "default_split_seasons",
    "evaluate_wet_gate",
    "pit_cost_matrix",
    "race_group_key",
    "tune_pit",
]
