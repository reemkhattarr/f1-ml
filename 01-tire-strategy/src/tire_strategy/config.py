"""Project paths and FastF1 cache setup for tire-strategy.

The FastF1 cache and the processed-data directory are **shared across the whole
portfolio** and owned by :mod:`f1_foundation.config`; this module re-exports them
so existing ``from tire_strategy.config import DATA_DIR`` imports keep working,
and adds the paths that are specific to *this* project (artifacts, MLflow, tuned
configs). Every entry point that loads F1 data should call :func:`enable_cache`.
"""

from __future__ import annotations

from pathlib import Path

# Shared, portfolio-wide (single FastF1 cache + one processed-data dir).
from f1_foundation.config import (
    CACHE_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    SEED,
    enable_cache,
)

# Project root = .../01-tire-strategy (two parents up from this file's package).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Project-specific, gitignored outputs.
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
MLFLOW_DIR: Path = PROJECT_ROOT / "mlflow"

# Tracked configs (small, version-controlled): tuned hyperparameters live here so
# the tuned model is reproducible from a fresh clone without re-running Optuna.
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
TUNED_PARAMS_PATH: Path = CONFIGS_DIR / "best_params.json"

__all__ = [
    "ARTIFACTS_DIR",
    "CACHE_DIR",
    "CONFIGS_DIR",
    "DATA_DIR",
    "MLFLOW_DIR",
    "PROCESSED_DIR",
    "PROJECT_ROOT",
    "SEED",
    "TUNED_PARAMS_PATH",
    "enable_cache",
]
