"""f1_foundation — shared F1 data layer for the portfolio.

Owns the FastF1 ingestion (raw timing → clean per-lap tables) and the data
explorer, plus the shared cache/data configuration. Projects depend on this; it
depends on none of them.
"""

from __future__ import annotations

from f1_foundation.config import (
    CACHE_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    SEED,
    enable_cache,
)
from f1_foundation.ingest import load_race, load_results, retired_drivers

__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "PROCESSED_DIR",
    "SEED",
    "enable_cache",
    "load_race",
    "load_results",
    "retired_drivers",
]
