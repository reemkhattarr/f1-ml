"""Shared paths and FastF1 cache setup for the whole portfolio.

Every project (01-tire-strategy and beyond) builds on this foundation, so the
FastF1 on-disk cache and the processed-data directory live here, at the workspace
root, and are shared across projects. A single cache means a race downloaded for
one project is instantly available to the next; a single ``data/processed``
directory is what lets the data explorer browse every project's outputs without
depending back on any of them.

FastF1 hits the live F1 timing API; the on-disk cache is mandatory to avoid rate
limits and keep re-runs fast. Every entry point that loads F1 data should call
:func:`enable_cache` first.
"""

from __future__ import annotations

from pathlib import Path

import fastf1

# Global seed for reproducible splits, model training and any sampling.
SEED: int = 42

# Workspace root = .../f1-ml (three parents up: f1_foundation -> src -> 00-... -> root).
WORKSPACE_ROOT: Path = Path(__file__).resolve().parents[3]

# Shared, gitignored data lives at the workspace root so every project + the
# explorer read and write the same FastF1 cache and processed parquets.
DATA_DIR: Path = WORKSPACE_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"
PROCESSED_DIR: Path = DATA_DIR / "processed"


def enable_cache(cache_dir: Path = CACHE_DIR) -> Path:
    """Enable the FastF1 on-disk cache, creating the directory if needed.

    Returns the resolved cache directory so callers can log it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir
