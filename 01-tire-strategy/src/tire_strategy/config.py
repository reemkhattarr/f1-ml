"""Project paths and FastF1 cache setup.

FastF1 hits the live F1 timing API; a local on-disk cache is mandatory to avoid
rate limits and to keep development fast and reproducible. Every entry point
that loads F1 data should call :func:`enable_cache` first.
"""

from __future__ import annotations

from pathlib import Path

import fastf1

# Project root = .../01-tire-strategy (two parents up from this file's package).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"


def enable_cache(cache_dir: Path = CACHE_DIR) -> Path:
    """Enable the FastF1 on-disk cache, creating the directory if needed.

    Returns the resolved cache directory so callers can log it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    return cache_dir
