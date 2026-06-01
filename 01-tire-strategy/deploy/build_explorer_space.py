"""Assemble a self-contained Hugging Face Space for the F1 Data Explorer.

Mirrors the repo layout (so the package's path config resolves) and bundles the
processed parquets the explorer browses, alongside the Space's app.py /
requirements.txt / README.md. Push the output folder's contents to a Gradio Space.

    uv run python 01-tire-strategy/deploy/build_explorer_space.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # 01-tire-strategy/deploy
PROJECT = HERE.parent  # 01-tire-strategy
TEMPLATE = HERE / "explorer_space"
OUT = HERE / "explorer_space_build"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")
_PARQUETS = (
    "tire_strategy_dataset.parquet",
    "stint_laps.parquet",
    "telemetry_features.parquet",
)


def main() -> int:
    processed = PROJECT / "data" / "processed"
    missing = [p for p in _PARQUETS if not (processed / p).exists()]
    if missing:
        print(
            f"Missing parquets {missing} — run `just data`/`stint-data`/telemetry first.",
            file=sys.stderr,
        )
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    inner = OUT / "01-tire-strategy"
    shutil.copytree(
        PROJECT / "src" / "tire_strategy", inner / "src" / "tire_strategy", ignore=_IGNORE
    )
    (inner / "data" / "processed").mkdir(parents=True, exist_ok=True)
    for parquet in _PARQUETS:
        shutil.copy(processed / parquet, inner / "data" / "processed" / parquet)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(TEMPLATE / name, OUT / name)

    print(f"Explorer Space assembled at: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
