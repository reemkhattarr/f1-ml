"""Assemble a self-contained Hugging Face Space for the F1 Data Explorer.

The explorer lives in the shared foundation (`f1_foundation`) and browses the
processed parquets every project writes to the shared `data/processed/` dir. The
build mirrors the workspace layout so `f1_foundation.config` resolves the data
directory unchanged, bundling the foundation package + the parquets alongside the
Space's app.py / requirements.txt / README.md.

    uv run python 00-data-explorer/deploy/build_explorer_space.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # 00-data-explorer/deploy
PROJECT = HERE.parent  # 00-data-explorer
WORKSPACE = PROJECT.parent  # repo root
TEMPLATE = HERE / "explorer_space"
OUT = HERE / "explorer_space_build"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")
# Raw parquets the explorer builds itself, plus the project parquets it browses by
# convention (shown only when present). Foundation never imports the projects.
_PARQUETS = (
    "raw_laps.parquet",
    "weather.parquet",
    "results.parquet",
    "tire_strategy_dataset.parquet",
    "stint_laps.parquet",
    "telemetry_features.parquet",
)


def main() -> int:
    processed = WORKSPACE / "data" / "processed"
    present = [p for p in _PARQUETS if (processed / p).exists()]
    if "raw_laps.parquet" not in present:
        print("Missing raw_laps.parquet — run the explorer data build first.", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    # Mirror the workspace so f1_foundation.config.WORKSPACE_ROOT (= parents[3])
    # resolves to OUT and DATA_DIR resolves to OUT/data.
    shutil.copytree(
        PROJECT / "src" / "f1_foundation",
        OUT / "00-data-explorer" / "src" / "f1_foundation",
        ignore=_IGNORE,
    )
    (OUT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    for parquet in present:
        shutil.copy(processed / parquet, OUT / "data" / "processed" / parquet)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(TEMPLATE / name, OUT / name)

    print(f"Explorer Space assembled at: {OUT}  ({len(present)} parquet(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
