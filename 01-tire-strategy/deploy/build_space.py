"""Assemble a self-contained Hugging Face Space directory for the Gradio demo.

The build mirrors the workspace layout so both packages' path config resolves
unchanged: `f1_foundation` (the shared data layer the demo imports transitively)
and `tire_strategy`, plus the trained artifacts, tuned config, and the pre-built
dataset parquet in the shared `data/processed/`. Push the contents of the output
folder to a Gradio Space.

Prerequisites: `just data` (dataset) and `just train` (artifacts) have been run.

    uv run python 01-tire-strategy/deploy/build_space.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent  # 01-tire-strategy/deploy
PROJECT = HERE.parent  # 01-tire-strategy
WORKSPACE = PROJECT.parent  # repo root
FOUNDATION = WORKSPACE / "00-data-explorer"
TEMPLATE = HERE / "hf_space"
OUT = HERE / "space_build"
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def main() -> int:
    processed = WORKSPACE / "data" / "processed"
    dataset = processed / "tire_strategy_dataset.parquet"
    artifacts = PROJECT / "artifacts"
    if not dataset.exists():
        print("Missing dataset — run `just data` first.", file=sys.stderr)
        return 1
    if not (artifacts / "metadata.json").exists():
        print("Missing trained artifacts — run `just train` first.", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)

    # Mirror the workspace: 00 (foundation) + 01 (this project), shared data/ at root.
    shutil.copytree(
        FOUNDATION / "src" / "f1_foundation",
        OUT / "00-data-explorer" / "src" / "f1_foundation",
        ignore=_IGNORE,
    )
    inner = OUT / "01-tire-strategy"
    shutil.copytree(
        PROJECT / "src" / "tire_strategy", inner / "src" / "tire_strategy", ignore=_IGNORE
    )
    shutil.copytree(artifacts, inner / "artifacts", ignore=_IGNORE)
    shutil.copytree(PROJECT / "configs", inner / "configs")

    (OUT / "data" / "processed").mkdir(parents=True, exist_ok=True)
    shutil.copy(dataset, OUT / "data" / "processed" / dataset.name)
    # Optional: race results power the demo's classification panel + notes.
    results = processed / "results.parquet"
    if results.exists():
        shutil.copy(results, OUT / "data" / "processed" / results.name)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(TEMPLATE / name, OUT / name)

    print(f"Space assembled at: {OUT}")
    print("Next: create a Gradio Space on Hugging Face, then push this folder's contents.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
