"""Download the latest dataset and trained models from the GitHub release.

The weekly cloud run stores everything on the ``data-latest`` release, so a fresh
clone can run the full local app without downloading races or training:

    just fetch      # then: just app
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from tire_strategy.config import ARTIFACTS_DIR, PROCESSED_DIR

RELEASE = "https://github.com/reemkhattarr/f1-ml/releases/download/data-latest"
FILES: dict[str, Path] = {
    "tire_strategy_dataset.parquet": PROCESSED_DIR,
    "race_laps.parquet": PROCESSED_DIR,
    "results.parquet": PROCESSED_DIR,
    "strategy_params.json": PROCESSED_DIR,
    "pit_model.joblib": ARTIFACTS_DIR,
    "compound_model.joblib": ARTIFACTS_DIR,
    "metadata.json": ARTIFACTS_DIR,
}


def fetch(files: dict[str, Path] = FILES, base: str = RELEASE) -> list[str]:
    """Download every file that exists on the release; return the ones fetched."""
    got = []
    for name, folder in files.items():
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(f"{base}/{name}", timeout=120) as response:
                (folder / name).write_bytes(response.read())
            got.append(name)
            print(f"  {name} -> {folder}")
        except urllib.error.HTTPError as exc:
            print(f"  {name}: not on the release ({exc.code})")
    return got


if __name__ == "__main__":
    print("Downloading the latest data and models from the data-latest release:")
    sys.exit(0 if fetch() else 1)
