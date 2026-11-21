"""Export the race-replay demo as a static website (no server needed).

For every race in the dataset, run the trained models over every driver-lap and
write the results as small JSON files next to the page in ``site/``. The output
folder can be served by any static host; CI publishes it to GitHub Pages.

    python -m tire_strategy.export_demo            # -> deploy/site_build/

Output layout::

    index.html, app.js, style.css    copied from 01-tire-strategy/site/
    data/model.json                  threshold, split, metrics, training date
    data/races.json                  list of races (year, round, circuit, file)
    data/races/<year>-<round>.json   per-driver laps with model calls vs reality
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from tire_strategy.config import ARTIFACTS_DIR, PROJECT_ROOT
from tire_strategy.data import load_dataset
from tire_strategy.data.features import DEFAULT_WET_COMPOUND
from tire_strategy.serving.models import ModelBundle, load_bundle

logger = logging.getLogger(__name__)

SITE_SRC: Path = PROJECT_ROOT / "site"
OUT_DIR: Path = PROJECT_ROOT / "deploy" / "site_build"

# One letter per compound keeps the JSON small; the page maps them back.
COMPOUND_CODE = {"SOFT": "S", "MEDIUM": "M", "HARD": "H", "INTERMEDIATE": "I", "WET": "W"}
LAP_COLUMNS = ["lap", "compound", "tyre_age", "pit_prob", "model_pit", "model_compound",
               "team_pit", "team_compound", "rain", "caution"]  # fmt: skip

# Metrics shown on the page (subset of metadata["metrics"]).
SHOWN_METRICS = (
    "cv_model_cost_mean",
    "cv_majority_cost_mean",
    "cv_rule_cost_mean",
    "cv_ap_mean",
    "cv_compound_accuracy_mean",
    "pit_roc_auc",
    "pit_average_precision",
    "compound_accuracy",
)


def _code(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return ""
    return COMPOUND_CODE.get(str(value), "")


def _flag(series: pd.Series) -> np.ndarray:
    return series.astype("boolean").fillna(False).astype(bool).to_numpy()


def race_payload(race: pd.DataFrame, bundle: ModelBundle) -> dict:
    """Model calls vs what the team did, for every driver-lap of one race."""
    race = race.sort_values(["driver", "lap_number"]).reset_index(drop=True)
    pit_prob = bundle.pit_model.predict_proba(race)[:, 1]
    comp_proba = bundle.compound_model.predict_proba(race)
    classes = [str(c) for c in bundle.compound_model.classes_]
    slick_pick = np.array(classes)[comp_proba.argmax(axis=1)]
    rain = _flag(cast(pd.Series, race["rainfall"]))
    model_compound = np.where(rain, DEFAULT_WET_COMPOUND, slick_pick)
    model_pit = pit_prob >= bundle.threshold
    caution = (
        _flag(cast(pd.Series, race["under_caution"]))
        if "under_caution" in race
        else np.zeros(len(race), dtype=bool)
    )

    drivers = []
    for driver, idx in race.groupby("driver", sort=False).groups.items():
        rows = race.loc[idx]
        laps = []
        for i, row in rows.iterrows():
            i = cast(int, i)
            team_pit = bool(pd.notna(row["label_pit"]) and int(row["label_pit"]) == 1)
            laps.append([
                int(row["lap_number"]),
                _code(row["compound"]),
                int(row["tyre_age"]) if pd.notna(row["tyre_age"]) else None,
                round(float(pit_prob[i]), 3),
                int(model_pit[i]),
                _code(model_compound[i]) if model_pit[i] else "",
                int(team_pit),
                _code(row["label_compound"]) if team_pit else "",
                int(rain[i]),
                int(caution[i]),
            ])  # fmt: skip
        team = rows["team"].iloc[0] if "team" in rows else None
        team_name = str(team) if isinstance(team, str) else ""
        drivers.append({"code": str(driver), "team": team_name, "laps": laps})
    drivers.sort(key=lambda d: (d["team"], d["code"]))
    first = race.iloc[0]
    return {
        "year": int(first["year"]),
        "round": int(first["round"]),
        "circuit": str(first["circuit"]),
        "laps": int(race["lap_number"].max()),
        "threshold": bundle.threshold,
        "columns": LAP_COLUMNS,
        "drivers": drivers,
    }


def model_card(metadata: dict) -> dict:
    metrics = metadata.get("metrics", {})
    return {
        "threshold": metadata.get("pit_threshold"),
        "trained_at": metadata.get("trained_at"),
        "split": metadata.get("split"),
        "metrics": {k: metrics[k] for k in SHOWN_METRICS if k in metrics},
    }


def export(
    dataset: pd.DataFrame,
    bundle: ModelBundle,
    metadata: dict,
    out_dir: Path = OUT_DIR,
    site_src: Path = SITE_SRC,
) -> list[dict]:
    """Write the static site to ``out_dir`` and return the race index."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "data" / "races").mkdir(parents=True)
    if site_src.exists():
        for asset in site_src.iterdir():
            if asset.is_file():
                shutil.copy(asset, out_dir / asset.name)

    index = []
    for key, race in dataset.groupby(["year", "round"], sort=True):
        year, rnd = cast(tuple[int, int], key)
        payload = race_payload(race, bundle)
        name = f"{int(year)}-{int(rnd):02d}.json"
        (out_dir / "data" / "races" / name).write_text(json.dumps(payload, separators=(",", ":")))
        index.append({"year": int(year), "round": int(rnd), "circuit": payload["circuit"],
                      "file": f"data/races/{name}"})  # fmt: skip
    (out_dir / "data" / "races.json").write_text(json.dumps(index, separators=(",", ":")))
    (out_dir / "data" / "model.json").write_text(json.dumps(model_card(metadata), indent=1))
    (out_dir / "data" / "metadata.json").write_text(json.dumps(metadata, indent=1))
    logger.info("Exported %d races to %s", len(index), out_dir)
    return index


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    export(load_dataset(), load_bundle(), metadata)


if __name__ == "__main__":
    main()
