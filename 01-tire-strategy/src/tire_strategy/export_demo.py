"""Export the race-replay demo as a static website (no server needed).

For every race in the dataset, run the trained models over every driver-lap and
write the results as small JSON files next to the page in ``site/``. The output
folder can be served by any static host; CI publishes it to GitHub Pages.

    python -m tire_strategy.export_demo            # -> deploy/site_build/

Output layout::

    index.html, *.js, style.css      copied from 01-tire-strategy/site/
    data/model.json                  threshold, split, metrics, training date
    data/models.json                 both models as trees, run in the browser
    data/strategy_params.json        per-circuit tyre/pit/safety-car parameters
    data/calendar.json               this season's races and the next one
    data/history.json                what the top ten did at each circuit lately
    data/races.json                  list of races (year, round, circuit, file)
    data/races/<year>-<round>.json   per-driver laps with model calls vs reality
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from f1_foundation.circuits import canonical_circuit
from tire_strategy.config import ARTIFACTS_DIR, PROJECT_ROOT
from tire_strategy.data import load_dataset
from tire_strategy.data.features import DEFAULT_WET_COMPOUND
from tire_strategy.export_model import export_models
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


def strategy_history(race_laps: pd.DataFrame, per_circuit: int = 3, top: int = 10) -> dict:
    """What the top finishers did at each circuit in its latest races."""
    out: dict[str, list] = {}
    race_laps = race_laps.copy()
    race_laps["circuit"] = race_laps["circuit"].map(canonical_circuit)
    keys = cast(pd.DataFrame, race_laps[["year", "round", "circuit"]]).drop_duplicates()
    for circuit, group in keys.groupby("circuit"):
        latest = cast(pd.DataFrame, group).sort_values(["year", "round"]).tail(per_circuit)
        races = []
        for year, rnd in zip(latest["year"], latest["round"], strict=True):
            mask = (race_laps["year"] == year) & (race_laps["round"] == rnd)
            race = cast(pd.DataFrame, race_laps[mask])
            total = int(race["lap_number"].max())
            last = cast(pd.DataFrame, race.sort_values("lap_number")).groupby("driver").tail(1)
            finishers = cast(pd.DataFrame, last[last["lap_number"] >= total - 1])
            finishers = finishers.sort_values(["lap_number", "position"], ascending=[False, True])
            drivers = []
            ranked = finishers.head(top)
            for pos, (code, team) in enumerate(
                zip(ranked["driver"], ranked["team"], strict=True), 1
            ):
                laps = cast(pd.DataFrame, race[race["driver"] == code]).sort_values("lap_number")
                stints = []
                for _, st in laps.groupby("stint", sort=True):
                    compound = cast(pd.DataFrame, st)["compound"].iloc[0]
                    stints.append([_code(compound) or "?", len(st)])
                drivers.append({"pos": pos, "code": str(code), "team": str(team),
                                "stints": stints})  # fmt: skip
            wet = bool(race["compound"].isin(["INTERMEDIATE", "WET"]).any())
            races.append({"year": int(year), "round": int(rnd), "laps": total, "wet": wet,
                          "drivers": drivers})  # fmt: skip
        out[str(circuit)] = races[::-1]
    return out


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
    params: dict | None = None,
    race_laps: pd.DataFrame | None = None,
    calendar: dict | None = None,
    track_record: dict | None = None,
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
    compact = {"separators": (",", ":"), "ensure_ascii": False}
    if params is not None:
        (out_dir / "data" / "strategy_params.json").write_text(json.dumps(params, **compact))
    if race_laps is not None:
        history = strategy_history(race_laps)
        (out_dir / "data" / "history.json").write_text(json.dumps(history, **compact))
    if calendar is not None:
        (out_dir / "data" / "calendar.json").write_text(json.dumps(calendar, **compact))
    if track_record is not None:
        (out_dir / "data" / "track_record.json").write_text(json.dumps(track_record, **compact))
    if hasattr(bundle.pit_model, "calibrated_classifiers_"):
        models = export_models(bundle.pit_model, bundle.compound_model, bundle.threshold)
        (out_dir / "data" / "models.json").write_text(json.dumps(models, **compact))
    logger.info("Exported %d races to %s", len(index), out_dir)
    return index


def season_calendar(now: datetime | None = None) -> dict:
    """This season's races (from OpenF1) and the next one to be run."""
    from f1_foundation.openf1 import race_calendar

    now = now or datetime.now(UTC)
    races = []
    for year in (now.year, now.year + 1):
        try:
            races += race_calendar(year)
        except Exception as exc:  # the calendar is a nice-to-have, never fatal
            logger.warning("No calendar for %s: %s", year, exc)
    upcoming = [r for r in races if (r.date_end or r.date_start) > now]
    return {
        "generated": now.isoformat(timespec="minutes"),
        "races": [
            {
                "year": r.year,
                "round": r.round,
                "circuit": r.circuit,
                "date": r.date_start.isoformat(),
            }
            for r in races
            if r.year == now.year
        ],
        "next": None
        if not upcoming
        else {
            "year": upcoming[0].year,
            "round": upcoming[0].round,
            "circuit": upcoming[0].circuit,
            "date": upcoming[0].date_start.isoformat(),
        },
    }


def main() -> None:
    from tire_strategy.data.dataset import RACE_LAPS_PATH
    from tire_strategy.strategy import track_record as tr
    from tire_strategy.strategy.params import (
        PARAMS_PATH,
        check_params,
        fit_strategy_params,
        save_params,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text())
    race_laps = pd.read_parquet(RACE_LAPS_PATH) if RACE_LAPS_PATH.exists() else None
    params = record = None
    calendar = season_calendar()
    if race_laps is not None:
        params = fit_strategy_params(race_laps)
        problems = check_params(params)
        if problems:
            # Never publish impossible numbers: fail the run, the live site stays.
            raise SystemExit("Strategy parameters failed the sanity check:\n" + "\n".join(problems))
        save_params(params, PARAMS_PATH)
        now = datetime.now(UTC).isoformat(timespec="minutes")
        log = tr.update_predictions(tr.load_predictions(), params, calendar, race_laps, now)
        tr.PREDICTIONS_PATH.write_text(json.dumps(log, indent=1, ensure_ascii=False))
        record = {**tr.backtest(race_laps), "live": log, "updated": now}
    export(load_dataset(), load_bundle(), metadata, params=params, race_laps=race_laps,
           calendar=calendar, track_record=record)  # fmt: skip


if __name__ == "__main__":
    main()
