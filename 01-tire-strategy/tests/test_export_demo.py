"""Tests for the static race-replay export (with stub models, no training)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tire_strategy.export_demo import export, race_payload
from tire_strategy.serving.models import ModelBundle


class _PitStub:
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p = (df["tyre_age"].to_numpy() / 20).clip(0, 1)
        return np.column_stack([1 - p, p])


class _CompoundStub:
    classes_ = np.array(["HARD", "MEDIUM", "SOFT"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile([0.6, 0.3, 0.1], (len(df), 1))


def _race(year: int = 2026, rnd: int = 3) -> pd.DataFrame:
    rows = []
    for driver, team in (("VER", "Red Bull"), ("LEC", "Ferrari")):
        for lap in range(1, 6):
            rows.append(
                {
                    "year": year, "round": rnd, "circuit": "Melbourne", "driver": driver,
                    "team": team, "lap_number": lap, "compound": "MEDIUM", "tyre_age": lap * 4,
                    "rainfall": lap == 5, "under_caution": False,
                    "label_pit": int(driver == "LEC" and lap == 4),
                    "label_compound": "HARD" if driver == "LEC" and lap == 4 else None,
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def _bundle() -> ModelBundle:
    return ModelBundle(_PitStub(), _CompoundStub(), threshold=0.5, slick_compounds=["SOFT"])


def test_race_payload_marks_model_and_team_calls() -> None:
    payload = race_payload(_race(), _bundle())
    assert payload["laps"] == 5 and payload["circuit"] == "Melbourne"
    cols = payload["columns"]
    lec = next(d for d in payload["drivers"] if d["code"] == "LEC")
    lap4 = dict(zip(cols, lec["laps"][3], strict=True))
    assert lap4["team_pit"] == 1 and lap4["team_compound"] == "H"
    assert lap4["pit_prob"] == 0.8 and lap4["model_pit"] == 1 and lap4["model_compound"] == "H"
    lap5 = dict(zip(cols, lec["laps"][4], strict=True))
    assert lap5["rain"] == 1 and lap5["model_compound"] == "I"  # wet rule overrides slicks
    lap1 = dict(zip(cols, lec["laps"][0], strict=True))
    assert lap1["model_pit"] == 0 and lap1["model_compound"] == ""


def test_export_writes_index_and_race_files(tmp_path) -> None:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<html></html>")
    data = pd.concat([_race(2025, 1), _race(2026, 3)])
    meta = {
        "pit_threshold": 0.5,
        "trained_at": "2026-11-21",
        "metrics": {"cv_model_cost_mean": 0.3},
    }
    index = export(data, _bundle(), meta, out_dir=tmp_path / "out", site_src=site)
    assert [r["file"] for r in index] == ["data/races/2025-01.json", "data/races/2026-03.json"]
    out = tmp_path / "out"
    assert (out / "index.html").exists()
    card = json.loads((out / "data" / "model.json").read_text())
    assert card["metrics"] == {"cv_model_cost_mean": 0.3}
    assert json.loads((out / "data" / "races" / "2026-03.json").read_text())["year"] == 2026
