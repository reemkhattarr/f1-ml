"""The website's JavaScript (engine.js, model.js) must agree with Python."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from tire_strategy.config import PROJECT_ROOT
from tire_strategy.export_model import predict_compound, predict_pit
from tire_strategy.strategy.engine import advise, circuit_params, race_plan
from tire_strategy.strategy.params import fit_strategy_params

from .strategy_fixtures import synthetic_race_laps
from .test_export_model import _frame
from .test_export_model import models as models

SITE = PROJECT_ROOT / "site"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

RUNNER = """
const Engine = require(process.argv[1] + '/engine.js');
const Model = require(process.argv[1] + '/model.js');
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const out = {};
if (input.params) {
  const p = Engine.circuitParams(input.params, input.circuit);
  out.plans = input.laps.map((n) => Engine.racePlan(p, n));
  out.advice = input.states.map((s) => Engine.advise(p, s));
}
if (input.models) {
  out.pit = input.rows.map((r) => Model.pitProbability(input.models, r));
  out.compound = input.rows.map((r) => Model.compoundProbabilities(input.models, r));
}
process.stdout.write(JSON.stringify(out));
"""

STATES = [
    {"total_laps": 50, "lap": 18, "compound": "MEDIUM", "tyre_age": 17, "gap_ahead": 1.2,
     "gap_behind": 9.0},
    {"total_laps": 50, "lap": 10, "compound": "SOFT", "tyre_age": 9, "caution": "sc"},
    {"total_laps": 50, "lap": 35, "compound": "HARD", "tyre_age": 10, "used": ["MEDIUM"]},
    {"total_laps": 50, "lap": 44, "compound": "MEDIUM", "tyre_age": 30, "caution": "vsc"},
    {"total_laps": 50, "lap": 20, "compound": "SOFT", "tyre_age": 3, "rain": True},
]  # fmt: skip


def _node(payload: dict) -> dict:
    proc = subprocess.run(
        ["node", "-e", RUNNER, str(SITE)], input=json.dumps(payload), capture_output=True,
        text=True, check=True,
    )  # fmt: skip
    return json.loads(proc.stdout)


def _strip(plan: dict) -> dict:
    keep = ("stops", "stints", "pit_laps", "cost", "windows", "label")
    return {k: plan[k] for k in keep if k in plan}


def test_engine_matches_python() -> None:
    params = fit_strategy_params(synthetic_race_laps())
    laps = [50, 57, 70]
    js = _node({"params": params, "circuit": "Testville", "laps": laps, "states": STATES})
    p = circuit_params(params, "Testville")
    for n, got in zip(laps, js["plans"], strict=True):
        want = race_plan(p, n)
        assert [_strip(o) for o in got["by_stops"]] == [_strip(o) for o in want["by_stops"]]
        assert got["sensitivity"] == want["sensitivity"]
        assert [o["label"] for o in got["top"]] == [o["label"] for o in want["top"]]
    for state, got in zip(STATES, js["advice"], strict=True):
        want = advise(p, **state)
        assert (got["decision"], got["compound"], got["delta"]) == (
            want["decision"], want["compound"], want["delta"],
        )  # fmt: skip
        assert [b["cost"] for b in got["box"]] == [b["cost"] for b in want["box"]]
        assert len(got["notes"]) == len(want["notes"])


def test_models_match_python(models) -> None:
    _, _, exported = models
    rows = _frame(60, 9).replace({np.nan: None}).to_dict("records")
    js = _node({"models": exported, "rows": rows})
    want_pit = [predict_pit(exported, r) for r in rows]
    assert np.allclose(js["pit"], want_pit, atol=1e-9)
    for got, row in zip(js["compound"], rows, strict=True):
        want = predict_compound(exported, row)
        assert all(abs(got[k] - want[k]) < 1e-9 for k in want)
