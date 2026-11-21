"""Tests for the model quality gate."""

from __future__ import annotations

import json

from tire_strategy.gate import check, main


def _meta(cost: float, majority: float = 0.45, rule: float = 0.44) -> dict:
    return {
        "metrics": {
            "cv_model_cost_mean": cost,
            "cv_majority_cost_mean": majority,
            "cv_rule_cost_mean": rule,
        }
    }


def test_passes_when_no_worse_than_live() -> None:
    assert check(_meta(0.33), _meta(0.32)).passed  # within tolerance


def test_fails_when_clearly_worse_than_live() -> None:
    result = check(_meta(0.40), _meta(0.32))
    assert not result.passed
    assert any("worse than live" in r for r in result.reasons)


def test_fails_when_not_beating_baselines() -> None:
    assert not check(_meta(0.46, rule=0.44), None).passed


def test_first_model_only_needs_to_beat_baselines() -> None:
    assert check(_meta(0.31), None).passed


def test_cli_exit_codes(tmp_path) -> None:
    cand, live = tmp_path / "cand.json", tmp_path / "live.json"
    cand.write_text(json.dumps(_meta(0.31)))
    live.write_text(json.dumps(_meta(0.32)))
    assert main([str(cand), str(live)]) == 0
    assert main([str(cand), str(tmp_path / "missing.json")]) == 0  # no live model yet
    cand.write_text(json.dumps(_meta(0.50)))
    assert main([str(cand), str(live)]) == 1
