"""Tests for the experiment-logging harness (record + render round-trip)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tire_strategy import experiments
from tire_strategy.experiments import Run, record_runs, render_report


@pytest.fixture
def tmp_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the experiment log/report to a temp dir for the duration of a test."""
    monkeypatch.setattr(experiments, "EXPERIMENTS_DIR", tmp_path)
    monkeypatch.setattr(experiments, "RESULTS_JSONL", tmp_path / "results.jsonl")
    monkeypatch.setattr(experiments, "RESULTS_MD", tmp_path / "RESULTS.md")
    return tmp_path


def test_record_appends_and_assigns_ids(tmp_log: Path) -> None:
    record_runs(
        [Run("grp", "a", {"x": 1}, {"cv_ap_mean": 0.5})],
        timestamp="2026-06-01T00:00:00+00:00",
    )
    record_runs(
        [Run("grp", "b", {"x": 2}, {"cv_ap_mean": 0.6})],
        timestamp="2026-06-01T00:01:00+00:00",
    )
    records = [json.loads(line) for line in (tmp_log / "results.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == [1, 2]  # ids increment across calls
    assert [r["label"] for r in records] == ["a", "b"]


def test_render_writes_comparison_and_details(tmp_log: Path) -> None:
    record_runs(
        [
            Run("season-scope", "no 2025", {"years": [2024]}, {"cv_model_cost_mean": 0.30}),
            Run("season-scope", "+2025", {"years": [2025]}, {"cv_model_cost_mean": 0.28}),
        ],
        timestamp="2026-06-01T00:00:00+00:00",
    )
    md = (tmp_log / "RESULTS.md").read_text()
    assert "## season-scope" in md
    assert "no 2025" in md and "+2025" in md
    assert "0.3000" in md and "0.2800" in md  # metric values formatted
    assert "Full parameters" in md  # details section present


def test_render_handles_bool_and_empty(tmp_log: Path) -> None:
    record_runs(
        [Run("g", "x", {}, {"cv_beats_both": True, "extra": 1.0})],
        timestamp="2026-06-01T00:00:00+00:00",
    )
    md = (tmp_log / "RESULTS.md").read_text()
    assert "✅" in md  # bool rendered as a check
    # A run missing a column still renders (blank cell), not a crash:
    render_report(tmp_log / "RESULTS.md")
    assert (tmp_log / "RESULTS.md").exists()
