"""Tests for dataset assembly.

Failure-handling is tested with monkeypatching (no network); the happy-path
roundtrip is an integration test against a single cached race.
"""

from __future__ import annotations

import pytest

import tire_strategy.data.dataset as bd
from tire_strategy.data import FEATURE_COLUMNS


def test_build_one_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise ValueError("bad race")

    monkeypatch.setattr(bd, "load_labelled_race", boom)
    assert bd.build_one(2024, 1) is None


def test_assemble_raises_when_everything_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.setattr(bd, "build_one", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError):
        bd.assemble([(2024, 1)], out_path=None)


@pytest.mark.integration
def test_assemble_single_race_roundtrips_parquet(tmp_path) -> None:
    out = tmp_path / "ds.parquet"
    data = bd.assemble([(2024, "Monza")], out_path=out)

    assert len(data) > 900
    for col in FEATURE_COLUMNS:
        assert col in data.columns
    assert out.exists()

    reloaded = bd.load_dataset(out)
    assert len(reloaded) == len(data)
    assert list(reloaded.columns) == list(data.columns)
