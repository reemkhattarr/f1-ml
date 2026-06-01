"""Tests for leakage-safe temporal splitting."""

from __future__ import annotations

import pandas as pd

from tire_strategy.models import assign_temporal_split, race_group_key


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2024, 2025, 2025, 2021],
            "round": [5, 10, 3, 20, 1],
            "driver": ["A", "B", "C", "D", "E"],
        }
    )


def test_temporal_split_assignment() -> None:
    # Production defaults: train 2022-2024, hold out 2025 (val early, test late).
    out = assign_temporal_split(_df(), val_round_max=12)
    by_driver = dict(zip(out["driver"], out["split"], strict=True))
    assert by_driver["A"] == "train"  # 2022
    assert by_driver["B"] == "train"  # 2024
    assert by_driver["C"] == "val"  # 2025 round 3 <= 12
    assert by_driver["D"] == "test"  # 2025 round 20 > 12
    # A season outside the design (pre-2022 era) is left NA, not silently trained on.
    assert pd.isna(by_driver["E"])


def test_race_weekend_is_atomic() -> None:
    # Every lap of one race weekend must share a split.
    df = pd.DataFrame({"year": [2025, 2025, 2025], "round": [3, 3, 3], "driver": ["A", "A", "B"]})
    out = assign_temporal_split(df)
    assert (out["split"] == "val").all()  # 2025 round 3 -> all val


def test_race_group_key_format() -> None:
    keys = race_group_key(_df())
    assert keys.iloc[0] == "2022_5"
    assert keys.iloc[3] == "2025_20"
