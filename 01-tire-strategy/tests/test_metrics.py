"""Tests for the asymmetric pit-cost metrics."""

from __future__ import annotations

import pytest

from tire_strategy.models import asymmetric_pit_cost, pit_cost_matrix
from tire_strategy.models.metrics import (
    best_cost_threshold,
    best_f1_threshold,
    expected_pit_cost,
)


def test_cost_matrix_shape_and_values() -> None:
    c = pit_cost_matrix(cost_fn=5.0, cost_fp=1.0)
    assert c.shape == (2, 2)
    assert c[0, 0] == 0.0 and c[1, 1] == 0.0  # correct calls are free
    assert c[1, 0] == 5.0  # missed pit
    assert c[0, 1] == 1.0  # spurious pit


def test_asymmetric_cost_penalises_missed_pits_more() -> None:
    y_true = [1, 0, 1, 0]
    y_pred = [0, 0, 1, 1]  # one missed pit (idx 0), one spurious pit (idx 3)
    # cost = (5*1 + 1*1) / 4 = 1.5
    assert asymmetric_pit_cost(y_true, y_pred, cost_fn=5.0, cost_fp=1.0) == pytest.approx(1.5)

    # A perfect prediction costs nothing.
    assert asymmetric_pit_cost(y_true, y_true) == 0.0


def test_cost_input_validation() -> None:
    with pytest.raises(ValueError):
        asymmetric_pit_cost([1, 0], [1])
    with pytest.raises(ValueError):
        asymmetric_pit_cost([], [])


def test_expected_and_best_threshold() -> None:
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.6, 0.9]
    # Threshold 0.5 classifies perfectly -> zero cost.
    assert expected_pit_cost(y_true, proba, threshold=0.5) == 0.0
    thr, cost = best_cost_threshold(y_true, proba)
    assert cost == 0.0
    assert 0.4 < thr <= 0.6


def test_best_f1_threshold_separates_classes_and_beats_never_pit() -> None:
    y_true = [0, 0, 1, 1]
    proba = [0.1, 0.4, 0.6, 0.9]
    # A threshold in (0.4, 0.6] classifies perfectly -> F1 == 1.0.
    thr, f1 = best_f1_threshold(y_true, proba)
    assert f1 == 1.0
    assert 0.4 < thr <= 0.6

    # On an imbalanced target where "never pit" wins on accuracy, F1 must NOT pick
    # the degenerate all-negative threshold (which would give F1 = 0).
    y_imb = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    p_imb = [0.05, 0.1, 0.1, 0.15, 0.2, 0.2, 0.25, 0.3, 0.55, 0.8]
    thr2, f1_2 = best_f1_threshold(y_imb, p_imb)
    assert f1_2 > 0.0
    assert thr2 <= 0.55  # low enough to actually catch the positives
