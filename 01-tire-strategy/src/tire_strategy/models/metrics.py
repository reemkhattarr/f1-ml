"""Custom evaluation metrics for the pit-decision model.

Accuracy is meaningless here: with ~98% "stay out" laps, a model that never pits
scores ~98% and is useless. What matters strategically is **asymmetric**: failing
to pit when the team did (tyres past their cliff, or the undercut window missed)
is far costlier than an extra precautionary stop. This module encodes that as a
cost matrix and an expected-cost metric, so model selection optimises the thing a
strategist actually cares about.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Default costs (relative units). Missing a real pit is set 5x worse than a
# spurious one — a deliberate, tunable domain choice, not a statistical default.
DEFAULT_COST_FN: float = 5.0  # predicted "stay out" but the team pitted
DEFAULT_COST_FP: float = 1.0  # predicted "pit" but the team stayed out


def pit_cost_matrix(
    cost_fn: float = DEFAULT_COST_FN, cost_fp: float = DEFAULT_COST_FP
) -> npt.NDArray[np.float64]:
    """Return the 2x2 cost matrix ``C[true, pred]`` (correct calls cost 0)."""
    return np.array([[0.0, cost_fp], [cost_fn, 0.0]], dtype=np.float64)


def asymmetric_pit_cost(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """Mean asymmetric cost per decision (lower is better).

    A false negative (missed pit) costs ``cost_fn``; a false positive (spurious
    pit) costs ``cost_fp``; correct calls cost 0.
    """
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if yt.size == 0:
        raise ValueError("inputs are empty")

    false_neg = (yt == 1) & (yp == 0)
    false_pos = (yt == 0) & (yp == 1)
    total = cost_fn * false_neg.sum() + cost_fp * false_pos.sum()
    return float(total / yt.size)


def expected_pit_cost(
    y_true: npt.ArrayLike,
    pit_proba: npt.ArrayLike,
    threshold: float = 0.5,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """Asymmetric cost when thresholding predicted pit probabilities.

    Useful for choosing a decision threshold that minimises strategic cost rather
    than defaulting to 0.5.
    """
    proba = np.asarray(pit_proba, dtype=np.float64)
    y_pred = (proba >= threshold).astype(int)
    return asymmetric_pit_cost(y_true, y_pred, cost_fn=cost_fn, cost_fp=cost_fp)


def best_cost_threshold(
    y_true: npt.ArrayLike,
    pit_proba: npt.ArrayLike,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_steps: int = 101,
) -> tuple[float, float]:
    """Grid-search the probability threshold minimising expected cost.

    Returns ``(threshold, cost)``.
    """
    thresholds = np.linspace(0.0, 1.0, n_steps)
    costs = [
        expected_pit_cost(y_true, pit_proba, t, cost_fn=cost_fn, cost_fp=cost_fp)
        for t in thresholds
    ]
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def best_f1_threshold(
    y_true: npt.ArrayLike,
    pit_proba: npt.ArrayLike,
    n_steps: int = 101,
) -> tuple[float, float]:
    """Grid-search the probability threshold maximising F1.

    F1 (harmonic mean of precision and recall) is the right "predict correctly"
    objective for this imbalanced target. Plain accuracy is maximised by *never*
    pitting (~92% of laps are "stay out"), so it rewards a useless model; F1
    instead balances catching real pit windows (recall) against false alarms
    (precision). Returns ``(threshold, f1)``.
    """
    yt = np.asarray(y_true).astype(int)
    proba = np.asarray(pit_proba, dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, n_steps)
    f1s: list[float] = []
    for t in thresholds:
        yp = (proba >= t).astype(int)
        tp = float(((yt == 1) & (yp == 1)).sum())
        fp = float(((yt == 0) & (yp == 1)).sum())
        fn = float(((yt == 1) & (yp == 0)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    best = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])
