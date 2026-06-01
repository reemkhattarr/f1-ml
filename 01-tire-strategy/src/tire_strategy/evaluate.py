"""Evaluation metrics and diagnostic plots for the tire-strategy models.

Accuracy is deliberately absent from the pit metrics: under ~97% stay-out it is
meaningless. We report ranking quality (average precision, ROC-AUC), the
operating point (precision/recall at the chosen threshold), the strategic
:func:`asymmetric_pit_cost`, and a calibration curve — the things that actually
tell you whether the model is useful and trustworthy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from tire_strategy.models.metrics import asymmetric_pit_cost

plt.switch_backend("Agg")  # headless rendering


def pit_metrics(
    y_true: npt.ArrayLike,
    pit_proba: npt.ArrayLike,
    threshold: float,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> dict[str, float]:
    """Compute the pit-head metrics that matter under heavy imbalance."""
    yt = np.asarray(y_true).astype(int)
    proba = np.asarray(pit_proba, dtype=float)
    yp = (proba >= threshold).astype(int)

    tp = float(((yt == 1) & (yp == 1)).sum())
    fp = float(((yt == 0) & (yp == 1)).sum())
    fn = float(((yt == 1) & (yp == 0)).sum())
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "average_precision": float(average_precision_score(yt, proba)),
        "roc_auc": float(roc_auc_score(yt, proba)),
        "recall": recall,
        "precision": precision,
        "asymmetric_cost": asymmetric_pit_cost(yt, yp, cost_fn, cost_fp),
        "threshold": float(threshold),
        "n": float(yt.size),
        "positives": float(yt.sum()),
    }


def plot_reliability(
    y_true: npt.ArrayLike,
    pit_proba: npt.ArrayLike,
    path: Path,
    n_bins: int = 10,
    title: str = "Pit-probability calibration (test)",
) -> None:
    """Reliability curve: predicted probability vs observed frequency."""
    yt = np.asarray(y_true).astype(int)
    proba = np.asarray(pit_proba, dtype=float)
    prob_true, prob_pred = calibration_curve(yt, proba, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", color="tab:blue", label="model")
    ax.set_xlabel("Predicted pit probability")
    ax.set_ylabel("Observed pit frequency")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    labels: list,
    path: Path,
    title: str,
) -> None:
    """Annotated confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=[str(c) for c in labels])
    ax.set_yticks(range(len(labels)), labels=[str(c) for c in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
