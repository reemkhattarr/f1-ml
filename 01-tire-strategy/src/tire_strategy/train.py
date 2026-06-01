"""Train and evaluate the two-head tire-strategy model.

Pipeline (see ``docs/DESIGN.md``):
1. Load the assembled dataset and apply the race-weekend-atomic temporal split.
2. **Pit head** — fit LightGBM (``scale_pos_weight`` for the imbalance), calibrate
   on the validation set (isotonic), and pick the decision threshold that minimises
   the asymmetric strategic cost on validation.
3. **Compound head** — fit LightGBM on slick pit-laps only.
4. Evaluate on the held-out late-season test races against the majority and
   rule-based baselines; log params, metrics and plots to MLflow; save artifacts.

Run with ``just train`` or ``python -m tire_strategy.train``.
"""

from __future__ import annotations

import json
import logging
from typing import cast

import joblib
import mlflow
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from tire_strategy.config import ARTIFACTS_DIR, MLFLOW_DIR, SEED
from tire_strategy.data import SLICK_COMPOUNDS, load_dataset
from tire_strategy.evaluate import pit_metrics, plot_confusion, plot_reliability
from tire_strategy.models import (
    MajorityBaseline,
    RuleBasedBaseline,
    assign_temporal_split,
    build_compound_pipeline,
    build_pit_pipeline,
)
from tire_strategy.models.metrics import asymmetric_pit_cost, best_cost_threshold

logger = logging.getLogger(__name__)

EXPERIMENT = "tire-strategy"


def _prepare() -> dict[str, pd.DataFrame]:
    """Load the dataset, split it temporally, and keep only eligible labelled rows."""
    df = assign_temporal_split(load_dataset())
    df = df[df["split"].notna() & df["train_eligible"].astype("boolean").fillna(False)]
    return {
        name: cast(pd.DataFrame, df[df["split"] == name].copy())
        for name in ("train", "val", "test")
    }


def _slick_pits(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that are strategic pit stops onto a slick compound (compound head data)."""
    mask = (frame["label_pit"] == 1) & frame["label_compound"].isin(SLICK_COMPOUNDS)
    return cast(pd.DataFrame, frame[mask].copy())


def train() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    # SQLite backend (the file store is deprecated in current MLflow).
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(MLFLOW_DIR / "mlartifacts").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    splits = _prepare()
    train_df, val_df, test_df = splits["train"], splits["val"], splits["test"]
    y_train = train_df["label_pit"].astype(int)
    y_val = val_df["label_pit"].astype(int)
    y_test = test_df["label_pit"].astype(int)

    # scale_pos_weight = negatives / positives on the training split.
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "seed": SEED,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "train_pit_rate": round(pos / len(train_df), 5),
                "scale_pos_weight": round(scale_pos_weight, 3),
            }
        )

        # --- Pit head: fit on train, calibrate on val ---
        logger.info("Fitting pit head (scale_pos_weight=%.2f) ...", scale_pos_weight)
        pit_pipeline = build_pit_pipeline(scale_pos_weight=scale_pos_weight)
        pit_pipeline.fit(train_df, y_train)

        # FrozenEstimator marks the fitted pipeline as prefit, so calibration runs
        # on the validation set only (sklearn 1.6 replaced cv="prefit" with this).
        calibrated = CalibratedClassifierCV(FrozenEstimator(pit_pipeline), method="isotonic")
        calibrated.fit(val_df, y_val)

        proba_val = calibrated.predict_proba(val_df)[:, 1]
        threshold, val_cost = best_cost_threshold(y_val, proba_val)
        logger.info("Chosen pit threshold=%.3f (val cost=%.4f)", threshold, val_cost)

        proba_test = calibrated.predict_proba(test_df)[:, 1]
        pit_test = pit_metrics(y_test, proba_test, threshold)

        # --- Baselines on the test set ---
        maj_pred = MajorityBaseline().predict(test_df)
        rule_pred = RuleBasedBaseline().predict(test_df)
        baseline_costs = {
            "baseline_majority_cost": asymmetric_pit_cost(y_test, maj_pred),
            "baseline_rule_cost": asymmetric_pit_cost(y_test, rule_pred),
        }

        # --- Compound head: slick pit-laps only ---
        comp_train = _slick_pits(train_df)
        comp_test = _slick_pits(test_df)
        logger.info("Fitting compound head (%d train pits) ...", len(comp_train))
        compound_pipeline = build_compound_pipeline()
        compound_pipeline.fit(comp_train, comp_train["label_compound"].astype(str))
        comp_pred = compound_pipeline.predict(comp_test)
        compound_acc = float((comp_pred == comp_test["label_compound"].astype(str)).mean())

        # --- Log metrics ---
        metrics = {
            **{f"pit_{k}": v for k, v in pit_test.items()},
            **baseline_costs,
            "compound_accuracy": compound_acc,
            "compound_n_test": float(len(comp_test)),
        }
        mlflow.log_metrics(metrics)

        # --- Plots ---
        pit_pred_test = (proba_test >= threshold).astype(int)
        rel_path = ARTIFACTS_DIR / "pit_calibration.png"
        pit_cm_path = ARTIFACTS_DIR / "pit_confusion.png"
        comp_cm_path = ARTIFACTS_DIR / "compound_confusion.png"
        plot_reliability(y_test, proba_test, rel_path)
        plot_confusion(y_test, pit_pred_test, [0, 1], pit_cm_path, "Pit decision (test)")
        plot_confusion(
            comp_test["label_compound"].astype(str),
            comp_pred,
            list(SLICK_COMPOUNDS),
            comp_cm_path,
            "Compound choice (test)",
        )
        for plot in (rel_path, pit_cm_path, comp_cm_path):
            mlflow.log_artifact(str(plot))

        # --- Save artifacts ---
        joblib.dump(calibrated, ARTIFACTS_DIR / "pit_model.joblib")
        joblib.dump(compound_pipeline, ARTIFACTS_DIR / "compound_model.joblib")
        metadata = {
            "pit_threshold": threshold,
            "scale_pos_weight": scale_pos_weight,
            "slick_compounds": list(SLICK_COMPOUNDS),
            "metrics": metrics,
        }
        (ARTIFACTS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(ARTIFACTS_DIR / "metadata.json"))

    _log_summary(metrics, threshold)
    return metrics


def _log_summary(metrics: dict, threshold: float) -> None:
    ap = metrics["pit_average_precision"]
    auc = metrics["pit_roc_auc"]
    n_comp = int(metrics["compound_n_test"])
    print("\n=== Tire-strategy model — held-out test results ===")
    print(f"Pit head    AP={ap:.3f}  ROC-AUC={auc:.3f}")
    print(
        f"            recall={metrics['pit_recall']:.3f}"
        f"  precision={metrics['pit_precision']:.3f}  (threshold={threshold:.3f})"
    )
    print("Asymmetric cost (lower is better):")
    print(f"            model       = {metrics['pit_asymmetric_cost']:.4f}")
    print(f"            rule-based  = {metrics['baseline_rule_cost']:.4f}")
    print(f"            majority    = {metrics['baseline_majority_cost']:.4f}")
    print(f"Compound head accuracy = {metrics['compound_accuracy']:.3f}  (n={n_comp})")


if __name__ == "__main__":
    train()
