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

from tire_strategy.config import ARTIFACTS_DIR, MLFLOW_DIR, SEED, TUNED_PARAMS_PATH
from tire_strategy.data import SLICK_COMPOUNDS, load_dataset
from tire_strategy.data.labels import add_pit_window_label
from tire_strategy.evaluate import (
    pit_metrics,
    plot_confusion,
    plot_feature_importance,
    plot_reliability,
)
from tire_strategy.models import (
    MajorityBaseline,
    RuleBasedBaseline,
    assign_temporal_split,
    build_compound_pipeline,
    build_pit_pipeline,
    cross_validate_compound,
    cross_validate_pit,
)
from tire_strategy.models.baselines import MostFrequentCompound
from tire_strategy.models.metrics import asymmetric_pit_cost, best_cost_threshold

logger = logging.getLogger(__name__)

EXPERIMENT = "tire-strategy"

# Predict "a strategic pit happens within the next PIT_HORIZON laps" rather than
# "exactly this lap" — the exact-lap target is too sparse to beat "never pit".
PIT_HORIZON = 3


def _prepare(horizon: int = PIT_HORIZON) -> dict[str, pd.DataFrame]:
    """Load the dataset, add the pit-window target, split temporally, keep eligible rows."""
    df = assign_temporal_split(load_dataset())
    df = add_pit_window_label(df, horizon=horizon)
    df = df[df["split"].notna() & df["train_eligible"].astype("boolean").fillna(False)]
    return {
        name: cast(pd.DataFrame, df[df["split"] == name].copy())
        for name in ("train", "val", "test")
    }


def _slick_pits(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that are strategic pit stops onto a slick compound (compound head data)."""
    mask = (frame["label_pit"] == 1) & frame["label_compound"].isin(SLICK_COMPOUNDS)
    return cast(pd.DataFrame, frame[mask].copy())


def _load_tuned_params() -> dict:
    """Load Optuna-tuned pit-head params if present (see `just tune`), else {}."""
    return json.loads(TUNED_PARAMS_PATH.read_text()) if TUNED_PARAMS_PATH.exists() else {}


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
    # The pit head predicts the *window* target (see PIT_HORIZON).
    y_train = train_df["label_pit_window"].astype(int)
    y_val = val_df["label_pit_window"].astype(int)
    y_test = test_df["label_pit_window"].astype(int)

    # scale_pos_weight = negatives / positives on the training split.
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "seed": SEED,
                "pit_horizon": PIT_HORIZON,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "train_pit_rate": round(pos / len(train_df), 5),
                "scale_pos_weight": round(scale_pos_weight, 3),
            }
        )

        tuned = _load_tuned_params()
        mlflow.log_params({f"tuned_{k}": v for k, v in tuned.items()})

        # --- Trustworthy estimate: grouped CV across all races (vs baselines) ---
        cv_df = pd.concat([train_df, val_df, test_df])
        logger.info("Grouped CV across all races ...")
        cv = cross_validate_pit(cv_df, n_splits=5, params=tuned)
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv["agg"].items()})
        cv_comp = cross_validate_compound(cv_df, n_splits=5)
        mlflow.log_metrics({f"cv_compound_{k}": v for k, v in cv_comp.items()})

        # --- Pit head: fit on train, calibrate on val (deployment-style holdout) ---
        logger.info("Fitting pit head (scale_pos_weight=%.2f) ...", scale_pos_weight)
        pit_pipeline = build_pit_pipeline(scale_pos_weight=scale_pos_weight, **tuned)
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
        comp_y_train = cast(pd.Series, comp_train["label_compound"].astype(str))
        comp_y_test = cast(pd.Series, comp_test["label_compound"].astype(str))
        compound_pipeline = build_compound_pipeline()
        compound_pipeline.fit(comp_train, comp_y_train)
        comp_pred = compound_pipeline.predict(comp_test)
        compound_acc = float((comp_pred == comp_y_test).mean())
        # Baseline: always guess the most frequent training compound.
        comp_base = MostFrequentCompound().fit(comp_y_train)
        compound_base_acc = float((comp_base.predict(comp_test) == comp_y_test).mean())

        # --- Log metrics ---
        metrics = {
            **{f"pit_{k}": v for k, v in pit_test.items()},
            **{f"cv_{k}": v for k, v in cv["agg"].items()},
            **baseline_costs,
            **{f"cv_compound_{k}": v for k, v in cv_comp.items()},
            "compound_accuracy": compound_acc,
            "compound_baseline_accuracy": compound_base_acc,
            "compound_n_test": float(len(comp_test)),
        }
        mlflow.log_metrics(metrics)

        # --- Plots ---
        pit_pred_test = (proba_test >= threshold).astype(int)
        rel_path = ARTIFACTS_DIR / "pit_calibration.png"
        pit_cm_path = ARTIFACTS_DIR / "pit_confusion.png"
        comp_cm_path = ARTIFACTS_DIR / "compound_confusion.png"
        imp_path = ARTIFACTS_DIR / "pit_importance.png"
        plot_reliability(y_test, proba_test, rel_path)
        plot_confusion(y_test, pit_pred_test, [0, 1], pit_cm_path, "Pit decision (test)")
        plot_confusion(
            comp_y_test, comp_pred, list(SLICK_COMPOUNDS), comp_cm_path, "Compound choice (test)"
        )
        top_features = plot_feature_importance(pit_pipeline, imp_path)
        logger.info("Top pit features: %s", ", ".join(n for n, _ in top_features[:6]))
        for plot in (rel_path, pit_cm_path, comp_cm_path, imp_path):
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

    holdout_season = int(test_df["year"].max())
    _log_summary(metrics, threshold, holdout_season)
    return metrics


def _log_summary(metrics: dict, threshold: float, holdout_season: int) -> None:
    ap = metrics["pit_average_precision"]
    auc = metrics["pit_roc_auc"]
    print("\n=== Tire-strategy model results ===")
    print("Grouped CV across all races (trustworthy estimate, lower cost is better):")
    print(
        f"            model={metrics['cv_model_cost_mean']:.4f}"
        f"+/-{metrics['cv_model_cost_std']:.4f}"
        f"   majority={metrics['cv_majority_cost_mean']:.4f}"
        f"   rule={metrics['cv_rule_cost_mean']:.4f}   AP={metrics['cv_ap_mean']:.3f}"
    )
    beats = metrics["cv_model_cost_mean"] < min(
        metrics["cv_majority_cost_mean"], metrics["cv_rule_cost_mean"]
    )
    print(f"            -> beats both baselines: {beats}")
    print(f"Temporal holdout (late {holdout_season}): pit AP={ap:.3f}  ROC-AUC={auc:.3f}")
    print(
        f"            recall={metrics['pit_recall']:.3f}"
        f"  precision={metrics['pit_precision']:.3f}  (threshold={threshold:.3f})"
    )
    print(
        f"Compound head (grouped CV): acc={metrics['cv_compound_accuracy_mean']:.3f}"
        f"  macroF1={metrics['cv_compound_macro_f1_mean']:.3f}"
        f"  (baseline {metrics['cv_compound_baseline_accuracy_mean']:.3f})"
    )


if __name__ == "__main__":
    train()
