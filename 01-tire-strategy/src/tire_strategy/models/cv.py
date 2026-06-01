"""Grouped cross-validation and hyperparameter tuning for the pit head.

A single temporal test slice is noisy (few races). Grouping folds by race weekend
and averaging across the whole calendar gives a trustworthy estimate of whether
the model genuinely beats the baselines — and how much that estimate wobbles
(the fold std). Tuning then optimises the *strategic cost* directly, not accuracy.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from tire_strategy.config import SEED
from tire_strategy.models.baselines import MajorityBaseline, RuleBasedBaseline
from tire_strategy.models.metrics import (
    asymmetric_pit_cost,
    best_cost_threshold,
    expected_pit_cost,
)
from tire_strategy.models.pipeline import DEFAULT_FEATURES, FeatureSpec, build_pit_pipeline
from tire_strategy.models.split import race_group_key

_TARGET = "label_pit_window"


def _scale_pos_weight(y: np.ndarray) -> float:
    return float((y == 0).sum()) / max(int((y == 1).sum()), 1)


def cross_validate_pit(
    df: pd.DataFrame,
    target: str = _TARGET,
    n_splits: int = 5,
    params: dict | None = None,
    features: FeatureSpec = DEFAULT_FEATURES,
) -> dict:
    """GroupKFold (by race weekend) evaluation of the pit head vs the baselines.

    The decision threshold is tuned on each fold's *training* portion and applied
    to its held-out portion, so no test information leaks into the threshold.
    """
    groups = race_group_key(df).to_numpy()
    y = df[target].astype(int).to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    rows: list[dict] = []
    for fold, (tr, te) in enumerate(gkf.split(df, y, groups)):
        x_tr, x_te = df.iloc[tr], df.iloc[te]
        y_tr, y_te = y[tr], y[te]
        model = build_pit_pipeline(
            scale_pos_weight=_scale_pos_weight(y_tr), features=features, **(params or {})
        )
        model.fit(x_tr, y_tr)

        thr, _ = best_cost_threshold(y_tr, model.predict_proba(x_tr)[:, 1])
        proba_te = model.predict_proba(x_te)[:, 1]
        y_pred = (proba_te >= thr).astype(int)
        rows.append(
            {
                "fold": fold,
                "ap": float(average_precision_score(y_te, proba_te)),
                "model_cost": asymmetric_pit_cost(y_te, y_pred),
                "majority_cost": asymmetric_pit_cost(y_te, MajorityBaseline().predict(x_te)),
                "rule_cost": asymmetric_pit_cost(y_te, RuleBasedBaseline().predict(x_te)),
                "threshold": thr,
            }
        )

    folds = pd.DataFrame(rows)
    agg = {
        f"{c}_mean": float(folds[c].mean())
        for c in ("ap", "model_cost", "majority_cost", "rule_cost")
    }
    agg["ap_std"] = float(folds["ap"].std())
    agg["model_cost_std"] = float(folds["model_cost"].std())
    return {"folds": folds, "agg": agg}


def tune_pit(
    df: pd.DataFrame,
    target: str = _TARGET,
    n_trials: int = 25,
    n_splits: int = 3,
    features: FeatureSpec = DEFAULT_FEATURES,
) -> dict:
    """Optuna search minimising the mean cross-validated asymmetric cost."""
    groups = race_group_key(df).to_numpy()
    y = df[target].astype(int).to_numpy()
    folds = list(GroupKFold(n_splits=n_splits).split(df, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        costs: list[float] = []
        for tr, te in folds:
            x_tr, x_te = df.iloc[tr], df.iloc[te]
            y_tr, y_te = y[tr], y[te]
            model = build_pit_pipeline(
                scale_pos_weight=_scale_pos_weight(y_tr), features=features, **params
            )
            model.fit(x_tr, y_tr)
            thr, _ = best_cost_threshold(y_tr, model.predict_proba(x_tr)[:, 1])
            costs.append(expected_pit_cost(y_te, model.predict_proba(x_te)[:, 1], thr))
        return float(np.mean(costs))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"best_params": cast(dict, study.best_params), "best_cost": float(study.best_value)}


def _main(argv: list[str] | None = None) -> int:
    """`just tune`: tune the pit head and save artifacts/best_params.json."""
    import argparse
    import json
    import logging

    from tire_strategy.config import TUNED_PARAMS_PATH
    from tire_strategy.data import load_dataset
    from tire_strategy.data.labels import add_pit_window_label

    parser = argparse.ArgumentParser(description="Tune the pit-head hyperparameters.")
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--splits", type=int, default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = add_pit_window_label(load_dataset())  # horizon default matches training
    df = cast(pd.DataFrame, df[df["train_eligible"].astype("boolean").fillna(False)])
    result = tune_pit(df, n_trials=args.trials, n_splits=args.splits)

    TUNED_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TUNED_PARAMS_PATH.write_text(json.dumps(result["best_params"], indent=2))
    print(f"best CV cost = {result['best_cost']:.4f}  ->  saved {TUNED_PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
