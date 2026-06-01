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
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold

from tire_strategy.config import SEED
from tire_strategy.data.features import SLICK_COMPOUNDS
from tire_strategy.models.baselines import MajorityBaseline, RuleBasedBaseline
from tire_strategy.models.metrics import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    asymmetric_pit_cost,
    best_cost_threshold,
    expected_pit_cost,
)
from tire_strategy.models.pipeline import (
    DEFAULT_FEATURES,
    FeatureSpec,
    build_compound_pipeline,
    build_pit_pipeline,
)
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
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict:
    """GroupKFold (by race weekend) evaluation of the pit head vs the baselines.

    The decision threshold is tuned on each fold's *training* portion and applied
    to its held-out portion, so no test information leaks into the threshold.
    ``cost_fn``/``cost_fp`` set the asymmetric cost (see :func:`cost_ratio_sweep`).
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

        thr, _ = best_cost_threshold(
            y_tr, model.predict_proba(x_tr)[:, 1], cost_fn=cost_fn, cost_fp=cost_fp
        )
        proba_te = model.predict_proba(x_te)[:, 1]
        y_pred = (proba_te >= thr).astype(int)
        rows.append(
            {
                "fold": fold,
                "ap": float(average_precision_score(y_te, proba_te)),
                "model_cost": asymmetric_pit_cost(y_te, y_pred, cost_fn, cost_fp),
                "majority_cost": asymmetric_pit_cost(
                    y_te, MajorityBaseline().predict(x_te), cost_fn, cost_fp
                ),
                "rule_cost": asymmetric_pit_cost(
                    y_te, RuleBasedBaseline().predict(x_te), cost_fn, cost_fp
                ),
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


def cost_ratio_sweep(
    df: pd.DataFrame,
    ratios: tuple[float, ...] = (2.0, 3.0, 5.0, 8.0, 10.0),
    n_splits: int = 5,
    params: dict | None = None,
    features: FeatureSpec = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Robustness of the model's win to the missed-pit:spurious-pit cost ratio.

    The 5:1 default is a domain guess: a spurious stop costs the pit-lane time
    (~20-25 s), while a missed pit costs the time lost on dead tyres (often more).
    Rather than defend one number, we show the model beats "never pit" across a
    sensible range of ratios (cost_fp fixed at 1, cost_fn = ratio).
    """
    rows: list[dict] = []
    for ratio in ratios:
        agg = cross_validate_pit(
            df, n_splits=n_splits, params=params, features=features, cost_fn=ratio, cost_fp=1.0
        )["agg"]
        rows.append(
            {
                "ratio": ratio,
                "model_cost": agg["model_cost_mean"],
                "majority_cost": agg["majority_cost_mean"],
                "rule_cost": agg["rule_cost_mean"],
                "beats_majority": agg["model_cost_mean"] < agg["majority_cost_mean"],
                "beats_rule": agg["model_cost_mean"] < agg["rule_cost_mean"],
            }
        )
    return pd.DataFrame(rows)


def cross_validate_compound(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """GroupKFold evaluation of the compound head vs the most-frequent baseline.

    Reports accuracy and macro-F1 (the latter matters because HARD is ~53% of
    pits, so accuracy alone hides the minority SOFT/MEDIUM classes).
    """
    pits = cast(
        pd.DataFrame, df[(df["label_pit"] == 1) & df["label_compound"].isin(SLICK_COMPOUNDS)].copy()
    )
    y = cast(pd.Series, pits["label_compound"].astype(str))
    groups = race_group_key(pits).to_numpy()

    accs: list[float] = []
    macro_f1s: list[float] = []
    baseline_accs: list[float] = []
    for tr, te in GroupKFold(n_splits=n_splits).split(pits, y, groups):
        model = build_compound_pipeline()
        model.fit(pits.iloc[tr], y.iloc[tr])
        pred = model.predict(pits.iloc[te])
        y_te = y.iloc[te]
        accs.append(float((pred == y_te).mean()))
        macro_f1s.append(float(f1_score(y_te, pred, average="macro", labels=list(SLICK_COMPOUNDS))))
        most_frequent = str(y.iloc[tr].mode().iloc[0])
        baseline_accs.append(float((y_te == most_frequent).mean()))

    return {
        "accuracy_mean": float(np.mean(accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "baseline_accuracy_mean": float(np.mean(baseline_accs)),
        "n_pits": float(len(pits)),
    }


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
