"""A persistent, human-readable experiment log for the tire-strategy model.

Every experiment we run (data-scope ablations, feature ablations, cost-ratio
sweeps, ...) is appended as one record to ``experiments/results.jsonl`` (the
machine-readable source of truth) and re-rendered into ``experiments/RESULTS.md``
(a readable comparison + full params/metrics per run). Both are committed, so the
history of what we tried — and exactly which parameters produced which numbers —
survives across sessions and is easy to mine for the next improvement.

Each record captures the full configuration (data scope, split, features, cost
ratio, hyperparameters, seed) and *every* metric we can compute, so nothing that
might matter is silently dropped.

Run an experiment from the CLI::

    uv run python -m tire_strategy.experiments season-scope

then inspect ``experiments/RESULTS.md``.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import cast

import pandas as pd

from f1_foundation.explorer_data import RAW_LAPS_PATH
from tire_strategy.config import PROJECT_ROOT, SEED
from tire_strategy.data import SLICK_COMPOUNDS, load_dataset
from tire_strategy.data.features import BOOLEAN_FEATURES, NUMERIC_FEATURES
from tire_strategy.data.labels import add_pit_window_label
from tire_strategy.evaluate import pit_metrics
from tire_strategy.models.baselines import (
    MajorityBaseline,
    MostFrequentCompound,
    RuleBasedBaseline,
)
from tire_strategy.models.cv import (
    cost_ratio_sweep,
    cross_validate_compound,
    cross_validate_pit,
)
from tire_strategy.models.metrics import (
    DEFAULT_COST_FN,
    DEFAULT_COST_FP,
    asymmetric_pit_cost,
    best_cost_threshold,
)
from tire_strategy.models.pipeline import (
    DEFAULT_FEATURES,
    FeatureSpec,
    build_compound_pipeline,
    build_pit_pipeline,
)

MODERN_ERA: tuple[int, ...] = (2022, 2023, 2024, 2025)
WET_COMPOUNDS: frozenset[str] = frozenset({"INTERMEDIATE", "WET"})

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
RESULTS_JSONL: Path = EXPERIMENTS_DIR / "results.jsonl"
RESULTS_MD: Path = EXPERIMENTS_DIR / "RESULTS.md"

PIT_HORIZON = 3  # keep in sync with tire_strategy.train.PIT_HORIZON


# --------------------------------------------------------------------------- #
# Recording                                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """One experiment result: a configuration and the metrics it produced."""

    group: str  # experiment family, e.g. "season-scope"
    label: str  # this run within the family, e.g. "train 2022-2024 -> test 2025"
    params: dict[str, object]
    metrics: dict[str, float]
    notes: str = ""
    tags: dict[str, object] = field(default_factory=dict)


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # pragma: no cover - git always present in this repo
        return "unknown"


def record_runs(runs: list[Run], timestamp: str | None = None) -> None:
    """Append runs to the JSONL log and re-render the markdown report.

    ``timestamp`` is injected for deterministic tests; production stamps UTC now.
    """
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(UTC).isoformat(timespec="seconds")
    sha = _git_sha()

    existing = _read_records()
    next_id = (max((r["id"] for r in existing), default=0)) + 1
    with RESULTS_JSONL.open("a", encoding="utf-8") as fh:
        for i, run in enumerate(runs):
            record = {
                "id": next_id + i,
                "timestamp": ts,
                "git_sha": sha,
                "group": run.group,
                "label": run.label,
                "params": run.params,
                "metrics": run.metrics,
                "notes": run.notes,
                "tags": run.tags,
            }
            fh.write(json.dumps(record, default=str) + "\n")
    render_report()
    logger.info("Recorded %d run(s) -> %s", len(runs), RESULTS_JSONL.name)


def _read_records() -> list[dict]:
    if not RESULTS_JSONL.exists():
        return []
    records = []
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# Markdown rendering                                                           #
# --------------------------------------------------------------------------- #
# Friendlier column order for the comparison tables; unknown keys are appended.
_METRIC_ORDER = [
    "n_rows",
    "n_races",
    "cv_model_cost_mean",
    "cv_model_cost_std",
    "cv_majority_cost_mean",
    "cv_rule_cost_mean",
    "cv_beats_both",
    "cv_ap_mean",
    "cv_compound_accuracy_mean",
    "cv_compound_macro_f1_mean",
    "cv_compound_baseline_accuracy_mean",
    "pit_average_precision",
    "pit_roc_auc",
    "pit_recall",
    "pit_precision",
    "pit_asymmetric_cost",
    "pit_majority_cost",
    "pit_rule_cost",
    "compound_accuracy",
    "compound_baseline_accuracy",
]


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:,.0f}"
    return str(value)


def _ordered_metric_keys(metric_dicts: list[dict[str, float]]) -> list[str]:
    keys = {k for d in metric_dicts for k in d}
    ordered = [k for k in _METRIC_ORDER if k in keys]
    ordered += sorted(k for k in keys if k not in _METRIC_ORDER)
    return ordered


def render_report(path: Path | None = None) -> None:
    """Render all recorded runs into a readable markdown report."""
    path = path if path is not None else RESULTS_MD
    records = _read_records()
    lines: list[str] = [
        "# Experiment results",
        "",
        "Auto-generated from `results.jsonl` by `tire_strategy.experiments` — "
        "do not edit by hand. Each experiment family shows a comparison table "
        "(key metrics) followed by the full parameters and metrics for every run.",
        "",
        "**Reading the pit metrics:** lower `*_cost` is better (asymmetric strategic "
        "cost, missed pit = 5x a spurious one); `cv_*` come from grouped 5-fold CV by "
        "race weekend (the trustworthy estimate); `pit_*` come from a temporal "
        "train-then-test holdout. Baselines are *never-pit* (majority) and a simple "
        "rule-based strategist.",
        "",
    ]

    # Group records preserving first-seen order.
    groups: dict[str, list[dict]] = {}
    for rec in records:
        groups.setdefault(rec["group"], []).append(rec)

    if not records:
        lines.append("_No experiments recorded yet._")

    for group, recs in groups.items():
        latest_ts = max(r["timestamp"] for r in recs)
        lines.append(f"## {group}")
        lines.append("")
        notes = next((r["notes"] for r in reversed(recs) if r["notes"]), "")
        if notes:
            lines.append(f"> {notes}")
            lines.append("")
        lines.append(f"_Last run: {latest_ts} · {len(recs)} run(s)._")
        lines.append("")

        metric_keys = _ordered_metric_keys([r["metrics"] for r in recs])
        header = ["run", "label", *metric_keys]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for r in recs:
            row = [f"#{r['id']}", r["label"]]
            row += [_fmt(r["metrics"].get(k, "")) for k in metric_keys]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        # Full params + metrics per run (for deep dives / reproduction).
        lines.append("<details><summary>Full parameters &amp; metrics</summary>")
        lines.append("")
        for r in recs:
            lines.append(f"**#{r['id']} — {r['label']}**  ")
            lines.append(f"`git {r['git_sha']}` · {r['timestamp']}")
            lines.append("")
            lines.append("| parameter | value | | metric | value |")
            lines.append("| --- | --- | --- | --- | --- |")
            pkeys = sorted(r["params"])
            mkeys = list(r["metrics"])
            for i in range(max(len(pkeys), len(mkeys))):
                pk = pkeys[i] if i < len(pkeys) else ""
                pv = _fmt(r["params"][pk]) if pk else ""
                mk = mkeys[i] if i < len(mkeys) else ""
                mv = _fmt(r["metrics"][mk]) if mk else ""
                lines.append(f"| {pk} | {pv} | | {mk} | {mv} |")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reusable evaluation building blocks                                         #
# --------------------------------------------------------------------------- #
def _prepare(df: pd.DataFrame, horizon: int = PIT_HORIZON) -> pd.DataFrame:
    """Add the pit-window label and keep only training-eligible rows."""
    labelled = add_pit_window_label(df, horizon=horizon)
    eligible = labelled["train_eligible"].astype("boolean").fillna(False)
    return cast(pd.DataFrame, labelled[eligible].copy())


def _scope(df: pd.DataFrame, years: tuple[int, ...]) -> pd.DataFrame:
    return cast(pd.DataFrame, df[df["year"].isin(list(years))].copy())


def _slick_pits(frame: pd.DataFrame) -> pd.DataFrame:
    mask = (frame["label_pit"] == 1) & frame["label_compound"].isin(SLICK_COMPOUNDS)
    return cast(pd.DataFrame, frame[mask].copy())


def grouped_cv_metrics(
    df: pd.DataFrame,
    params: dict | None = None,
    features: FeatureSpec = DEFAULT_FEATURES,
    n_splits: int = 5,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict[str, float]:
    """Flat metric dict from grouped CV of both heads on ``df`` (already prepared)."""
    pit = cross_validate_pit(
        df, n_splits=n_splits, params=params, features=features, cost_fn=cost_fn, cost_fp=cost_fp
    )["agg"]
    comp = cross_validate_compound(df, n_splits=n_splits, features=features)
    beats_both = pit["model_cost_mean"] < min(pit["majority_cost_mean"], pit["rule_cost_mean"])
    metrics = {
        "n_rows": float(len(df)),
        "n_races": float(df.groupby(["year", "round"]).ngroups),
        "cv_beats_both": bool(beats_both),
        **{f"cv_{k}": v for k, v in pit.items()},
        **{f"cv_compound_{k}": v for k, v in comp.items()},
    }
    return metrics


def temporal_holdout_metrics(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: dict | None = None,
    features: FeatureSpec = DEFAULT_FEATURES,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> dict[str, float]:
    """Train→test metrics: fit on ``train_df``, evaluate on the future ``test_df``.

    The decision threshold is chosen on the *training* portion (no test leakage),
    matching the grouped-CV convention. Both model heads plus baselines are scored.
    """
    y_train = train_df["label_pit_window"].astype(int).to_numpy()
    y_test = test_df["label_pit_window"].astype(int).to_numpy()
    pos, neg = int(y_train.sum()), int((y_train == 0).sum())
    spw = neg / max(pos, 1)

    pit = build_pit_pipeline(scale_pos_weight=spw, features=features, **(params or {}))
    pit.fit(train_df, y_train)
    thr, _ = best_cost_threshold(
        y_train, pit.predict_proba(train_df)[:, 1], cost_fn=cost_fn, cost_fp=cost_fp
    )
    proba_test = pit.predict_proba(test_df)[:, 1]
    pm = pit_metrics(y_test, proba_test, thr, cost_fn=cost_fn, cost_fp=cost_fp)

    maj = asymmetric_pit_cost(y_test, MajorityBaseline().predict(test_df), cost_fn, cost_fp)
    rule = asymmetric_pit_cost(y_test, RuleBasedBaseline().predict(test_df), cost_fn, cost_fp)

    comp_train, comp_test = _slick_pits(train_df), _slick_pits(test_df)
    comp_metrics: dict[str, float] = {}
    if len(comp_train) and len(comp_test):
        cy_train = cast(pd.Series, comp_train["label_compound"].astype(str))
        cy_test = cast(pd.Series, comp_test["label_compound"].astype(str))
        comp = build_compound_pipeline(features=features)
        comp.fit(comp_train, cy_train)
        comp_acc = float((comp.predict(comp_test) == cy_test).to_numpy().mean())
        base = MostFrequentCompound().fit(cy_train)
        base_acc = float((base.predict(comp_test) == cy_test).to_numpy().mean())
        comp_metrics = {
            "compound_accuracy": comp_acc,
            "compound_baseline_accuracy": base_acc,
            "compound_n_test": float(len(comp_test)),
        }

    return {
        "n_rows": float(len(train_df)),
        "n_races": float(train_df.groupby(["year", "round"]).ngroups),
        "n_test_rows": float(len(test_df)),
        **{f"pit_{k}": v for k, v in pm.items()},
        "pit_majority_cost": maj,
        "pit_rule_cost": rule,
        **comp_metrics,
    }


# --------------------------------------------------------------------------- #
# Experiments                                                                  #
# --------------------------------------------------------------------------- #
def season_scope() -> list[Run]:
    """Does adding the 2025 season (and older history) to training help?

    Two complementary views, because 2025 is the newest season and so there is no
    *future* hold-out to test a 2025-trained model on directly:

    1. Grouped-CV pool ablation — run the project's trustworthy 5-fold grouped CV
       on different year scopes. Adding 2025 enlarges the pool; we check the
       model-vs-baseline cost and AP hold up or improve.
    2. Forward-chaining temporal holdout — fix the test set to the newest full
       season (2025) and grow the training window. If more/recent training data
       improves prediction of the newest season, that is the evidence that adding
       2025 to training will help the *next* (2026) season.
    """
    raw = _prepare(load_dataset())
    runs: list[Run] = []

    note = (
        "Adding 2025 to training can only be judged indirectly (no season exists "
        "after it): grouped CV pools it in; forward-chaining tests whether each "
        "added season improves next-season prediction."
    )

    # --- View 1: grouped-CV pool ablation ---
    cv_scopes = {
        "CV pool 2022-2024 (no 2025)": (2022, 2023, 2024),
        "CV pool 2022-2025 (+2025)": (2022, 2023, 2024, 2025),
        "CV pool 2018-2025 (all years)": (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025),
    }
    for label, years in cv_scopes.items():
        df = _scope(raw, years)
        logger.info("season-scope CV: %s (%d rows)", label, len(df))
        runs.append(
            Run(
                group="season-scope",
                label=label,
                params={"view": "grouped-cv", "years": list(years), "n_splits": 5, "seed": SEED},
                metrics=grouped_cv_metrics(df),
                notes=note,
            )
        )

    # --- View 2: forward-chaining temporal holdout (fixed test = 2025) ---
    test_2025 = _scope(raw, (2025,))
    train_scopes = {
        "train 2022-2023 -> test 2025": (2022, 2023),
        "train 2022-2024 -> test 2025": (2022, 2023, 2024),
        "train 2018-2024 -> test 2025": (2018, 2019, 2020, 2021, 2022, 2023, 2024),
    }
    for label, years in train_scopes.items():
        train_df = _scope(raw, years)
        logger.info("season-scope temporal: %s", label)
        runs.append(
            Run(
                group="season-scope",
                label=label,
                params={
                    "view": "temporal-holdout",
                    "train_years": list(years),
                    "test_years": [2025],
                    "seed": SEED,
                },
                metrics=temporal_holdout_metrics(train_df, test_2025),
                notes=note,
            )
        )

    # Also show the *current production split* for reference (train 2022-23 -> 2024).
    runs.append(
        Run(
            group="season-scope",
            label="train 2022-2023 -> test 2024 (current prod split)",
            params={
                "view": "temporal-holdout",
                "train_years": [2022, 2023],
                "test_years": [2024],
                "seed": SEED,
            },
            metrics=temporal_holdout_metrics(_scope(raw, (2022, 2023)), _scope(raw, (2024,))),
            notes=note,
        )
    )
    return runs


def cost_ratio() -> list[Run]:
    """How sensitive is the model's win to the missed-pit:spurious-pit cost ratio?

    The 5:1 default is a domain guess. This sweeps the ratio and records, at each,
    the model vs both baselines on grouped CV — so the headline "model wins" is
    shown to be robust to the exact cost number, not an artefact of one choice.
    """
    pool = _scope(_prepare(load_dataset()), MODERN_ERA)
    sweep = cost_ratio_sweep(pool, ratios=(2.0, 3.0, 5.0, 8.0, 10.0))
    note = (
        "Model beats 'never pit' and the rule-based strategist across every "
        "missed:spurious cost ratio from 2:1 to 10:1 (grouped CV, 2022-2025)."
    )
    runs: list[Run] = []
    for row in sweep.to_dict("records"):
        ratio = float(row["ratio"])
        runs.append(
            Run(
                group="cost-ratio",
                label=f"cost ratio {ratio:.0f}:1 (missed:spurious)",
                params={"cost_fn": ratio, "cost_fp": 1.0, "n_splits": 5, "seed": SEED},
                metrics={
                    "model_cost": float(row["model_cost"]),
                    "majority_cost": float(row["majority_cost"]),
                    "rule_cost": float(row["rule_cost"]),
                    "beats_majority": bool(row["beats_majority"]),
                    "beats_rule": bool(row["beats_rule"]),
                },
                notes=note,
            )
        )
    return runs


@lru_cache(maxsize=1)
def _team_lookup() -> pd.DataFrame:
    """(year, round, driver) -> team, from the raw-laps parquet (1 team per race)."""
    raw = pd.read_parquet(RAW_LAPS_PATH, columns=["year", "round", "Driver", "Team"])
    teams = raw.drop_duplicates(subset=["year", "round", "Driver"])
    return cast(pd.DataFrame, teams.rename(columns={"Driver": "driver", "Team": "team"}))


def _with_team(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the constructor (`team`) to each driver-lap via a leak-free race join."""
    merged = df.merge(_team_lookup(), on=["year", "round", "driver"], how="left")
    merged["team"] = merged["team"].fillna("UNKNOWN").astype(str)
    return merged


def team_feature() -> list[Run]:
    """Do teams follow learnable strategy patterns? Add `team` and ablate.

    The constructor is a weak-but-plausible signal (some teams habitually undercut,
    run longer first stints, etc.), but it is confounded with car pace and shifts
    year to year. Rather than assume, we add `team` as a categorical to *both* heads
    and compare on grouped CV (2022-2025). Keep it only if it consistently helps.
    """
    pool = _scope(_prepare(load_dataset()), MODERN_ERA)
    teamed = _with_team(pool)
    spec_team = FeatureSpec(categorical=("compound", "circuit", "team"))
    note = (
        "Adds the constructor as a categorical to both heads. Compare cv_model_cost "
        "/ cv_ap_mean / cv_compound_accuracy_mean against the no-team baseline; adopt "
        "only on a consistent win (the project's measure-it-or-reject rule)."
    )
    return [
        Run(
            group="team-feature",
            label="baseline (no team)",
            params={"features": list(DEFAULT_FEATURES.columns), "n_splits": 5, "seed": SEED},
            metrics=grouped_cv_metrics(pool),
            notes=note,
        ),
        Run(
            group="team-feature",
            label="+ team (constructor)",
            params={"features": list(spec_team.columns), "n_splits": 5, "seed": SEED},
            metrics=grouped_cv_metrics(teamed, features=spec_team),
            notes=note,
        ),
    ]


def wet_regime() -> list[Run]:
    """Feasibility of a wet<->dry tyre-regime model (scoping, not a fitted model).

    The slick compound head excludes INTERMEDIATE/WET. A wet model is a *separate*
    build; before committing to it, quantify the opportunity honestly: how often
    wet tyres are used, and how many slick<->wet crossovers there are to learn from.
    No invented accuracy — just the data that decides whether it is worth building.
    """
    raw = pd.read_parquet(
        RAW_LAPS_PATH, columns=["year", "round", "Driver", "Compound", "LapNumber"]
    )
    raw = cast(pd.DataFrame, raw[raw["year"].isin(list(MODERN_ERA))].copy())

    def regime(compound: str) -> str:
        if compound in WET_COMPOUNDS:
            return "wet"
        if compound in SLICK_COMPOUNDS:
            return "slick"
        return "other"

    raw["regime"] = raw["Compound"].astype(str).map(regime)
    n_races = int(raw.groupby(["year", "round"]).ngroups)
    wet_laps = int((raw["regime"] == "wet").sum())
    races_with_wet = int(
        raw[raw["regime"] == "wet"].groupby(["year", "round"]).ngroups  # type: ignore[index]
    )

    slick_to_wet = wet_to_slick = 0
    driver_races_with_wet = 0
    for _, grp in raw.sort_values("LapNumber").groupby(["year", "round", "Driver"]):
        regimes = [r for r in grp["regime"].tolist() if r in ("slick", "wet")]
        if "wet" in regimes:
            driver_races_with_wet += 1
        for prev, curr in pairwise(regimes):
            if prev == "slick" and curr == "wet":
                slick_to_wet += 1
            elif prev == "wet" and curr == "slick":
                wet_to_slick += 1

    note = (
        "Scoping only. A wet-regime model is a separate build from the slick head; "
        "these counts decide whether the (thin) data supports it. The slick<->wet "
        "crossovers are the learnable transitions; rainfall is already a per-lap "
        "feature, so onset is observable but track-drying timing is not."
    )
    return [
        Run(
            group="wet-regime",
            label="2022-2025 wet-tyre usage & crossovers",
            params={"years": list(MODERN_ERA), "wet_compounds": sorted(WET_COMPOUNDS)},
            metrics={
                "n_races": float(n_races),
                "races_with_wet_tyres": float(races_with_wet),
                "pct_races_with_wet": float(round(100 * races_with_wet / max(n_races, 1), 1)),
                "wet_laps": float(wet_laps),
                "pct_wet_laps": float(round(100 * wet_laps / max(len(raw), 1), 2)),
                "driver_races_using_wet": float(driver_races_with_wet),
                "slick_to_wet_crossovers": float(slick_to_wet),
                "wet_to_slick_crossovers": float(wet_to_slick),
            },
            notes=note,
        )
    ]


RAIN_NUMERIC = ("laps_in_rain_state",)
RAIN_BOOLEAN = ("rain_just_started", "rain_just_stopped")


def _with_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add leak-safe rain-*transition* features from the per-lap `rainfall` flag.

    `rainfall` is a condition observed at the start of the lap (not a timing-derived
    value), so comparing it to the previous lap is leak-free — exactly like
    `sc_just_deployed`. The transition is the real trigger: a dry->wet switch is when
    teams box for intermediates; wet->dry is the crossover back to slicks.
    """
    out = df.sort_values(["year", "round", "driver", "lap_number"]).copy()
    grp = out.groupby(["year", "round", "driver"])
    rain = cast(pd.Series, out["rainfall"]).fillna(False).astype(bool)
    prev = cast(pd.Series, grp["rainfall"].shift(1)).fillna(False).astype(bool)
    out["rain_just_started"] = (rain & ~prev).astype("boolean")
    out["rain_just_stopped"] = (~rain & prev).astype("boolean")
    out["_change"] = (rain.to_numpy() != prev.to_numpy()).astype(int)
    spell = out.groupby(["year", "round", "driver"])["_change"].cumsum()
    out["laps_in_rain_state"] = (
        out.assign(_spell=spell).groupby(["year", "round", "driver", "_spell"]).cumcount()
    )
    return cast(pd.DataFrame, out.drop(columns="_change"))


def rain_onset() -> list[Run]:
    """Add leak-safe rain-transition features and ablate (if it helps it helps,
    if not it does no harm — we only adopt on a measured CV win).

    Dry→rain onset is the obvious, learnable trigger (box for inters); we encode it
    as `rain_just_started` / `rain_just_stopped` plus `laps_in_rain_state` (how long
    the current wet/dry spell has lasted) and compare to the no-rain-feature baseline
    on grouped CV across 2022-2025.
    """
    pool = _scope(_prepare(load_dataset()), MODERN_ERA)
    rained = _with_rain_features(pool)
    spec_rain = FeatureSpec(
        numeric=tuple(NUMERIC_FEATURES) + RAIN_NUMERIC,
        boolean=tuple(BOOLEAN_FEATURES) + RAIN_BOOLEAN,
    )
    note = (
        "Leak-safe rain-transition features (dry->wet / wet->dry / laps-in-spell). "
        "Adopt only if grouped-CV cost/AP improve or hold (no harm); rain is 13% of "
        "races so the overall move is expected to be small but should not be negative."
    )
    return [
        Run(
            group="rain-onset",
            label="baseline (rainfall only)",
            params={"features": list(DEFAULT_FEATURES.columns), "n_splits": 5, "seed": SEED},
            metrics=grouped_cv_metrics(pool),
            notes=note,
        ),
        Run(
            group="rain-onset",
            label="+ rain onset/stop/spell-age",
            params={"features": list(spec_rain.columns), "n_splits": 5, "seed": SEED},
            metrics=grouped_cv_metrics(rained, features=spec_rain),
            notes=note,
        ),
    ]


EXPERIMENTS = {
    "season-scope": season_scope,
    "cost-ratio": cost_ratio,
    "team-feature": team_feature,
    "wet-regime": wet_regime,
    "rain-onset": rain_onset,
}


def _main(argv: list[str] | None = None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run a named experiment and log its results.")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS), help="experiment to run")
    args = parser.parse_args(argv)

    runs = EXPERIMENTS[args.experiment]()
    record_runs(runs)
    print(f"\nLogged {len(runs)} run(s) for '{args.experiment}'.")
    print(f"See {RESULTS_MD.relative_to(PROJECT_ROOT)} for the full report.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
