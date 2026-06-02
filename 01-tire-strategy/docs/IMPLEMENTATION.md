# Project 1 — Tire Strategy Prediction: Complete Implementation Document

This document explains **everything** about Project 1: what it does, how every file
works, why each decision was made, the alternatives considered, every experiment
run (including the ones we discarded), and the precise meaning of every metric.

It is written to be honest. Where a number was **measured**, it is given. Where a
change was adopted on **principle without an isolated A/B**, that is stated. Where
something is genuinely **unknown** (e.g. the theoretical best achievable score), it
is said so rather than invented.

> Companion docs: [`DESIGN.md`](DESIGN.md) is the up-front data/labelling contract.
> This file is the full retrospective.

---

## 1. What the project is

**Goal:** given the live state of an F1 car at the start of a lap, predict the
tyre-strategy decision a race engineer would make:

1. **Pit head** — will this car pit soon? (a calibrated probability)
2. **Compound head** — if it pits, onto which slick compound (SOFT / MEDIUM / HARD)?

It is built entirely on free, public F1 timing data via the **FastF1** library, on
**2022–2025** (the current ground-effect regulation era). Adding 2025 to training
was a measured improvement, not an assumption — see the season-scope study (§12 #15)
and the committed `experiments/RESULTS.md`. The headline result:

| Metric (5-fold grouped CV across all 92 races) | Model | "Never pit" | Rule-based |
|---|---|---|---|
| Asymmetric strategic cost (lower better) | **0.312 ± 0.010** | 0.457 | 0.441 |
| Average precision (pit head) | **0.354** | — | — |
| Compound accuracy / macro-F1 | **0.659 / 0.626** | 0.505 (most-frequent) | — |

The model beats both baselines by ~29–32% on cost, and the win holds across every
cost ratio from 2:1 to 10:1 (see §11). The production split trains on 2022–2024 and
holds out 2025 (late-2025 temporal holdout: ROC-AUC 0.83); folding 2025 into the
training pool also cut the CV fold std ~4× (0.039 → 0.010), a much more stable
estimate.

---

## 2. The core modelling decisions (and the alternatives)

### 2.1 Decomposed target, not a flat 4-class
**Decision:** two separate models — a binary pit-timing head and a conditional
compound head — instead of one 4-class `{stay, pit-soft, pit-med, pit-hard}`.

**Why:** a driver pits ~1–3 times in a ~55-lap race, so the positive ("pit") class
is ~3% of laps. A single 4-class model maximises accuracy by always predicting
"stay out" (~97% accurate, 0% useful). Splitting lets us (a) calibrate a single pit
probability and (b) train the compound model only on the balanced pit-lap subset.

**Alternative considered:** flat 4-class. Rejected — severe imbalance, muddier
calibration, no real benefit. We did not even build it as a baseline because the
informative baselines are "never pit" and a rule-based strategy (§9.3).

### 2.2 Imitation labels, not "optimal" labels
**Decision:** the label is **what the team actually did** ("predict the strategist's
decision").

**Why:** imitation labels are objective and unarguable. An "optimal" label would
require a race simulator whose invented assumptions (tyre fade, pit loss, traffic)
we would then be grading ourselves against — circular reasoning.

**Alternative considered:** optimal-via-simulation. Deferred to a future
"Project 1.5" (a counterfactual race simulator), kept separate so its assumptions
never contaminate this honest core. This is the known **imitation ceiling**: the
model inherits teams' real-world mistakes (§13).

### 2.3 "Pit within K laps", not "pit on exactly this lap"
**Decision:** the pit target is `label_pit_window` = 1 if a strategic pit happens on
this lap **or within the next K−1 laps** (K=3).

**Why:** the exact-lap target was the single biggest problem. A strategy call has a
*window* — boxing on lap 22 vs 24 is essentially the same decision. The exact-lap
target (a) made positives vanishingly rare and (b) punished the model for being
"off by one" when it was essentially right. **Measured effect** (temporal holdout):
average precision 0.10 → 0.17, recall 0.17 → 0.70.

This target is **forward-looking by design** — but it is the *target*, not a
feature, so it does not leak. Features remain strictly leak-safe (§7.4).

---

## 3. Requirements and tech stack

- **Python 3.11** (pinned via `.python-version`).
- **uv** for environment + dependency management (single workspace, one `uv.lock`).
- **LightGBM needs the OpenMP runtime** on macOS: `brew install libomp`. Without it,
  `import lightgbm` fails with a `dlopen` error.
- Runtime deps (declared in `01-tire-strategy/pyproject.toml`): `fastf1`, `pandas`,
  `numpy`, `scikit-learn`, `lightgbm`, `matplotlib`, `mlflow`, `pyarrow`, `optuna`.
- Dev deps (root `pyproject.toml`): `ruff` (lint+format), `pyright` (types),
  `pytest`, `pre-commit`, `dvc`, `mlflow`.
- **MLflow** uses a **SQLite backend** (`mlflow/mlflow.db`); the legacy file store is
  deprecated in current MLflow and raises unless you opt out.

**Commands** (`just`):
`just install` · `just data` (build dataset) · `just tune` (Optuna) ·
`just train` · `just test` · `just check` (lint+types+tests) · `just mlflow-ui`.

---

## 4. Repository layout and what each file does

```
01-tire-strategy/
├── configs/best_params.json     # tracked, Optuna-tuned pit-head hyperparameters
├── data/                        # gitignored: FastF1 cache + built parquet
├── docs/                        # DESIGN.md, this file
├── src/tire_strategy/
│   ├── config.py                # paths, SEED, FastF1 cache helper
│   ├── data/
│   │   ├── ingest.py            # FastF1 -> clean per-lap table + results
│   │   ├── labels.py            # strategy labels + pit-window + exclusions
│   │   ├── features.py          # leak-safe feature engineering
│   │   └── dataset.py           # batch all races -> one parquet
│   ├── models/
│   │   ├── split.py             # temporal + grouped-CV splitting
│   │   ├── metrics.py           # asymmetric cost + threshold search
│   │   ├── baselines.py         # majority, rule-based, most-frequent-compound
│   │   ├── pipeline.py          # sklearn Pipelines (FeatureSpec + LightGBM)
│   │   └── cv.py                # grouped CV, Optuna tuning, cost sweep
│   ├── evaluate.py              # metrics + diagnostic plots
│   └── train.py                 # orchestrates train + eval + MLflow + artifacts
└── tests/                       # 29 tests (pure + integration)
```

### 4.1 `config.py`
Project paths (`PROJECT_ROOT`, `DATA_DIR`, `CACHE_DIR`, `ARTIFACTS_DIR`,
`MLFLOW_DIR`, `CONFIGS_DIR`, `TUNED_PARAMS_PATH`), the global `SEED = 42`, and
`enable_cache()` which turns on the mandatory FastF1 on-disk cache (FastF1 hits the
live F1 timing API; caching avoids rate limits and makes re-runs instant).

### 4.2 `data/ingest.py`
- `load_race(year, gp)` → one clean row per driver-lap. It loads the session
  (`laps=True, weather=True, telemetry=False`), then produces typed columns:
  identifiers (year, round, circuit, driver, team), lap/stint/tyre fields, timing
  converted from timedeltas to **float seconds**, weather (air/track temp, rainfall)
  aligned to each lap by FastF1, **per-lap track-status flags** (SC/VSC/yellow/red),
  pit in/out flags, and data-quality flags (`is_accurate`, `deleted`, `fresh_tyre`).
- `load_results(year, gp)` → per-driver classification, used to detect DNFs honestly
  (a driver is *classified* only if `ClassifiedPosition` is numeric).
- `retired_drivers(results)` → the set of non-classified driver codes.
- Track status is encoded by FastF1 as a string of concatenated codes per lap
  (e.g. `"14"` = codes 1 and 4 both occurred); `_contains(code)` tests membership.

### 4.3 `data/labels.py`
- `add_pit_and_compound_labels(df)` →
  - `label_pit` = 1 on a **strategic** in-lap (an in-lap that is **not** a free
    red-flag tyre change), else 0.
  - `label_compound` = the compound fitted at the stop (read from the **next** lap,
    the out-lap on the new tyre), defined only on pit laps.
- `add_train_eligibility(df, retired)` → `train_eligible` + `exclude_reason`,
  flagging out **red-flag changes** (free, rule-driven, not strategic) and a
  **retiree's final lap** (the car stopped; it was never a "stay out" decision).
- `add_pit_window_label(df, horizon=3)` → the `label_pit_window` target (§2.3),
  grouped by race+driver so it never spills across races.
- `build_labels`, `load_labelled_race` compose the above.

### 4.4 `data/features.py`
Builds the 19 model features (§7) from a labelled race, leak-safe by construction.
Key internals: `_clean_lap_mask` (which laps have representative pace),
`_add_stint_features` (stint counters, mandatory-compound flag),
`_add_gap_features` (reconstruct gaps from line-crossing order, lag one lap),
`_add_pace_features` (last clean lap, **field-relative pace**, **fuel-corrected
degradation slope**). Exposes `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`,
`BOOLEAN_FEATURES`, `FEATURE_COLUMNS`, `SLICK_COMPOUNDS`, `FUEL_S_PER_LAP`.

### 4.5 `data/dataset.py`
- `iter_calendar(seasons)` → `(year, round)` for every race.
- `build_one(year, race)` → that race's feature table, or `None` if it fails.
- `assemble(races, out_path)` → concatenate all, **logging and counting any skipped
  race** (silent gaps would misrepresent coverage), write parquet.
- `build_dataset`, `load_dataset`, and a CLI (`python -m tire_strategy.data.dataset`).

### 4.6 `models/split.py`
- `assign_temporal_split` → `train`/`val`/`test` by **race weekend** (atomic): train
  2022–23, validate early-2024, test late-2024. Mimics deployment (predict future
  races from past).
- `race_group_key` → a `"<year>_<round>"` key for grouped cross-validation.

### 4.7 `models/metrics.py`
- `pit_cost_matrix`, `asymmetric_pit_cost` (the custom metric, §10.5),
  `expected_pit_cost` (cost from probabilities + threshold), `best_cost_threshold`
  (grid-search the threshold minimising expected cost). Defaults `cost_fn=5`,
  `cost_fp=1`.

### 4.8 `models/baselines.py`
- `MajorityBaseline` — always "stay out".
- `RuleBasedBaseline` — pit when tyre age exceeds a compound-specific threshold
  (`SOFT 16, MEDIUM 26, HARD 36` laps) or when worn (`age ≥ 8`) under a safety car.
- `MostFrequentCompound` — always guess the most common training compound (the bar
  the compound head must clear).

### 4.9 `models/pipeline.py`
- `FeatureSpec` — a frozen dataclass naming the numeric/categorical/boolean columns,
  so feature ablations need **no pipeline edits**.
- `build_pit_pipeline` / `build_compound_pipeline` — full sklearn `Pipeline`s:
  coerce dtypes → one-hot the categoricals → LightGBM. Preprocessing lives **inside**
  the estimator, so training and serving transform identically (avoids a classic
  production skew bug). LightGBM handles NaN natively (no imputation). The pit head
  uses `scale_pos_weight`; the compound head uses `class_weight="balanced"`.

### 4.10 `models/cv.py`
- `cross_validate_pit` — GroupKFold by race; per fold, train on the train folds,
  tune the threshold on the **train** portion, evaluate on the held-out fold vs both
  baselines. Returns per-fold + aggregated cost/AP. Parameterised by `cost_fn/cost_fp`.
- `cost_ratio_sweep` — runs the above across cost ratios (robustness, §11).
- `cross_validate_compound` — grouped-CV accuracy + macro-F1 vs the most-frequent
  baseline.
- `tune_pit` — Optuna search minimising mean CV cost; `_main` saves
  `configs/best_params.json` (`just tune`).

### 4.11 `evaluate.py`
- `pit_metrics` (AP, ROC-AUC, recall, precision, asymmetric cost — **no accuracy**,
  it is meaningless here), `plot_reliability` (calibration curve),
  `plot_feature_importance` (gain-based audit), `plot_confusion`.

### 4.12 `train.py`
Orchestrates everything: temporal split → grouped-CV (the trustworthy headline) →
fit pit head → **isotonic-calibrate on validation** → cost-minimising threshold →
compound head → evaluate vs baselines → MLflow logging → save model artifacts +
plots + `metadata.json`. Loads tuned params from `configs/best_params.json` if present.

---

## 5. The data pipeline internals

**Source:** FastF1 `session.laps`, `session.weather_data`, `session.results`, and
the per-lap `TrackStatus`. The cache is mandatory and lives in `data/cache/`.

**Scope:** 2022–2024, all circuits. **Measured dataset: 74,601 laps, 68 races**
(2022: 22, 2023: 22, 2024: 24 rounds), 25 circuits. Building it cold hits FastF1's
**500-calls/hour** rate limit (the first attempt got 32/68 races before throttling;
re-running after the window reset completed all 68 — cached races make no API calls).

**Gap reconstruction:** FastF1 exposes no gap column. For each lap we order cars by
classified `Position` and take the difference of their line-crossing session `Time`
— the "gap at the line, each lap", exactly what a strategist watches. Lapped traffic
/ missing neighbours become NaN (~7% of rows). This is an **approximation**, stated
honestly (§13).

---

## 6. Labels in detail

- A pit is recorded on the **in-lap** (`PitInTime` present). `label_pit = 1` there,
  **unless** the lap is under red flag (free change).
- `label_compound` reads the **out-lap's** compound (the tyre actually fitted).
- **Exclusions** (`train_eligible = False`): red-flag changes and retirees' final
  laps. In one partial build these were ~102 red-flag + ~77 retirement rows; exact
  counts vary by build. The mechanism, not the exact count, is the point.
- **Class balance (measured):** exact-pit rate ~3.2%; the K=3 window target gives
  `scale_pos_weight ≈ 9.38` (≈9.7% positive). Slick pits total **2162**
  (HARD 1137 ≈ 53%, MEDIUM 626, SOFT 399). ~3578 laps are wet (`rainfall`).

---

## 7. Features in detail (19 inputs)

All features are the **state knowable at the start of the decision lap**. The rule:
lap-start state uses the current lap; anything derived from lap *timing* is computed
from completed laps and **lagged one lap**, because the decision lap's own time is
contaminated (most obviously on an in-lap, where it includes pit-lane time).

### 7.1 Numeric (11)
- `lap_number`, `laps_remaining` — race position in time.
- `tyre_age` (tyre life), `stint_position`, `pit_count_so_far` — stint context.
- `track_temp`, `air_temp` — conditions (hotter → faster tyre fade).
- `gap_ahead_s`, `gap_behind_s` — reconstructed gaps, lagged.
- `last_clean_lap_time_s` — most recent representative lap time (excludes in/out/SC
  laps), lagged.
- `deg_slope_s_per_lap` — **fuel-corrected** in-stint degradation slope (§7.3).
- `relative_pace_s` — the car's recent clean lap minus the **field median** that lap
  (positive = slower than rivals → a pit signal). Added in round 2; **measured to help**.

### 7.2 Categorical (2)
- `compound` (current), `circuit` (added round 2; **measured to help** — strategy is
  intensely circuit-specific: pit-loss time, abrasiveness, overtaking difficulty).

### 7.3 Boolean (7)
- `fresh_tyre`, `under_caution`, `flag_sc`, `flag_vsc`, `flag_yellow`,
  `used_second_compound` (mandatory-rule status), `rainfall` (so rain-driven stops
  are not confounding noise; added when wet compounds were found in the data).

### 7.3 Fuel-corrected degradation
Raw lap times **fall** through a stint as ~100 kg of fuel burns (~0.05 s/kg ×
~1.6 kg/lap ≈ **0.055 s/lap faster**), which *masks* tyre wear. Before fitting the
degradation slope we subtract `FUEL_S_PER_LAP × laps_remaining`, so the slope
reflects tyre wear, not a lightening car. `FUEL_S_PER_LAP = 0.055` is a deliberate
**domain constant, not a fitted value**. The slope is fitted over the last 5 clean
laps of the current stint, using only laps before the current one.

> **Honesty note:** fuel correction was adopted on physical principle as part of the
> "hardening" bundle and was **not** isolated in its own A/B. Its effect is mixed
> into the hardening gains. By contrast, `circuit`, `relative_pace_s`, `driver`, and
> the gap-trend idea **were** each independently ablated (§12).

### 7.4 Why it is leak-free
A unit test asserts that an outlier on the current lap never enters that lap's pace
features, and that on a real race Leclerc's ~90 s in-lap yields a pace feature of
~85 s (a prior green lap). The labels look forward; the features never do.

---

## 8. Splitting and evaluation methodology

- **Temporal split** (deployment-style): train 2022–23, validate early-2024, test
  late-2024. Honest but **one small, noisy slice**.
- **Grouped 5-fold CV** (by race weekend): the **trustworthy headline**. This was the
  single most important evaluation decision — see §12, where a result that looked
  like "tied with the baseline" on the temporal slice was clearly a **win** on
  grouped CV. Lesson: never trust one slice.

The threshold is always tuned on training data (per fold, or on the validation set),
never on the test data.

---

## 9. The models

### 9.1 Pit head
LightGBM binary classifier inside the sklearn pipeline. `scale_pos_weight = neg/pos`
(~9.38) handles imbalance **without resampling** (resampling would distort
calibration). It is then **isotonically calibrated on the validation set**
(`CalibratedClassifierCV(FrozenEstimator(pipeline))` — sklearn 1.6 replaced
`cv="prefit"` with `FrozenEstimator`). The decision **threshold** is chosen to
minimise the asymmetric cost (not fixed at 0.5).

**Tuned hyperparameters** (`configs/best_params.json`, found by Optuna minimising CV
cost): `n_estimators=419, learning_rate≈0.0104, num_leaves=15, min_child_samples=74,
reg_lambda≈0.209, colsample_bytree≈0.685, subsample≈0.885`. Shallow, regularised,
slow-learning trees — sensible for this signal-to-noise.

### 9.2 Compound head
LightGBM multiclass over slick compounds only (wet stops are rain-driven, excluded).
Uses `class_weight="balanced"` (**measured** to lift minority SOFT/MEDIUM:
macro-F1 0.60 → 0.62, accuracy 0.655 → 0.668).

**Serving rain gate.** Because the head is slick-only, calling it in the wet would
return a slick recommendation in the rain. So `serving/recommend.py` gates the
compound on `rainfall` **and track temperature**: on a rainy pit it recommends the
rule-based wet tyre (`INTERMEDIATE`) only when the track is **cool** (`track_temp <
WET_TRACK_TEMP_C`, 26 °C); when it is warm-and-raining (likely a light shower or a
drying track) it defers to the slick head, as it does in the dry. The response
carries `compound_source ∈ {"model", "wet-rule"}` so callers see which path decided.

The wet branch is **a rule, not ML, on measured grounds** (§12 #20): ML with
`rainfall` as a feature was tried two ways and neither reliably beat the rule — a
4-class head scored *worse* on the wet subset (0.524 vs 0.610) and dented the dry
head; a binary wet-vs-slick model's apparent gain sat inside a ±0.175 fold std. But
the rule is **not** flat "always INTERMEDIATE": **track temperature** is the
strongest *available* wet-vs-slick discriminator (a warm track sheds water → slicks
viable; a cool track holds it → inters). Adding that one factor lifts the gate's
wet-regime accuracy **0.587 → 0.706** on 378 rainy pits (leak-free nested CV ≈0.70,
robust across 24–28 °C, variance roughly halved). `evaluate_wet_gate` logs both
(`wet_gate_gate_accuracy` vs `wet_gate_flat_gate_accuracy`) plus the **0.413**
slick-fit rate. INTERMEDIATE and WET are collapsed (full `WET` is unpredictable —
~13 stops, and inter-vs-full-wet needs rain intensity / standing water we lack). The
*timing* of the wet↔dry switch (rain onset, dry-out crossback) lives on the pit
head's rain-state features (§12 #18–19), not here; track-drying time is unobservable
in public data, so no drying-rate model is attempted.

### 9.3 Baselines
"Never pit" (majority) and a rule-based strategy for the pit head; most-frequent
compound for the compound head.

---

## 10. Every metric, what it means, and its expected range

- **Average Precision (AP / PR-AUC)** — area under the precision-recall curve.
  Range [0, 1]; a random model scores ≈ the positive rate (here ≈0.097 for the
  window target). **Ours ≈0.335 (grouped CV)** ≈ 3.4× the random floor. 1.0 is
  perfect; the *practical* ceiling is unknown and bounded by imitation-label noise
  (real teams are themselves inconsistent), so we do not claim a specific
  "achievable best".
- **ROC-AUC** — probability the model ranks a random pit-lap above a random
  stay-lap. 0.5 = random, 1.0 = perfect. **Ours ≈0.79** = good discrimination.
- **Recall (sensitivity)** — of actual positives, the fraction caught.
  **Ours ≈0.68** (temporal). 1.0 = catches every pit.
- **Precision** — of predicted positives, the fraction correct. **Ours ≈0.21**
  (temporal) — deliberately low: positives are rare and the asymmetric cost favours
  catching pits (recall) over avoiding false alarms.
- **Asymmetric strategic cost** — mean per-decision cost: a missed pit costs
  `cost_fn` (default 5), a spurious pit costs `cost_fp` (default 1), correct calls
  cost 0. **Lower is better; 0 is perfect.** "Never pit" costs `cost_fn × positive
  rate`. **Ours ≈0.317 vs 0.462** for never-pit (grouped CV, 5:1).
- **Macro-F1 (compound)** — unweighted mean of per-class F1, so SOFT/MEDIUM/HARD
  count equally despite imbalance. Range [0, 1]. **Ours ≈0.62.**
- **Accuracy (compound)** — fraction of correct compound predictions. **Ours ≈0.668
  vs 0.526** most-frequent. (Accuracy is deliberately **absent** from the pit head —
  under ~97% stay-out it is meaningless.)
- **Calibration** — a predicted probability *p* should occur with frequency *p*;
  inspected via the reliability curve (`pit_calibration.png`).
- **scale_pos_weight** — LightGBM's positive-class weight = negatives/positives
  (~9.38), countering imbalance.

---

## 11. Cost-ratio robustness (grounding the 5:1)

The 5:1 missed-pit:spurious-pit ratio is a domain **guess** (a spurious stop costs
the pit-lane time, ~20–25 s; a missed pit costs the time lost on dead tyres, often
more). Rather than defend one number, `cost_ratio_sweep` shows the model beats both
baselines across the whole sensible range (**measured, 5-fold grouped CV**):

| Cost ratio | Model | Never-pit | Beats? |
|---|---|---|---|
| 2:1 | 0.174 | 0.185 | ✅ |
| 3:1 | 0.231 | 0.277 | ✅ |
| 5:1 | 0.317 | 0.462 | ✅ |
| 8:1 | 0.410 | 0.740 | ✅ |
| 10:1 | 0.463 | 0.925 | ✅ |

The conclusion does not depend on the exact ratio.

---

## 12. Full experiment log (chronological, including discarded work)

Every number below was measured in this project. "Temporal" = late-2024 holdout
slice; "CV" = 5-fold grouped cross-validation across all races.

1. **Exact-lap pit target (temporal):** AP 0.100, recall 0.173, precision 0.142;
   cost — model 0.1447, rule 0.1785, never-pit 0.1397. → Model beat the rule but
   **not** "never pit". Diagnosed as label sparsity (§2.3).
2. **Window K=3 target (temporal):** AP 0.174, recall 0.703, precision 0.165; cost —
   model 0.4001, rule 0.3689, never-pit 0.3976. Recall jumped 4×; cost still ~tied
   on this noisy slice.
3. **Grouped CV (window, default params):** cost 0.3515 ± 0.034, AP 0.288 vs
   never-pit 0.462, rule 0.450 — **beats both on every fold.** This revealed the
   "tie" in step 2 was the single-slice noise.
4. **Optuna tuning (window):** 5-fold cost 0.3366 ± 0.031, AP 0.298. (This stage also
   included the fuel-corrected degradation feature, adopted on principle — not
   isolated.) Temporal AP rose to ~0.220, ROC-AUC ~0.774.
5. **+ `circuit` feature (CV, ablation):** cost 0.3366 → 0.3295, AP 0.298 → 0.318.
   **Kept.**
6. **+ `relative_pace_s` (CV, ablation):** cost 0.3295 → 0.3169, AP 0.318 → 0.336.
   **Kept.** Re-tuned Optuna for the new feature set.
7. **`driver` feature (CV, ablation):** Δcost −0.0012, ΔAP +0.0019 — within noise,
   and a memorisation risk. **Rejected.**
8. **Feature-importance audit:** top features = `laps_remaining`, `tyre_age`,
   `stint_position`, `lap_number`, `last_clean_lap_time_s`, `relative_pace_s` —
   sensible strategy signals, not artifacts. (`laps_remaining`/`lap_number` are
   legitimate — pit windows cluster mid-race — but noted as a possible crutch.)
9. **Compound head `class_weight="balanced"` (CV):** macro-F1 0.600 → 0.621,
   accuracy 0.655 → 0.668, balanced-acc 0.592 → 0.616. **Kept.** A more-regularised
   variant was worse (macro-F1 0.603) and **rejected**.
10. **Cost-ratio sweep:** model beats both baselines at 2:1…10:1 (§11).
11. **Gap-trend features (`gap_ahead_trend_s`, `gap_behind_trend_s`) (CV, ablation):**
    Δcost −0.0001, ΔAP +0.002 — within noise. **Rejected and fully reverted.** The
    static gaps + `relative_pace_s` already capture the racing context.
12. **Safety-car deployment feature `sc_just_deployed` (CV, paired per-fold ablation):**
    teams pit far more under a safety car (pit rate ~22% on the deployment lap vs
    ~2.8% otherwise). Adding "the caution came out *this* lap" improved cost on
    **5/5 folds** (mean −0.0044). **Kept** (re-tuned). A companion `laps_under_sc`
    was tested and **rejected** (4/5 alone, and it made the combo *less* consistent,
    3/5). Headline after this round: grouped-CV cost 0.317 → **0.311**, AP **0.343**.
13. **Telemetry / driving-aggression features (CV, ablation on 2024):** added per-lap
    `full_throttle_pct`, `brake_pct`, `mean_speed`, `long_load`, `throttle_modulation`
    from the ~10 Hz car-data trace (a deliberately heavy data pull). On a 2024-only
    grouped CV: **cost Δ −0.0002 (only 3/5 folds, per-fold swings ±0.03 — unstable),
    AP +0.030.** So telemetry showed a possible *ranking* signal but **no stable cost
    improvement** — likely because degradation is *already observed directly* via
    `deg_slope_s_per_lap`, making "how hard the lap was driven" largely redundant for
    the pit-timing model. **Not added to the production model.** The telemetry
    extractor (`data/telemetry.py`) is kept as infrastructure for the simulator and
    a future driving-style project, where predicting *unseen* degradation and
    characterising driver style are the actual goals.
14. **Dirty-air / gap-degradation feature (CV, paired per-fold ablation):** running
    close behind another car (dirty air) degrades tyres faster, so we tested
    "cumulative laps this stint within ~1.5 s of the car ahead" (derived from the
    leak-safe `gap_ahead_s`). Result across thresholds: Δcost ≈ −0.0013 on **only
    3/5 folds** (1.0 s threshold: +0.0002, 2/5). Marginal and inconsistent — same
    territory as the rejected `driver`/gap-trend features. **Rejected.** The physics
    is real, but `deg_slope_s_per_lap` already *measures* the degradation (including
    any dirty-air contribution), so the gap-based proxy is redundant and noisy.

**On circuit abrasiveness (decision):** not added as an explicit feature. It is
already captured two ways — by `circuit` (a tree learns each track's effect) and,
more directly, by `deg_slope_s_per_lap`, which *measures* the wear abrasiveness
causes. An explicit abrasiveness index (e.g. a per-circuit historical degradation
rate) is redundant for a tree that already has `circuit`, so it was not pursued.

**On fuel load (decision):** not added as an explicit feature — fuel mass ∝ laps
remaining, and `laps_remaining` is already a feature (an explicit "fuel kg" is just
a rescaling of it).

15. **Training-data scope — does adding 2025 (and older history) help?**
    (`experiments season-scope`, logged in `experiments/RESULTS.md`). 2025 is the
    newest season, so it can only be judged indirectly: a grouped-CV pool ablation
    plus forward-chaining temporal holdouts (fixed test = 2025).
    - *Grouped CV (trustworthy):* adding 2025 to the 2022–2024 pool lowers the pit
      cost slightly (0.326 → 0.322) and, more importantly, **tightens the estimate
      sharply** (fold std 0.039 → 0.0095); pit AP improves 0.334 → 0.348. Going
      further back to all years (2018–2025) helps cost marginally but *hurts* AP
      (0.348 → 0.307) and compound accuracy (0.665 → 0.624 vs 0.649 for 2022–2025).
    - *Temporal holdout (test = 2025):* the model beats both baselines on 2025, and
      a larger/newer training window monotonically lowers strategic cost
      (0.342 → 0.330 → 0.325 as we grow train 22–23 → 22–24 → 18–24).
    - **Decision:** the modern ground-effect era (**2022 onward**) is the sweet spot.
      Adding 2025 is a measured net win for the pit head; pre-2022 data (different
      tyres/regulations, sparse 2020) adds noise to the compound head. Recommend
      expanding the production split to train 2022–2024 / hold out 2025. **Shipped.**
    - *2026 update (measured):* adding the 2026 races to the training pool **hurts**
      (grouped CV AP 0.350 → 0.338, compound acc 0.659 → 0.640, cost std 2.5× noisier).
      2026 is a **new regulation era** (new power units / active aero), so it is to
      2022–2025 what pre-2022 is — a different distribution. **2026 is therefore kept
      out of training** (the model stays on 2022–2025) but **is browsable in the demo**;
      it will get its own era boundary once it has enough complete races. This is the
      guardrail working as intended — we measured before changing the model, and didn't.
16. **Team / constructor feature (`experiments team-feature`, grouped CV 2022–2025):**
    do teams follow learnable strategy patterns? Added `team` as a categorical to
    both heads. **Pit head:** negligible (cost 0.3224 → 0.3222, AP 0.3479 → 0.3499 —
    within noise, fold std slightly worse). **Compound head:** a small but consistent
    gain — accuracy 0.6492 → 0.6588 (+1.0pp), macro-F1 0.6166 → 0.6263 (+1.0pp).
    Reading: *when* to pit is driven by tyre/race state, not team identity, but *which
    compound* is mildly team-flavoured (constructor tyre preferences). **Adopted into
    the compound head only** — a per-fold check confirmed accuracy improves in **5/5
    folds** (macro-F1 4/5). `team` is carried in the dataset and added to
    `COMPOUND_FEATURES`; the pit head stays on `DEFAULT_FEATURES`, so its metrics are
    byte-identical (verified). Serving adds an optional `team` field (defaults to
    `UNKNOWN`, handled by `OneHotEncoder(handle_unknown="ignore")`, so old callers and
    unseen teams degrade gracefully).
17. **Cost-ratio robustness (`experiments cost-ratio`, grouped CV 2022–2025):** the
    model beats both baselines at **every** missed:spurious ratio from 2:1 to 10:1
    (e.g. 5:1 → model 0.322 vs never-pit 0.457 vs rule 0.441; 10:1 → 0.499 vs 0.913
    vs 0.784). The win is not an artefact of the 5:1 choice. Logged for the record.
18. **Wet↔dry regime feasibility (`experiments wet-regime`, 2022–2025 scoping):** of
    92 races, **12 (13%)** saw wet tyres; **6,406 wet laps (6.3%)**; **253 slick↔wet
    crossovers** (95 slick→wet, 158 wet→slick). Enough signal to exploit the *onset*
    (dry→rain) cheaply via leak-safe rain-transition features on the existing pit head
    (rain is an observable condition, not timing-derived, so it does not leak), and
    only adopt on a measured, non-harmful CV result. A full wet-tyre *compound* model
    stays a separate, deferred build (track-drying time is unobservable in public data).
19. **Rain-onset transition features (`experiments rain-onset`, grouped CV 2022–2025):**
    added leak-safe `rain_just_started` / `rain_just_stopped` / `laps_in_rain_state`
    (the dry→wet switch is when teams box for inters; all observed at lap start, so
    leak-free like `sc_just_deployed`). **Result — overall:** neutral-to-slightly
    negative on the pit head (cost 0.3224 → 0.3257, *inside* the fold std, so noise;
    AP 0.3479 → 0.3454) with a tiny compound gain (0.6492 → 0.6530). **Wet races only
    (13 races):** directionally positive but very noisy (cost 0.5168 → 0.5087, AP
    0.194 → 0.197, ±0.06). **Reading:** the dry→rain signal is real and helps *where it
    matters*, but the existing per-lap `rainfall` flag already captures most of it, so
    the transition features add complexity for a within-noise overall change. By the
    measure-it-or-reject rule this is a **borderline reject** for the production serving
    contract — kept tested-and-documented, available to wire in (the ablation lives in
    `experiments.py`) if the wet-race narrative is wanted. **No harm done either way.**
20. **Wet compound choice — ML vs deterministic rain gate (grouped CV, 2018–2026):**
    serving gates the compound on rainfall (raining → INTERMEDIATE) rather than calling
    the slick-only head in the wet. Tested whether ML beats that rule, since `rainfall`
    is already a feature: (a) a **4-class head** `{SOFT,MED,HARD,INTER}` (WET collapsed
    into INTER) trained on *all* strategic pits scored **0.524** on the wet subset vs the
    gate's **0.610**, and *also* nudged the dry subset down (0.635 → 0.621) — strictly
    worse; (b) a **binary wet-vs-slick** model on rainy pits only scored 0.662 but with a
    **±0.175** fold std — the apparent +0.05 sits entirely inside the noise, and it needs
    extra two-stage machinery. **Reading:** the variable that truly decides slick-vs-wet
    is rain *intensity* / standing water, absent from our (boolean `rainfall`) features;
    with ~12 wet races the proxy signal can't beat a majority-class rule with any
    stability. **Kept a deterministic gate** — but not a *flat* one. **Track temperature
    refinement (measured):** of the available features, `track_temp` is the strongest
    wet-vs-slick discriminator at a rainy pit (slick-fits averaged 29 °C, wet-fits 24 °C
    — a warm track sheds water). A rule "fit INTERMEDIATE only when raining *and*
    `track_temp < 26 °C`, else defer to the slick head" lifts wet-regime accuracy
    **0.587 → 0.706** (whole set) / **≈0.70 ±0.19** under *leak-free nested CV* (cutoff
    chosen on train races, scored on unseen races; folds converged on 25–26.5 °C), and
    **every** cutoff in 24–32 °C beats flat-INTER (0.65–0.73) — a broad, physically
    grounded effect, not a tuned knife-edge — with the fold variance roughly halved
    (0.27 → ~0.15–0.19). This is why a *rule* using one strong feature beats the ML
    models here: on ~360 race-clustered points a single hard threshold is more stable
    than a many-feature classifier. `evaluate_wet_gate` reports `gate_accuracy` (the
    track-temp rule), `flat_gate_accuracy` (the old always-INTER baseline), and the
    **0.413** slick-fit rate. INTERMEDIATE and WET are collapsed (full WET is
    unpredictable, ~13 stops; inter-vs-full-wet needs intensity data we lack).

**Other things tried and discarded along the way:** the MLflow file store (deprecated
→ switched to SQLite); `black` (superseded by `ruff format`); `CalibratedClassifierCV(cv="prefit")`
(removed in sklearn 1.6 → `FrozenEstimator`).

---

## 13. Known limitations and honest caveats

- **Imitation ceiling.** We predict what teams *did*, inheriting their mistakes. The
  fix is the deferred Project 1.5 simulator (counterfactual "optimal" labels).
- **Compound head depends on hidden information** — remaining tyre sets, the planned
  strategy, track position — that public data does not expose. 0.668 acc / 0.62
  macro-F1 may be near the achievable ceiling for this feature set.
- **Gaps are approximate** (line-crossing order; lapped traffic → NaN). Rival-gap
  *dynamics* were tested and added nothing (§12.11).
- **`FUEL_S_PER_LAP = 0.055` is an approximation**, and fuel correction was not
  independently A/B'd.
- **The cost ratio is a domain choice**, mitigated by the robustness sweep (§11).
- **The temporal test slice is small and noisy** — which is exactly why the grouped
  CV is the headline.
- **Precision is low (~0.21)** by design (recall-favouring); a deployment would pick
  the threshold from the real operating cost.

---

## 14. How to reproduce

```bash
brew install libomp            # macOS: LightGBM's OpenMP runtime
just install                   # uv sync the workspace
just data                      # build the 2022-2024 dataset (FastF1, cached)
just tune                      # optional: re-run Optuna -> configs/best_params.json
just train                     # train + evaluate + log to MLflow + save artifacts
just mlflow-ui                 # browse experiments
just check                     # ruff + pyright + pytest
```

Artifacts land in `artifacts/` (gitignored): `pit_model.joblib`,
`compound_model.joblib`, `metadata.json`, and the calibration / confusion /
importance plots.

---

## 15. What is genuinely still open (no invented answers)

- Whether a survival / time-to-next-pit formulation beats the binary window — **not
  tested**.
- The true achievable ceiling on AP / cost given imitation-label noise — **unknown**;
  we do not claim a number.
- Serving (FastAPI `/health`, `/predict`) and a Gradio demo — **not yet built**;
  planned next.
- A SHAP analysis (beyond gain-based importance) — **not done**.
