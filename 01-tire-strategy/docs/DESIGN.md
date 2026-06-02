# Project 1 — Tire Strategy Prediction: Data & Modelling Design

> The hard part of this project is **not** the model. It is constructing honest
> labels and leak-free features from historical races. This document is the
> contract the code must satisfy. Decided before any pipeline code was written.

## 1. Problem statement

Given the **live state of a car at the start of a lap**, predict the strategy
decision a race engineer would make:

1. **Pit at the end of this lap, or stay out?**
2. **If pitting, onto which compound** (SOFT / MEDIUM / HARD)?

## 2. Target formulation — decomposed, not flat 4-class

We do **not** model a flat 4-class target `{stay, pit-soft, pit-med, pit-hard}`.

**Why:** a driver pits ~1–3 times in a ~55-lap race, so ≈98% of all lap rows are
"stay out". A single 4-class model maximises accuracy by always predicting
"stay out" (≈98% accurate, 0% useful). Instead we mirror how strategists reason:

- **Head 1 — Pit decision (binary, calibrated).** `P(pit at end of lap | state)`.
  Calibration is a first-class deliverable: a predicted 30% pit probability must
  occur ~30% of the time. Reliability curve is a headline result.
- **Head 2 — Compound choice (conditional).** `P(compound | state, pitting)`,
  trained **only on pit laps**, where the three compound classes are balanced.

Baselines for comparison (these replace a flat-4-class baseline):
- **Majority class** — always "stay out". Demonstrates why accuracy is the wrong
  metric (≈98% accuracy, ~0 pit recall).
- **Rule-based strategy** — pit when tyre age exceeds a compound-specific
  threshold, or under a safety car with the mandatory-second-compound rule unmet.
  A domain heuristic that is itself credible to a strategy engineer.

## 3. Label construction — imitation of actual team decisions

The label is **what the team actually did** ("predict the strategist's decision").
Imitation labels are objective and unarguable; "optimal" labels would require a
simulator whose invented assumptions we'd then be grading ourselves on
(circular). The optimal-strategy angle is deferred — see §9.

**Decision point.** A car decides at the *end* of a lap whether to pit. FastF1
marks the **in-lap** (the lap the car enters the pits) via a non-null `PitInTime`.

For each (driver, race, lap):
- `label_pit = 1` if the lap is an in-lap, else `0`.
- `label_compound` (only where `label_pit = 1`) = the compound of the **next
  stint** (the tyre fitted at the stop) — i.e. the `Compound` of the following
  stint's laps, **not** the worn compound on the in-lap.

## 4. Leakage register (the core engineering discipline)

Every feature must be knowable **at the start of the decision lap**, before the
outcome of that lap exists. Known traps and how we handle them:

| # | Trap | Handling |
|---|------|----------|
| 1 | In-lap `LapTime` is mechanically slow (pit-lane time) → "slow lap ⇒ pit" leak | Never use the current lap's outcome. Features are lagged to completed laps. |
| 2 | Future stint info (tyre life / compound at lap N+1) reveals the pit | Only past + current-start state is used. |
| 3 | Out-lap / in-lap pace is abnormal | Excluded from pace & degradation features (flagged, not used as "normal" laps). |
| 4 | DNF / retirement truncates a stint | The retirement lap is not a "stay out" decision → marked and excluded; valid prior laps kept. |
| 5 | Red-flag tyre changes are free & rule-driven, not strategic | Flagged; red-flag-driven changes excluded from positive pit labels. |
| 6 | Safety car / VSC — **not leakage, it is signal** (known live, drives many pits) | Included as a feature. |
| 7 | End-of-race (no one pits on the last laps) | Captured by a `laps_remaining` feature, not by dropping rows. |

Data-quality flags used in cleaning: FastF1's `IsAccurate` (timing quality),
`Deleted` / `DeletedReason` (track-limits-deleted laps), and `FreshTyre`.

## 5. Features (all as-of the start of the decision lap)

- `lap_number`, `laps_remaining` (needs race lap total)
- `compound` (current, categorical), `tyre_age` (stint laps so far)
- `stint_number`, `stint_position`
- `degradation_estimate` — slope of clean lap times within the current stint up to
  this lap (excludes in/out/SC laps; uses **past** laps only)
- `gap_ahead`, `gap_behind` — **reconstructed** (FastF1 exposes no gap column).
  Method: for each lap, order cars by `Position`; the gap to the car ahead/behind
  is the difference in their line-crossing session `Time` on that lap. This is the
  "gap at the line each lap" — exactly the quantity a strategist acts on. Handle
  lapped traffic / leader / backmarker as null edge cases.
- `track_temp`, `air_temp` — from `session.weather_data`, merged by time
- `safety_car` / `vsc` status — from the per-lap `TrackStatus` code (1=clear,
  4=SC, 5=red, 6=VSC); no separate merge needed
- `used_second_compound` — whether the mandatory second compound is already used

Compound is **relative** (SOFT/MED/HARD) — the actual decision a strategist makes.
Absolute Pirelli C0–C5 enrichment is a later option.

## 6. Data scope

- **Seasons 2022–2025, all circuits** — the current ground-effect regulation era
  with consistent Pirelli allocation, so degradation behaves comparably. Pre-2022
  (and the new-regulation 2026) would mix car/tyre regulations, so they are excluded
  from training (the full 2018–2026 data is ingested for the demo/explorer only).
- Volume ≈ 70 races × 20 drivers × ~55 laps ≈ **~700k lap rows**.
- Source: FastF1 `session.laps`, `session.weather_data`, track status / race
  control messages. FastF1 cache is mandatory (`tire_strategy.config.enable_cache`).

## 7. Train / validation / test split (no leakage)

- **Atomic unit = race weekend.** All laps of a race go entirely to one split.
  Splitting by lap or by driver-within-race leaks the race's degradation curve.
- **Primary: temporal split** (mimics deployment — predict future races from past):
  train 2022–2023, validate early-2024, test late-2024.
- **Secondary: GroupKFold by race weekend** as a robustness check.

## 8. Evaluation

- **Pit head:** PR-AUC and recall at fixed precision (accuracy is meaningless under
  98% imbalance); **reliability/calibration curve**; confusion matrix.
- **Compound head:** per-compound confusion matrix.
- **Custom asymmetric cost metric:** missing a needed pit (predict "stay out" when a
  pit is overdue — tyre past its cliff) costs more than a spurious pit. Report
  expected cost vs both baselines. "Overdue" defined via compound-specific tyre-age
  / degradation thresholds.
- Class imbalance handled via LightGBM `scale_pos_weight` / class weights, **not**
  resampling that would distort calibration.
- All experiments logged to MLflow with full parameters and artifacts.

## 9. Deferred: optimal strategy via simulation ("Project 1.5")

After the imitation model ships, build a counterfactual race simulator
(tyre-degradation + pit-loss + traffic model) to label states with the
*time-optimal* action and compare against real team decisions. This showcases
simulation skills directly relevant to simulator-engineering roles, and is kept
separate so its modelling assumptions never contaminate the honest imitation core.
