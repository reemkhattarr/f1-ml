# Project 1 — Tire Strategy Prediction

Predict, from the live state of an F1 car at the start of a lap, the tyre-strategy
decision a race engineer would make:

1. **Pit head** — will this car pit soon? (a calibrated probability)
2. **Compound head** — if it pits, onto which slick compound (SOFT / MEDIUM / HARD)?

Built entirely on free, public F1 timing data (**FastF1**) for 2022–2025, the
current ground-effect regulation era. The goal is *production ML thinking* on real,
messy sequential data: honest labels, leak-free features, calibrated uncertainty,
a strategically-meaningful metric, trustworthy evaluation, and a deployable service
+ demo.

## Headline result (5-fold grouped CV across all 92 races)

| Metric | Model | "Never pit" | Rule-based |
|---|---|---|---|
| Asymmetric strategic cost ↓ | **0.312 ± 0.010** | 0.457 | 0.441 |
| Average precision (pit head) | **0.354** | — | — |
| Compound accuracy / macro-F1 | **0.659 / 0.626** | 0.505 | — |

The model beats both baselines by ~29–32% on cost, and the win **holds across every
cost ratio from 2:1 to 10:1** — it is not an artifact of one assumption.

## What makes it credible (not just "a model")

- **The labelling problem is the real work.** We predict *what the team did*
  (imitation), framed as "pit within the next 3 laps" — a window, because a strategy
  call is not an exact lap. We exclude red-flag (free) tyre changes and retirees'
  final laps so the labels are honest.
- **Leak-free by construction.** Anything derived from lap timing is lagged one lap;
  a test proves an in-lap's slow time never leaks into its own features.
- **Honest evaluation.** A single temporal slice looked like "tied with never-pit";
  grouped CV across all races showed a clear, consistent win. We report both.
- **Domain-aware features.** Fuel-corrected tyre degradation (separating wear from a
  lightening car), field-relative pace, circuit identity, safety-car/rain context.
- **Explainable.** A gain-importance audit shows the model leans on tyre age,
  degradation, and pace — sensible signals, not artifacts.

## Pipeline

```
f1_foundation.ingest → labels → leak-safe features → dataset (~176k laps, 2018-2026;
       trains on the 2022-2025 ground-effect era)
       → race-atomic temporal / grouped-CV split
       → LightGBM pit head (scale_pos_weight + isotonic calibration + cost threshold)
       + LightGBM compound head (slicks only, balanced, team-aware) + rain gate
       → evaluate vs baselines (MLflow) → FastAPI service → Gradio demo
```

## Quickstart

```bash
brew install libomp            # macOS: LightGBM's OpenMP runtime (one-time)
just install                   # uv sync the workspace
just data                      # build the dataset from FastF1 (cached); trains on 2022-2025
just train                     # train + evaluate + log to MLflow + save artifacts
just serve                     # FastAPI at http://localhost:8000 (docs at /docs)
just demo                      # interactive Gradio demo
just check                     # ruff + pyright + pytest (46 tests)
```

Other recipes: `just tune` (Optuna), `just mlflow-ui`, `just fmt`, `just experiment <name>`.

## Repository structure

This project depends on the shared **`f1_foundation`** package (`00-data-explorer`)
for raw FastF1 ingestion and the shared cache/data directory.

```
src/tire_strategy/
├── config.py            # re-exports the shared paths from f1_foundation; adds project paths
├── data/                # labels, features, dataset assembly (ingest lives in f1_foundation)
├── models/              # split, metrics, baselines, pipeline, cv (+ Optuna)
├── simulator/           # pit-loss vs degradation race simulator
├── experiments.py       # the persistent experiment log (-> experiments/RESULTS.md)
├── evaluate.py          # metrics + diagnostic plots
├── train.py             # orchestrates train + eval + MLflow + artifacts
└── serving/             # FastAPI app + Gradio demo + rain gate + model loader
docs/
├── DESIGN.md            # the up-front data/labelling contract
├── IMPLEMENTATION.md    # the complete retrospective (every decision + experiment)
├── DEPLOY.md            # hosting guide (HF Spaces / Render)
└── DATA_UPDATES.md      # keeping data fresh (manual / cron / GitHub Actions)
```

## Documentation

- **[docs/DESIGN.md](docs/DESIGN.md)** — the data & modelling contract decided before
  any code.
- **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** — the full retrospective: what
  every file does, every decision and the alternatives, the complete experiment log
  (including discarded work), and precise metric definitions.
- **[docs/DEPLOY.md](docs/DEPLOY.md)** — deployment.

## Tech stack

Python 3.11 · uv · FastF1 · pandas/numpy · scikit-learn · LightGBM · MLflow · Optuna
· FastAPI · Gradio · pytest · ruff · pyright. LightGBM requires `libomp` on macOS.

## Honest limitations

Imitation labels inherit teams' real mistakes (the planned **race-simulator
extension** addresses this with counterfactual "optimal" labels). The compound head
depends partly on information public data doesn't expose (remaining tyre sets,
planned strategy). Gaps are reconstructed from line-crossing order (approximate).
See `docs/IMPLEMENTATION.md` §13 for the full list.
