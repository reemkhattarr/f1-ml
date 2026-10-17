# F1 ML Portfolio — Claude Code Master Prompt

## Who you are working with

Reem Khattar. Masters-level ML engineer with hands-on experience in PyTorch and TensorFlow. Strong on deep learning, growing classical ML. Domain knowledge in F1 via a race engineer contact. Target roles: ML Engineer, Applied Scientist, with secondary interest in F1-specific roles (performance analyst, simulator engineer, data engineer at a team).

## What this repo is

A professional ML portfolio built entirely on F1 data using the FastF1 Python library. Four projects, each demonstrating a distinct skill cluster relevant to senior ML engineering roles. All free to build and deploy. No paid infrastructure.

The repo lives at: `/Users/reemkhattar/Desktop/jj/f1-ml`

## Repo structure

```
f1-ml/                               (uv workspace root; one shared uv.lock)
├── CLAUDE.md                        (this file)
├── README.md                        (portfolio landing page)
├── pyproject.toml                   (virtual workspace coordinator + shared tooling config)
├── justfile                         (reproducible commands — NOT a Makefile)
├── .github/workflows/               (CI: data refresh + explorer backfill)
├── data/                            (gitignored, SHARED: FastF1 cache + all processed parquets)
├── 00-data-explorer/                (shared data FOUNDATION — package f1_foundation)
├── 01-tire-strategy/
├── 02-lap-anomaly-detection/
├── 03-race-outcome-prediction/
└── 04-driving-style-clustering/
```

`00-data-explorer` is the shared foundation every project builds on (FastF1
ingestion + the data explorer + the shared cache/data config). The dependency
arrow is always `01–04 → 00`, never back. Each project member follows:

```
00-data-explorer/                    (package f1_foundation)
├── pyproject.toml                   (its own deps; a workspace member)
├── src/f1_foundation/
│   ├── config.py                    (SEED + the SHARED FastF1 cache and data/ paths)
│   ├── ingest.py                    (raw FastF1 → clean per-lap tables)
│   ├── explorer.py / explorer_data.py   (the data explorer + all-sessions builder)
├── deploy/                          (explorer HF Space build/deploy + backfill workflow)
└── tests/

01-tire-strategy/                    (package tire_strategy; depends on f1_foundation)
├── pyproject.toml
├── notebooks/                       (exploration only, not production code)
├── src/tire_strategy/
│   ├── config.py                    (re-exports shared paths from f1_foundation; adds project paths)
│   ├── data/                        (labels, features, dataset — strategy-specific)
│   ├── models/                      (pipelines, CV, metrics, baselines)
│   ├── serving/                     (FastAPI app + Gradio demo)
│   └── simulator/                   (pit-loss vs degradation race simulator)
├── tests/  ·  docs/  ·  deploy/  ·  experiments/  ·  configs/
├── artifacts/                       (gitignored, trained models + plots)
└── mlflow/                          (gitignored, local MLflow tracking)
```

Commands live in the root `justfile` (`just install/data/train/serve/demo/test/check/...`).
Data and artifacts are gitignored; the FastF1 cache and **all** processed parquets
live once in the shared top-level `data/`.

## Tech stack — always use these, do not deviate unless there is a strong reason

- Python 3.11
- uv for environment + dependency management (NOT pip). The repo is a single uv **workspace**; each project is a member with its own `pyproject.toml` sharing one root `uv.lock`. Deps are pinned by the lockfile, not by `pip freeze`.
- FastF1 for all F1 data ingestion
- pandas, numpy for data manipulation
- scikit-learn for preprocessing, pipelines, and classical models
- XGBoost and LightGBM for gradient boosting
- PyTorch for all deep learning
- MLflow for experiment tracking (local server, free)
- FastAPI + uvicorn for serving
- pytest for tests
- just (justfile) for reproducible commands (NOT Make)
- ruff for BOTH linting and formatting (supersedes black — single tool)
- pyright for static type checking; code is fully typed
- pre-commit for git hooks; DVC for data/model versioning
- Hugging Face Spaces + Gradio for live demos
- GitHub Pages for portfolio landing page

Do not use TensorFlow in these projects. PyTorch only for deep learning. Do not use Jupyter notebooks for production code, only for exploration.

## The four projects

### Project 1: Tire Strategy Prediction
`01-tire-strategy/`

**The problem:** Predict optimal pit stop timing and tire compound selection given live race state. This is what race strategy engineers actually do. The model takes as input: lap number, current compound, tire age, gap to car ahead, gap to car behind, track temperature, air temperature, safety car status, degradation rate estimate, and stint position. Output is a multiclass classification: stay out, pit for soft, pit for medium, pit for hard.

**Data:** FastF1 lap data and telemetry across at least 3 full seasons (2022, 2023, 2024). All circuits. Use `fastf1.get_session()` and session.laps for lap-level data. Extract pit stop events from timing data.

**ML approach:** Full sklearn Pipeline with custom transformers for feature engineering. Primary model: LightGBM classifier. Baseline: majority class and a simple rule-based strategy. Evaluation: accuracy is not enough -- use calibration curves, confusion matrices per compound, and a custom metric that penalizes strategically catastrophic errors (e.g. predicting "stay out" when a pit is overdue). Track all experiments in MLflow with full parameter logging.

**Serving:** FastAPI endpoint that accepts race state as JSON and returns compound probabilities with confidence. Include a `/health` endpoint and a `/predict` endpoint. Write OpenAPI docs properly.

**Demo:** Gradio app on Hugging Face Spaces. User selects a historical race, scrubs through laps, and sees what the model would have recommended at each point vs what actually happened.

**What this demonstrates to hiring managers:** Production ML thinking. Clean pipelines. Experiment tracking. Serving infrastructure. Honest evaluation. Domain credibility.

### Project 2: Lap Time Anomaly Detection
`02-lap-anomaly-detection/`

**The problem:** Model normal driving signatures from raw telemetry and detect anomalies -- mechanical issues, incidents, traffic, unusual tire behavior. Secondary problem: compare driver performance on the same circuit in the same machinery to quantify the gap in a defensible way.

**Data:** FastF1 telemetry at ~10Hz per lap: speed, throttle, brake, gear, DRS, RPM, x/y coordinates on track. Focus on 2-3 circuits with clean data across multiple drivers and seasons. Silverstone, Monza, Spa are good choices.

**ML approach:** Two components. First, an LSTM autoencoder in PyTorch trained on "clean" laps to learn normal telemetry sequences. Reconstruction error is the anomaly score. Second, a representation layer that embeds laps into a latent space for driver comparison. Train on laps flagged as clean (no yellow flags, no incidents, not an in or out lap). Evaluate anomaly detection against known incidents from race reports.

**Architecture specifics:** Input is a multivariate time series of shape (sequence_length, n_features). Normalize per feature per session. Use a bidirectional LSTM encoder, bottleneck layer, LSTM decoder. Loss is MSE on reconstruction. Threshold on reconstruction error using a percentile of the training distribution, not a hardcoded value.

**What this demonstrates:** Deep learning on real sequential data. Proper data cleaning for time series. Unsupervised anomaly detection. PyTorch fluency. The driver comparison angle is credible to F1 domain experts.

### Project 3: Race Outcome Prediction with Uncertainty Quantification
`03-race-outcome-prediction/`

**The problem:** Predict finishing position distributions for each driver, not point estimates. Predictions should update as more information becomes available: pre-race (grid position only), post-qualifying (sector times), mid-race (live gaps, safety car events). The key differentiator vs naive race prediction is calibrated uncertainty.

**Data:** Full session results, qualifying times, lap data, weather, and historical head-to-head records across 2018-2024. FastF1 combined with the Ergast API for historical results (Ergast is free and has a Python wrapper called `python-f1`).

**ML approach:** Gradient boosting baseline (XGBoost) for point estimates. Main model: use conformal prediction to wrap the XGBoost model and produce valid prediction intervals with guaranteed coverage. This is more defensible and production-realistic than a Bayesian neural network for this problem. Implement conformal prediction from scratch using `nonconformist` or manually -- do not hide it behind a library. Evaluate calibration explicitly: a 90% prediction interval should contain the true outcome 90% of the time. Plot calibration curves.

**Serving:** FastAPI with three endpoints corresponding to the three prediction stages. State is passed in the request, not stored server-side.

**Demo:** Gradio app. User selects a race weekend, sees pre-race predictions, then reveals qualifying results and sees predictions update, then simulates mid-race lap by lap.

**What this demonstrates:** Probabilistic ML and uncertainty quantification, which is what separates senior ML engineers from junior ones in interviews. Conformal prediction is a current and credible technique. The multi-stage prediction angle is a strong system design story.

### Project 4: Driving Style Clustering
`04-driving-style-clustering/`

**The problem:** Characterize and cluster driving styles from raw telemetry without labels. Extract meaningful representations of how different drivers attack corners, apply brakes, manage throttle, and recover energy. Tell a visual, data-driven story about driver signatures.

**Data:** FastF1 telemetry for all drivers across a full season (2023 or 2024) at 5-6 circuits covering different character: Monza (power), Monaco (technical), Silverstone (high-speed), Hungary (slow corners), Spa (mixed).

**ML approach:** Two-stage. First, engineer handcrafted features per corner per lap: brake point distance from apex, throttle application point, minimum speed at apex, steering smoothness (derivative of steering angle), DRS usage patterns, ERS deployment signature. These features encode domain knowledge explicitly. Second, train a convolutional autoencoder on raw telemetry traces to learn a latent representation without handcrafting. Compare the two: do they agree on driver similarity? The comparison itself is a finding worth discussing. Clustering: HDBSCAN (better than k-means for this, handles noise, no need to specify k). Visualization: UMAP to 2D, plotted with driver labels and circuit annotations.

**What this demonstrates:** Unsupervised learning done properly. Feature engineering with domain knowledge. The ability to tell a story from data, which is underrated as a portfolio signal and critical for any role that involves communicating to non-ML stakeholders (which F1 roles all require).

## Working conventions

Every time a new Claude Code session starts, read this file first and confirm you understand the current state of the project before doing anything.

At the start of each session, check which project is being worked on and ask Reem: "Where did we leave off? Which file were you last working on?" Do not assume.

Never rewrite working code without explaining why the current version is wrong. Prefer surgical edits over full rewrites.

When writing model training code, always include: a seed for reproducibility, a clear train/val/test split with no leakage, logging to MLflow, and a saved artifact (the trained model) at the end.

When writing FastAPI code, always include: input validation with Pydantic, proper HTTP status codes, a health check endpoint, and at least one test in pytest.

When writing data ingestion code with FastF1, always enable the shared on-disk cache first via `f1_foundation.enable_cache()` (it points at the shared top-level `data/cache`). FastF1 makes HTTP requests to the F1 timing API and is rate-limited to ~500 calls/hour, so caching is essential to avoid hitting the limit and to make development fast. Raw ingestion (`load_race`, `load_results`) lives in `f1_foundation` — projects import it, they do not re-implement it.

Do not write notebooks for anything that will be part of the final codebase. Notebooks go in `notebooks/` and are for exploration only. Production code goes in `src/`.

Manage dependencies with uv: `uv add <pkg>` inside the relevant workspace member, never `pip install`. The `uv.lock` file pins exact versions and is committed. Keep each project's declared dependencies minimal and purposeful — add a dep only when it is actually imported.

## Deployment targets

- Live demos: Hugging Face Spaces (Gradio). Free tier. Each project gets its own Space.
- Portfolio landing page: GitHub Pages from the root README or a `/docs` folder.
- FastAPI backends for demos: Render free tier or Railway free tier. Stateless, no persistent DB needed.

## Current state

_Last updated: 2026-06-02._

**00 — Data foundation & explorer: COMPLETE and live.** Extracted into its own
workspace member `00-data-explorer` (package `f1_foundation`): raw FastF1 ingestion
(`load_race`/`load_results`), the shared FastF1 cache + top-level `data/`, and the
**F1 Data Explorer** (a Gradio app that browses every field of every dataset — all
session types FP/Q/Sprint/R, weather, results, lap timing). Live on Hugging Face.
An hourly, rate-limit-aware GitHub Actions workflow backfills the all-sessions data
in the cloud (resumable; ~300k+ lap rows and growing).

**01 — Tire Strategy: COMPLETE end-to-end and live.** Data (2018–2026 ingested;
trained on the 2022–2025 ground-effect era, holdout 2025), a decomposed two-head
model (calibrated pit-timing head + slick compound head, with a flat
raining -> INTERMEDIATE gate for wet stops; the track-temp refinement was reverted in #30), grouped-CV evaluation with an asymmetric strategic-cost
metric, MLflow tracking, a persistent experiment log (`experiments/RESULTS.md`),
a FastAPI service, a pit-loss-vs-degradation race simulator, and a Gradio demo
(strategy timeline, results panel, auto race notes) — all live on Hugging Face.
Headline (5-fold grouped CV): pit cost **0.312** vs 0.46 never-pit; compound acc
**0.659**. See `01-tire-strategy/docs/IMPLEMENTATION.md` for the full retrospective.

**Tooling/infra in place:** uv workspace (one `uv.lock`), ruff + pyright + pytest
all green, pre-commit, the root `justfile`, and **two GitHub Actions workflows**
(weekly model refresh + the explorer backfill). Docker/Terraform/DVC still deferred
per the ML-first plan.

**Next:** Project **02 — Lap-Time Anomaly Detection** (PyTorch LSTM autoencoder on
telemetry). Add it as a workspace member depending on `f1_foundation`; new project
parquets written to the shared `data/processed/` appear in the explorer automatically.

### Decisions made
- Repo layout: single uv-workspace **monorepo**; `00-data-explorer`/`f1_foundation`
  is the shared foundation (`01–04 → 00`). Each project stays independently demoable.
- Training era boundary is **2022+** (ground-effect regs); pre-2022 and the new-reg
  **2026** season measurably hurt, so both are excluded from training (2026 is
  browsable in the demo only). Every feature/data change is adopted only on a
  measured CV win — see the experiment log.
- Infra: **ML-first** — models/pipelines first, layer Docker/Terraform/DVC later.
- Tooling overrides: uv (not pip), just (not Make), ruff format (not black).