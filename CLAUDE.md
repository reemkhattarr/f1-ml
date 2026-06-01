# F1 ML Portfolio — Claude Code Master Prompt

## Who you are working with

Reem Khattar. Masters-level ML engineer with hands-on experience in PyTorch and TensorFlow. Strong on deep learning, growing classical ML. Domain knowledge in F1 via a race engineer contact. Target roles: ML Engineer, Applied Scientist, with secondary interest in F1-specific roles (performance analyst, simulator engineer, data engineer at a team).

## What this repo is

A professional ML portfolio built entirely on F1 data using the FastF1 Python library. Four projects, each demonstrating a distinct skill cluster relevant to senior ML engineering roles. All free to build and deploy. No paid infrastructure.

The repo lives at: `/Users/reemkhattar/Desktop/jj/f1-ml`

## Repo structure

```
f1-ml-portfolio/
├── CLAUDE.md                        (this file)
├── README.md                        (portfolio landing page)
├── 01-tire-strategy/
├── 02-lap-anomaly-detection/
├── 03-race-outcome-prediction/
├── 04-driving-style-clustering/
```

Each project folder follows this internal structure:

```
01-tire-strategy/
├── README.md
├── requirements.txt
├── data/                            (gitignored, populated by data scripts)
├── notebooks/                       (exploration only, not production code)
├── src/
│   ├── data/                        (ingestion, cleaning, feature engineering)
│   ├── models/                      (training, evaluation, serialization)
│   ├── serving/                     (FastAPI app where applicable)
│   └── utils/
├── tests/
├── mlflow/                          (gitignored, local MLflow tracking)
└── Makefile                         (commands: install, data, train, serve, test)
```

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

When writing data ingestion code with FastF1, always cache sessions locally using `fastf1.Cache.enable_cache('data/cache')`. FastF1 makes HTTP requests to the F1 timing API and caching is essential to avoid hitting rate limits and to make development fast.

Do not write notebooks for anything that will be part of the final codebase. Notebooks go in `notebooks/` and are for exploration only. Production code goes in `src/`.

Manage dependencies with uv: `uv add <pkg>` inside the relevant workspace member, never `pip install`. The `uv.lock` file pins exact versions and is committed. Keep each project's declared dependencies minimal and purposeful — add a dep only when it is actually imported.

## Deployment targets

- Live demos: Hugging Face Spaces (Gradio). Free tier. Each project gets its own Space.
- Portfolio landing page: GitHub Pages from the root README or a `/docs` folder.
- FastAPI backends for demos: Render free tier or Railway free tier. Stateless, no persistent DB needed.

## Current state

_Last updated: 2026-06-01._

**Foundation complete.** The repo is a working uv workspace:
- `pyproject.toml` (root) — virtual workspace coordinator; shared dev toolchain (ruff, pyright, pytest, pre-commit, mlflow, dvc); shared ruff/pyright/pytest config.
- `.python-version` (3.11), `.gitignore`, `justfile` (install/lint/fmt/types/test/check/hooks), `.pre-commit-config.yaml`.
- `01-tire-strategy/` is the first workspace member: own `pyproject.toml` (fastf1, pandas, numpy, scikit-learn, lightgbm, matplotlib, mlflow), `src/tire_strategy/` package with `config.py` (FastF1 cache helper), `tests/`, `data/` (gitignored).
- `uv sync --all-packages` succeeds; FastF1 verified by loading Monza 2024 R (1008 laps, 20 drivers, compounds + stint/tyre-life present).
- Not yet committed to git (no commits on `main` yet); Docker/Terraform/CI/DVC deferred per ML-first plan.

**Next:** begin Project 1 (Tire Strategy) — design the data ingestion across 2022–2024, then the feature pipeline. Projects 02–04 are not yet scaffolded; add each as a workspace member when started.

### Decisions made (session 2026-06-01)
- Repo layout: single uv-workspace **monorepo**; a portfolio-site layer + per-project demos sit on top later. Each project stays independently runnable/demoable.
- Infra: **ML-first** — build models/pipelines first, layer Docker/Terraform/CI/DVC as each project matures.
- Tooling overrides applied: uv (not pip), just (not Make), ruff format (not black). See the Tech stack section.