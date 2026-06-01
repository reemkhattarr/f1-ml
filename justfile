# F1 ML portfolio — task runner.
# `just` with no arguments lists all available recipes.

set dotenv-load := true

default:
    @just --list

# Install/sync the entire workspace (all members + dev tools) from uv.lock.
install:
    uv sync --all-packages

# Install git pre-commit hooks.
hooks:
    uv run pre-commit install

# Lint and format-check (no changes written) — CI parity.
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix lint issues and format the codebase.
fmt:
    uv run ruff check --fix .
    uv run ruff format .

# Static type checking.
types:
    uv run pyright

# Run the test suite.
test:
    uv run pytest

# Full local CI gate: lint + types + tests.
check: lint types test

# --- Project 1: Tire Strategy -------------------------------------------------

# Build the full modelling dataset. Heavy on first run (many FastF1 loads),
# fast thereafter (cached). Optionally pass seasons, e.g. `just data 2024`.
data *SEASONS:
    uv run python -m tire_strategy.data.dataset {{SEASONS}}

# Build the clean stint-laps dataset for the simulator (Project 1.2).
stint-data:
    uv run python -m tire_strategy.simulator.data

# Estimate per-circuit pit-loss for the simulator (Project 1.2).
pitloss-data:
    uv run python -m tire_strategy.simulator.pitloss

# Incrementally add new/missing races to the dataset (backfill or post-race).
# e.g. `just update-data 2025 2026`. No args -> the default seasons.
update-data *SEASONS:
    uv run python -m tire_strategy.data.dataset --update {{SEASONS}}

# Post-race refresh: add new races, retrain, rebuild the deployable demo bundle.
refresh:
    uv run python -m tire_strategy.data.dataset --update 2022 2023 2024 2025 2026
    just train
    just build-space

# Push the rebuilt demo bundle live to Hugging Face Spaces.
deploy-demo:
    uv run hf upload reemkhattar/f1-tire-strategy 01-tire-strategy/deploy/space_build . --repo-type space

# Launch the data explorer locally (browse all dataset fields).
explorer:
    uv run python -m tire_strategy.serving.explorer

# Build + push the data-explorer Space to Hugging Face.
build-explorer-space:
    uv run python 01-tire-strategy/deploy/build_explorer_space.py
deploy-explorer:
    uv run hf upload reemkhattar/f1-data-explorer 01-tire-strategy/deploy/explorer_space_build . --repo-type space

# Rebuild the explorer datasets (all sessions, all seasons) + explorer bundle.
refresh-explorer:
    uv run python -m tire_strategy.serving.explorer_data
    just build-explorer-space

# Full post-weekend refresh: model demo + explorer bundles (build only, no deploy).
refresh-all: refresh refresh-explorer

# Deploy both Spaces live (run after `just refresh-all`). See docs/DATA_UPDATES.md.
deploy-all: deploy-demo deploy-explorer

# Tune the pit-head hyperparameters (Optuna) -> artifacts/best_params.json.
tune *ARGS:
    uv run python -m tire_strategy.models.cv {{ARGS}}

# Train + evaluate both model heads, logging to local MLflow.
train:
    uv run python -m tire_strategy.train

# Run a named experiment and append results to experiments/RESULTS.md.
# e.g. `just experiment season-scope`
experiment NAME:
    uv run python -m tire_strategy.experiments {{NAME}}

# Launch the local MLflow UI to browse experiments.
mlflow-ui:
    uv run mlflow ui --backend-store-uri file:01-tire-strategy/mlflow

# Serve the trained model via FastAPI (docs at http://localhost:8000/docs).
serve:
    uv run uvicorn tire_strategy.serving.app:app --reload --port 8000

# Launch the Gradio demo (scrub a historical race: model vs reality).
demo:
    uv run python -m tire_strategy.serving.demo

# Assemble a deployable Hugging Face Space (needs `just data` + `just train` first).
build-space:
    uv run python 01-tire-strategy/deploy/build_space.py
