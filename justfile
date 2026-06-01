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

# Tune the pit-head hyperparameters (Optuna) -> artifacts/best_params.json.
tune *ARGS:
    uv run python -m tire_strategy.models.cv {{ARGS}}

# Train + evaluate both model heads, logging to local MLflow.
train:
    uv run python -m tire_strategy.train

# Launch the local MLflow UI to browse experiments.
mlflow-ui:
    uv run mlflow ui --backend-store-uri file:01-tire-strategy/mlflow

# Serve the trained model via FastAPI (docs at http://localhost:8000/docs).
serve:
    uv run uvicorn tire_strategy.serving.app:app --reload --port 8000
