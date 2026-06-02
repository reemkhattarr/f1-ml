"""FastAPI app serving the tire-strategy model.

Endpoints:
- ``GET /health``  — liveness + whether the trained models are available.
- ``POST /predict`` — race state in, pit + compound recommendation out.

The model bundle is provided via a dependency so tests can inject a lightweight
in-memory model instead of the on-disk artifacts.
"""

from __future__ import annotations

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from tire_strategy.serving.models import ModelBundle, load_bundle
from tire_strategy.serving.recommend import recommend_compound
from tire_strategy.serving.schemas import HealthResponse, PredictionResponse, RaceState

app = FastAPI(
    title="F1 Tire Strategy",
    version="0.1.0",
    summary="Predict pit timing and tyre compound from live race state.",
)

_cache: dict[str, ModelBundle] = {}


def get_bundle() -> ModelBundle:
    """Dependency: load (and cache) the trained model bundle, or 503 if missing."""
    if "bundle" not in _cache:
        try:
            _cache["bundle"] = load_bundle()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503, detail="Models unavailable; run `just train` first."
            ) from exc
    return _cache["bundle"]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Reports whether the trained models can be loaded."""
    try:
        get_bundle()
        loaded = True
    except HTTPException:
        loaded = False
    return HealthResponse(status="ok", models_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(state: RaceState, bundle: ModelBundle = Depends(get_bundle)) -> PredictionResponse:
    """Return the pit + compound recommendation for one race state."""
    features = pd.DataFrame([state.model_dump()])

    pit_probability = float(bundle.pit_model.predict_proba(features)[0, 1])
    compound = recommend_compound(bundle.compound_model, features, rainfall=state.rainfall)

    return PredictionResponse(
        pit_probability=pit_probability,
        pit_recommended=pit_probability >= bundle.threshold,
        decision_threshold=bundle.threshold,
        recommended_compound=compound.compound,
        compound_probabilities=compound.probabilities,
        compound_source=compound.source,
    )
