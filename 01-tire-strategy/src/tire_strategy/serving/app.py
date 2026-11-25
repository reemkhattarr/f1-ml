"""FastAPI app: the tire-strategy API and the full local version of the website.

Endpoints:
- ``GET /health``  — liveness + whether the trained models are available.
- ``POST /predict`` — race state in, pit + compound recommendation out.
- ``POST /api/plan`` — pre-race strategy plan for a circuit (strategy engine).
- ``POST /api/advise`` — box or stay out from a live race state (strategy engine).
- ``GET /api/replay`` — any race loaded live (FastF1, else OpenF1) with model calls.
- ``POST /api/refresh`` — add new races, refit the engine, rebuild the site.
- ``/`` — the website (``just app`` builds it first), with the local-only tools on.

The model bundle is provided via a dependency so tests can inject a lightweight
in-memory model instead of the on-disk artifacts.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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


# ----------------------------------------------------------------- strategy engine
class PlanRequest(BaseModel):
    circuit: str = Field(description="Circuit, e.g. 'Monza'. Unknown circuits use averages.")
    laps: int | None = Field(default=None, ge=5, le=90, description="Race laps (default: real)")
    compounds: list[Literal["SOFT", "MEDIUM", "HARD"]] = ["SOFT", "MEDIUM", "HARD"]
    max_stops: int = Field(default=3, ge=1, le=3)
    two_compound_rule: bool = True
    pit_loss: float | None = Field(default=None, ge=5, le=60)
    wear: float = Field(default=1.0, ge=0.3, le=2.0, description="Scale tyre wear (1 = measured)")


class AdviseRequest(BaseModel):
    circuit: str
    total_laps: int | None = Field(default=None, ge=5, le=90)
    lap: int = Field(ge=1)
    compound: Literal["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
    tyre_age: int = Field(ge=0)
    used: list[str] = []
    caution: Literal["none", "sc", "vsc"] = "none"
    rain: bool = False
    gap_ahead: float | None = None
    gap_behind: float | None = None


def _engine_params(circuit: str, pit_loss: float | None = None, wear: float = 1.0) -> dict:
    from tire_strategy.strategy.engine import circuit_params
    from tire_strategy.strategy.params import PARAMS_PATH, load_params

    if not PARAMS_PATH.exists():
        raise HTTPException(status_code=503, detail="No strategy parameters; run `just app`.")
    p = circuit_params(load_params(PARAMS_PATH), circuit)
    if pit_loss is not None:
        p["pit_loss"] = pit_loss
    if wear != 1.0:
        p["degradation"] = {c: v * wear for c, v in p["degradation"].items()}
    return p


@app.post("/api/plan")
def api_plan(req: PlanRequest) -> dict[str, Any]:
    """Fastest legal strategies for a race at ``circuit``."""
    from tire_strategy.strategy.engine import race_plan

    p = _engine_params(req.circuit, req.pit_loss, req.wear)
    laps = req.laps or p["race_laps"]
    if not laps:
        raise HTTPException(status_code=422, detail="Unknown circuit: give the race laps.")
    return race_plan(p, laps, compounds=req.compounds, two_compound_rule=req.two_compound_rule,
                     max_stops=req.max_stops)  # fmt: skip


@app.post("/api/advise")
def api_advise(req: AdviseRequest) -> dict[str, Any]:
    """Box now or stay out, priced over the rest of the race."""
    from tire_strategy.strategy.engine import advise

    p = _engine_params(req.circuit)
    total = req.total_laps or p["race_laps"]
    if not total:
        raise HTTPException(status_code=422, detail="Unknown circuit: give total_laps.")
    try:
        return advise(p, total_laps=total, lap=req.lap, compound=req.compound,
                      tyre_age=req.tyre_age, used=req.used, caution=req.caution, rain=req.rain,
                      gap_ahead=req.gap_ahead, gap_behind=req.gap_behind)  # fmt: skip
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# --------------------------------------------------------------------- local tools
@app.get("/api/health")
def api_health() -> dict[str, Any]:
    """Tells the website it is running locally, so it switches on the local tools."""
    return {"status": "ok", "local": True, "models_loaded": health().models_loaded}


def _live_race(year: int, rnd: int) -> pd.DataFrame:
    """Labelled feature table for one race: FastF1 first (full detail), else OpenF1."""
    from f1_foundation import load_race, load_results, retired_drivers
    from f1_foundation.openf1 import load_race_openf1, load_results_openf1
    from tire_strategy.data.features import build_features
    from tire_strategy.data.labels import build_labels

    try:
        laps, results = load_race(year, rnd), load_results(year, rnd)
    except Exception as exc:  # FastF1 unavailable or blocked: use OpenF1
        logging.getLogger(__name__).info("FastF1 failed (%s); trying OpenF1", exc)
        laps, results = load_race_openf1(year, rnd), load_results_openf1(year, rnd)
    return build_features(build_labels(laps, retired=retired_drivers(results)))


@app.get("/api/replay")
def api_replay(year: int, round: int, bundle: ModelBundle = Depends(get_bundle)) -> dict:
    """Any race, loaded live, with the model's call on every lap."""
    from tire_strategy.export_demo import race_payload

    try:
        race = _live_race(year, round)
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"Could not load {year} round {round}: {exc}"
        ) from exc
    return race_payload(race, bundle)


@app.post("/api/refresh")
def api_refresh() -> dict[str, Any]:
    """Add finished races from OpenF1, refit the strategy engine and rebuild the site."""
    from tire_strategy.data.dataset import update_from_openf1
    from tire_strategy.export_demo import main as export_site

    year = datetime.now(UTC).year
    n_ds, n_laps = update_from_openf1(tuple(range(2023, year + 1)))
    export_site()
    _cache.clear()
    return {"message": f"Added {n_ds} races to the model data and {n_laps} to the strategy data; "
            "refitted the engine. Run `just train` to retrain the models on them."}  # fmt: skip


def _mount_site() -> None:
    from tire_strategy.export_demo import OUT_DIR

    if OUT_DIR.exists():
        app.mount("/", StaticFiles(directory=OUT_DIR, html=True), name="site")


_mount_site()
