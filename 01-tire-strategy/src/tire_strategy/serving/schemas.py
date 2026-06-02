"""Request/response schemas for the tire-strategy API.

``RaceState`` mirrors the model's feature columns. Fields the caller always knows
(lap, tyre age, compound, conditions) are required; quantities derived from race
history (gaps, recent pace, degradation slope) are optional and default to
``None`` -> missing, which LightGBM handles natively.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from tire_strategy.serving.recommend import CompoundSource

Compound = Literal["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


class RaceState(BaseModel):
    """The live state of one car at the start of a lap."""

    # --- always known ---
    lap_number: int = Field(ge=1, description="Current lap number (1-based).")
    laps_remaining: int = Field(ge=0, description="Laps left in the race.")
    tyre_age: int = Field(ge=0, description="Laps completed on the current tyre.")
    stint_position: int = Field(ge=1, description="Lap index within the current stint.")
    pit_count_so_far: int = Field(ge=0, default=0, description="Stops made so far.")
    track_temp: float = Field(description="Track temperature (deg C).")
    air_temp: float = Field(description="Air temperature (deg C).")
    compound: Compound = Field(description="Current tyre compound.")
    circuit: str = Field(min_length=1, description="Circuit name, e.g. 'Monza'.")
    team: str = Field(
        default="UNKNOWN",
        description="Constructor, e.g. 'Ferrari'. Optional — used only by the "
        "compound head; an unknown team degrades gracefully to the other features.",
    )

    # --- flags (default to a clean, green-flag, fresh-tyre state) ---
    fresh_tyre: bool = True
    under_caution: bool = False
    flag_sc: bool = False
    flag_vsc: bool = False
    flag_yellow: bool = False
    used_second_compound: bool = False
    rainfall: bool = False
    sc_just_deployed: bool = Field(default=False, description="The safety car came out this lap.")

    # --- derived from history (optional; None -> unknown) ---
    gap_ahead_s: float | None = Field(default=None, description="Gap to car ahead (s).")
    gap_behind_s: float | None = Field(default=None, description="Gap to car behind (s).")
    last_clean_lap_time_s: float | None = Field(default=None, description="Last clean lap (s).")
    deg_slope_s_per_lap: float | None = Field(
        default=None, description="Fuel-corrected degradation slope (s/lap)."
    )
    relative_pace_s: float | None = Field(
        default=None, description="Recent lap vs the field median that lap (s)."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "lap_number": 14,
                    "laps_remaining": 39,
                    "tyre_age": 14,
                    "stint_position": 14,
                    "pit_count_so_far": 0,
                    "track_temp": 48.0,
                    "air_temp": 33.0,
                    "compound": "MEDIUM",
                    "circuit": "Monza",
                    "team": "Ferrari",
                    "gap_ahead_s": 2.9,
                    "gap_behind_s": 1.0,
                    "last_clean_lap_time_s": 85.5,
                    "deg_slope_s_per_lap": 0.28,
                    "relative_pace_s": 0.4,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """The model's strategy recommendation for one race state."""

    pit_probability: float = Field(ge=0.0, le=1.0, description="P(pit within the next few laps).")
    pit_recommended: bool = Field(description="True if pit_probability >= decision_threshold.")
    decision_threshold: float = Field(description="Cost-optimised threshold used.")
    recommended_compound: str = Field(description="Most likely compound if pitting.")
    compound_probabilities: dict[str, float] = Field(
        description="Probability per compound if pitting. Slick classes "
        "(SOFT/MEDIUM/HARD) from the ML head in the dry; a single wet tyre at "
        "probability 1.0 when the rain gate fires."
    )
    compound_source: CompoundSource = Field(
        description="How the compound was chosen: 'model' (slick ML head, dry) or "
        "'wet-rule' (deterministic rain gate — wet-tyre choice is not learnable "
        "from the available features)."
    )


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
