"""Single-race ingestion.

Load one FastF1 race session and return a clean, typed, per-driver-per-lap
DataFrame. This module *only* cleans and flattens the raw FastF1 data — it does
not derive strategy labels (see ``labels.py``) or model features (see
``features.py``). Keeping those stages separate is what makes the leakage rules
in ``docs/DESIGN.md`` auditable: anything time-sensitive lives downstream.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import fastf1
import pandas as pd

from f1_foundation.config import enable_cache

# FastF1 encodes track status per lap as a string of concatenated single-digit
# codes (e.g. "14" means codes 1 and 4 both occurred during that lap).
_SC = "4"  # Safety Car deployed
_VSC = "6"  # Virtual Safety Car deployed
_VSC_ENDING = "7"  # VSC ending
_YELLOW = "2"  # Yellow flag
_RED = "5"  # Red flag

# Timedelta columns -> float-seconds columns (easier to model with).
_TIMEDELTA_COLS: dict[str, str] = {
    "LapTime": "lap_time_s",
    "PitInTime": "pit_in_time_s",
    "PitOutTime": "pit_out_time_s",
    "LapStartTime": "lap_start_time_s",
    "Time": "time_s",
    "Sector1Time": "sector1_s",
    "Sector2Time": "sector2_s",
    "Sector3Time": "sector3_s",
}


def _seconds(series: pd.Series) -> pd.Series:
    """Convert a timedelta Series to float seconds (NaT -> NaN)."""
    return series.dt.total_seconds()


def _contains(code: str) -> Callable[[object], bool]:
    """Build a predicate that tests whether a track-status string contains ``code``."""

    def predicate(status: object) -> bool:
        return isinstance(status, str) and code in status

    return predicate


def load_race(year: int, grand_prix: int | str, session_id: str = "R") -> pd.DataFrame:
    """Load and clean one race session into a per-driver-per-lap table.

    Parameters
    ----------
    year:
        Season, e.g. ``2024``.
    grand_prix:
        Event name or location accepted by FastF1, e.g. ``"Monza"``.
    session_id:
        Session identifier; defaults to the race (``"R"``).

    Returns
    -------
    DataFrame
        One row per lap with cleaned identifiers, timing (seconds), tyre, weather
        and track-status fields, plus boolean flags. Strategy labels and derived
        features are intentionally **not** included here.
    """
    enable_cache()
    session = fastf1.get_session(year, grand_prix, session_id)
    session.load(laps=True, weather=True, telemetry=False, messages=False)

    # Pull weather aligned to each lap *before* resetting the index (the helper
    # is a method on the FastF1 Laps object).
    laps_raw = session.laps
    try:
        weather = laps_raw.get_weather_data().reset_index(drop=True)
    except Exception:  # pragma: no cover - weather occasionally unavailable
        weather = None
    laps = laps_raw.reset_index(drop=True)

    out = pd.DataFrame(index=laps.index)

    # --- identifiers / event metadata ---
    event = session.event
    out["year"] = int(year)
    out["round"] = int(event["RoundNumber"])
    out["circuit"] = str(event["Location"])
    out["event_name"] = str(event["EventName"])
    out["session"] = session_id
    out["driver"] = laps["Driver"].astype("string")
    out["driver_number"] = laps["DriverNumber"].astype("string")
    out["team"] = laps["Team"].astype("string")

    # --- lap / stint / tyre ---
    out["lap_number"] = laps["LapNumber"].astype("Int64")
    out["stint"] = laps["Stint"].astype("Int64")
    out["compound"] = laps["Compound"].astype("string")
    out["tyre_life"] = laps["TyreLife"].astype("Int64")
    out["fresh_tyre"] = laps["FreshTyre"].astype("boolean")
    out["position"] = laps["Position"].astype("Int64")

    total_laps = int(laps["LapNumber"].max())
    out["total_laps"] = total_laps
    out["laps_remaining"] = (total_laps - laps["LapNumber"]).astype("Int64")

    # --- timing (converted to seconds) ---
    for src, dst in _TIMEDELTA_COLS.items():
        if src in laps.columns:
            out[dst] = _seconds(cast(pd.Series, laps[src]))

    # --- weather (FastF1 aligns each lap to the nearest weather sample) ---
    # track_temp + rainfall drive the strategy model (track temp is the strongest
    # wet-vs-slick discriminator; see docs §9.2). humidity / wind / pressure carry
    # no measured signal for the wet decision but are cheap to keep here for future
    # use; they are not in the feature set, so the model simply ignores them.
    if weather is not None:
        out["air_temp"] = weather["AirTemp"].astype(float)
        out["track_temp"] = weather["TrackTemp"].astype(float)
        out["rainfall"] = weather["Rainfall"].astype("boolean")
        out["humidity"] = weather["Humidity"].astype(float)
        out["pressure"] = weather["Pressure"].astype(float)
        out["wind_speed"] = weather["WindSpeed"].astype(float)
        out["wind_direction"] = weather["WindDirection"].astype(float)
    else:
        for col in (
            "air_temp",
            "track_temp",
            "rainfall",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
        ):
            out[col] = pd.NA

    # --- pit flags ---
    out["is_in_lap"] = laps["PitInTime"].notna()
    out["is_out_lap"] = laps["PitOutTime"].notna()

    # --- track-status flags ---
    status = laps["TrackStatus"].astype("string")
    out["track_status"] = status
    out["flag_yellow"] = status.map(_contains(_YELLOW)).astype("boolean")
    out["flag_sc"] = status.map(_contains(_SC)).astype("boolean")
    out["flag_vsc"] = (status.map(_contains(_VSC)) | status.map(_contains(_VSC_ENDING))).astype(
        "boolean"
    )
    out["flag_red"] = status.map(_contains(_RED)).astype("boolean")
    out["under_caution"] = (out["flag_sc"] | out["flag_vsc"]).astype("boolean")

    # --- data-quality flags (used by the cleaning step) ---
    out["is_accurate"] = laps["IsAccurate"].astype("boolean")
    out["deleted"] = laps["Deleted"].astype("boolean")

    return out


def load_results(year: int, grand_prix: int | str, session_id: str = "R") -> pd.DataFrame:
    """Load per-driver classification for one session.

    Used to identify DNFs honestly: a driver is *classified* only if their
    ``ClassifiedPosition`` is a number; codes like ``R`` (retired), ``D``
    (disqualified) or ``W`` (withdrawn) mean they did not finish.

    Returns one row per driver with ``driver``, ``status``,
    ``classified_position``, ``classified`` (bool) and ``finish_position``.
    """
    enable_cache()
    session = fastf1.get_session(year, grand_prix, session_id)
    session.load(laps=False, telemetry=False, weather=False, messages=False)

    results = session.results
    out = pd.DataFrame(index=results.index)
    out["driver"] = results["Abbreviation"].astype("string")
    out["status"] = results["Status"].astype("string")
    out["classified_position"] = results["ClassifiedPosition"].astype("string")
    out["classified"] = out["classified_position"].str.fullmatch(r"\d+").fillna(False).astype(bool)
    out["finish_position"] = cast(
        pd.Series, pd.to_numeric(results["Position"], errors="coerce")
    ).astype("Int64")
    return out.reset_index(drop=True)


def retired_drivers(results: pd.DataFrame) -> set[str]:
    """Return the set of driver codes that were not classified (DNF)."""
    not_classified = results.loc[~results["classified"], "driver"].dropna()
    return {str(code) for code in not_classified}
