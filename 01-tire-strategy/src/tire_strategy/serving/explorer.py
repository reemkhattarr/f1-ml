"""Gradio data explorer: browse every field in the project's F1 data.

Organised into logical tabs (lap timing, strategy/model data, weather, results,
telemetry, stints), each with sensible filters and a clean, readable table.
Read-only browsing of the FastF1 data behind the portfolio.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

import pandas as pd

from tire_strategy.data.dataset import DATASET_PATH
from tire_strategy.data.telemetry import TELEMETRY_PATH
from tire_strategy.serving.explorer_data import RAW_LAPS_PATH, RESULTS_PATH, WEATHER_PATH
from tire_strategy.simulator.data import STINT_PATH

_ALL = "All"

# Clean sans-serif everywhere; force table headers + cells onto a single line and
# scroll horizontally instead of wrapping a word across lines.
_CSS = """
.gradio-container, .gradio-container * {
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
    Helvetica, Arial, sans-serif !important;
}
.gradio-container table th, .gradio-container table td {
  white-space: nowrap !important;
  text-overflow: ellipsis;
  max-width: 320px;
  padding: 6px 12px !important;
  font-size: 13px !important;
}
.gradio-container table th { font-weight: 600 !important; }
.gradio-container .table-wrap,
.gradio-container div[class*="table"] { overflow-x: auto !important; }
"""


@dataclass(frozen=True)
class Section:
    name: str
    blurb: str
    path: Path
    columns: tuple[str, ...] | None  # subset to show, or None for all
    year_col: str | None = "year"
    circuit_col: str | None = "circuit"
    driver_col: str | None = "driver"
    session_col: str | None = "session"


SECTIONS: list[Section] = [
    Section(
        "Lap timing",
        "Every lap: lap + sector times, speed-trap speeds, position, pit stops, tyres.",
        RAW_LAPS_PATH,
        (
            "year",
            "circuit",
            "session",
            "Driver",
            "Team",
            "LapNumber",
            "Stint",
            "Compound",
            "TyreLife",
            "FreshTyre",
            "LapTime",
            "Sector1Time",
            "Sector2Time",
            "Sector3Time",
            "SpeedI1",
            "SpeedI2",
            "SpeedFL",
            "SpeedST",
            "Position",
            "PitInTime",
            "PitOutTime",
            "TrackStatus",
            "IsPersonalBest",
            "Deleted",
        ),
        driver_col="Driver",
    ),
    Section(
        "Strategy & model data",
        "The engineered model inputs + the pit/compound labels, per driver-lap (race only).",
        DATASET_PATH,
        None,
        session_col=None,
    ),
    Section(
        "Weather",
        "Track + air temperature, humidity, pressure, wind and rain over each session.",
        WEATHER_PATH,
        (
            "year",
            "circuit",
            "session",
            "Time",
            "AirTemp",
            "TrackTemp",
            "Humidity",
            "Pressure",
            "Rainfall",
            "WindSpeed",
            "WindDirection",
        ),
        driver_col=None,
    ),
    Section(
        "Results",
        "Finishing position, grid, status, points, laps and qualifying times per driver.",
        RESULTS_PATH,
        (
            "year",
            "circuit",
            "session",
            "Abbreviation",
            "TeamName",
            "Position",
            "ClassifiedPosition",
            "GridPosition",
            "Status",
            "Points",
            "Laps",
            "Q1",
            "Q2",
            "Q3",
        ),
        driver_col="Abbreviation",
    ),
    Section(
        "Telemetry (aggression)",
        "Per-lap driving-aggression / tyre-load aggregates from the race car-data "
        "trace (2024 sample).",
        TELEMETRY_PATH,
        None,
        circuit_col=None,
        session_col=None,
    ),
    Section(
        "Clean stint laps",
        "Clean slick race laps with lap times — the basis for the degradation model.",
        STINT_PATH,
        None,
        session_col=None,
    ),
]


@lru_cache(maxsize=16)
def _load(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str)


def _options(section: Section, column: str | None) -> list[str]:
    if column is None:
        return []
    df = _load(str(section.path))
    if column not in df.columns:
        return []
    return [str(v) for v in sorted(df[column].dropna().unique().tolist())]


def view(section: Section, year: str, circuit: str, session: str, driver: str, max_rows: int):
    """Filtered, display-friendly view of one section plus a row/column summary.

    Filtering happens on the *full* frame first; the display-column subset is
    applied only afterwards. (Subsetting first would drop the year/circuit/driver
    columns the filters key on, silently turning every filter into a no-op.)
    """
    df = _load(str(section.path)).copy()

    def _eq(col: str | None, value: str) -> None:
        nonlocal df
        if col and value not in (None, _ALL) and col in df.columns:
            df = cast(pd.DataFrame, df[df[col].astype(str) == str(value)])

    _eq(section.year_col, year)
    _eq(section.circuit_col, circuit)
    _eq(section.session_col, session)
    _eq(section.driver_col, driver)

    if section.columns:
        df = cast(pd.DataFrame, df[[c for c in section.columns if c in df.columns]])

    floats = df.select_dtypes("float").columns
    df[floats] = df[floats].round(3)
    info = f"**{len(df):,} rows · {df.shape[1]} columns** (showing up to {int(max_rows)})"
    return cast(pd.DataFrame, df.head(int(max_rows))), info


def build_explorer():  # pragma: no cover - UI wiring
    import gradio as gr

    theme = gr.themes.Soft()  # type: ignore[attr-defined]
    with gr.Blocks(theme=theme, css=_CSS, title="F1 Data Explorer") as demo:
        gr.Markdown(
            "# 🏎️ F1 Data Explorer\n"
            "The raw F1 timing data behind the portfolio. Pick a tab, then filter by "
            "season, circuit or driver."
        )
        for section in SECTIONS:
            if not section.path.exists():
                continue
            with gr.Tab(section.name):
                gr.Markdown(f"_{section.blurb}_")
                with gr.Row():
                    year = gr.Dropdown(
                        [_ALL, *_options(section, section.year_col)], value=_ALL, label="Season"
                    )
                    circuit = gr.Dropdown(
                        [_ALL, *_options(section, section.circuit_col)], value=_ALL, label="Circuit"
                    )
                    session = gr.Dropdown(
                        [_ALL, *_options(section, section.session_col)], value=_ALL, label="Session"
                    )
                    driver = gr.Dropdown(
                        [_ALL, *_options(section, section.driver_col)], value=_ALL, label="Driver"
                    )
                    max_rows = gr.Slider(50, 1000, value=200, step=50, label="Max rows")
                info = gr.Markdown()
                table = gr.Dataframe(interactive=False, wrap=False, max_height=560)

                def _update(y, c, s, d, m, _section=section):
                    return view(_section, y, c, s, d, m)

                init = view(section, _ALL, _ALL, _ALL, _ALL, 200)
                table.value, info.value = init[0], init[1]
                for ctrl in (year, circuit, session, driver, max_rows):
                    ctrl.change(_update, [year, circuit, session, driver, max_rows], [table, info])

    return demo


if __name__ == "__main__":  # pragma: no cover
    build_explorer().launch()
