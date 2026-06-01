"""Gradio data explorer: browse every field in the project's built datasets.

Loads the processed parquets and lets you filter by season / circuit / driver and
see all columns. Read-only browsing of the data behind the tire-strategy project.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

import pandas as pd

from tire_strategy.data.dataset import DATASET_PATH
from tire_strategy.data.telemetry import TELEMETRY_PATH
from tire_strategy.simulator.data import STINT_PATH

# Display name -> parquet path.
SOURCES: dict[str, Path] = {
    "Lap features + labels (main dataset)": DATASET_PATH,
    "Clean stint laps (with lap times)": STINT_PATH,
    "Telemetry features (per lap)": TELEMETRY_PATH,
}
_ALL = "All"


@lru_cache(maxsize=8)
def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(SOURCES[name])


def _options(name: str, column: str) -> list[str]:
    df = _load(name)
    if column not in df.columns:
        return []
    return [str(v) for v in sorted(df[column].dropna().unique().tolist())]


def view(
    name: str, year: str, circuit: str, driver: str, max_rows: int
) -> tuple[pd.DataFrame, str]:
    """Filtered view of one dataset plus a row x column summary."""
    df = _load(name)
    if year not in (None, _ALL) and "year" in df.columns:
        df = cast(pd.DataFrame, df[df["year"] == int(year)])
    if circuit not in (None, _ALL, "") and "circuit" in df.columns:
        df = cast(pd.DataFrame, df[df["circuit"] == circuit])
    if driver not in (None, _ALL, "") and "driver" in df.columns:
        df = cast(pd.DataFrame, df[df["driver"] == driver])
    info = f"**{len(df):,} rows x {df.shape[1]} columns** -- showing up to {int(max_rows)}."
    return cast(pd.DataFrame, df.head(int(max_rows))), info


def build_explorer():  # pragma: no cover - UI wiring
    import gradio as gr

    first = next(iter(SOURCES))

    with gr.Blocks(title="F1 Data Explorer") as demo:
        gr.Markdown(
            "# F1 Data Explorer\n"
            "Browse every field in the datasets behind the F1 tire-strategy project. "
            "Pick a dataset and filter by season, circuit, or driver."
        )
        with gr.Row():
            source = gr.Dropdown(list(SOURCES), value=first, label="Dataset")
            year = gr.Dropdown([_ALL], value=_ALL, label="Season")
            circuit = gr.Dropdown([_ALL], value=_ALL, label="Circuit")
            driver = gr.Dropdown([_ALL], value=_ALL, label="Driver")
            max_rows = gr.Slider(50, 2000, value=300, step=50, label="Max rows")
        info = gr.Markdown()
        table = gr.Dataframe(interactive=False, wrap=True)

        def _refresh_filters(name: str):
            years = [_ALL, *_options(name, "year")]
            circuits = [_ALL, *_options(name, "circuit")]
            drivers = [_ALL, *_options(name, "driver")]
            return (
                gr.update(choices=years, value=_ALL),
                gr.update(choices=circuits, value=_ALL),
                gr.update(choices=drivers, value=_ALL),
            )

        source.change(_refresh_filters, [source], [year, circuit, driver])
        for ctrl in (source, year, circuit, driver, max_rows):
            ctrl.change(view, [source, year, circuit, driver, max_rows], [table, info])
        demo.load(_refresh_filters, [source], [year, circuit, driver])
        demo.load(view, [source, year, circuit, driver, max_rows], [table, info])

    return demo


if __name__ == "__main__":  # pragma: no cover
    build_explorer().launch()
