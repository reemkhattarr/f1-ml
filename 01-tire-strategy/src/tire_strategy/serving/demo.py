"""Gradio demo: scrub through a historical race and see, at each lap, what the
model would have recommended versus what the team actually did.

Reads the **pre-built dataset parquet** (not live FastF1), so it is instant and
deployable on a free, ephemeral host without hitting FastF1's rate limit. The
data/prediction logic is separate from the Gradio UI so it can be unit-tested.
Requires the dataset (`just data`) and trained artifacts (`just train`).
"""

from __future__ import annotations

from functools import lru_cache
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd

from tire_strategy.data import load_dataset
from tire_strategy.serving.models import ModelBundle, load_bundle

plt.switch_backend("Agg")


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    return load_dataset()


def available_years() -> list[int]:
    return sorted(int(y) for y in _dataset()["year"].dropna().unique())


def circuits_for(year: int) -> list[str]:
    df = _dataset()
    return sorted(df.loc[df["year"] == year, "circuit"].dropna().unique().tolist())


def race_table(year: int, circuit: str) -> pd.DataFrame:
    df = _dataset()
    return cast(pd.DataFrame, df[(df["year"] == year) & (df["circuit"] == circuit)])


def drivers_in(year: int, circuit: str) -> list[str]:
    return sorted(race_table(year, circuit)["driver"].dropna().unique().tolist())


def lap_prediction(bundle: ModelBundle, year: int, circuit: str, driver: str, lap: int) -> dict:
    """Model recommendation vs actual outcome for one driver-lap."""
    table = race_table(year, circuit)
    row = cast(pd.DataFrame, table[(table["driver"] == driver) & (table["lap_number"] == lap)])
    if row.empty:
        return {}

    pit_proba = float(bundle.pit_model.predict_proba(row)[0, 1])
    compound_proba = bundle.compound_model.predict_proba(row)[0]
    classes = [str(c) for c in bundle.compound_model.classes_]
    recommended_compound = classes[int(compound_proba.argmax())]
    actual_compound = row["label_compound"].iloc[0]

    return {
        "pit_probability": pit_proba,
        "pit_recommended": pit_proba >= bundle.threshold,
        "recommended_compound": recommended_compound,
        "actual_pit": bool(row["label_pit"].iloc[0] == 1),
        "actual_compound": None if pd.isna(actual_compound) else str(actual_compound),
        "compound": str(row["compound"].iloc[0]),
        "tyre_age": int(row["tyre_age"].iloc[0]),
    }


def probability_curve(bundle: ModelBundle, year: int, circuit: str, driver: str):
    """Pit probability across the race for one driver, with actual pits marked."""
    table = race_table(year, circuit)
    sub = cast(pd.DataFrame, table[table["driver"] == driver]).sort_values("lap_number")
    laps = sub["lap_number"].to_numpy()
    proba = bundle.pit_model.predict_proba(sub)[:, 1]
    actual_pit_laps = sub.loc[sub["label_pit"] == 1, "lap_number"].tolist()

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(laps, proba, color="tab:blue", label="model pit probability")
    ax.axhline(bundle.threshold, ls="--", color="grey", label=f"threshold ({bundle.threshold:.2f})")
    for i, pit_lap in enumerate(actual_pit_laps):
        ax.axvline(pit_lap, color="tab:red", alpha=0.7, label="actual pit" if i == 0 else None)
    ax.set_xlabel("Lap")
    ax.set_ylabel("Pit probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"{driver} — {circuit} {year}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _format(pred: dict) -> str:
    if not pred:
        return "_No data for this lap._"
    rec = "🟢 **PIT**" if pred["pit_recommended"] else "⚪ **STAY OUT**"
    actual = f"pitted for **{pred['actual_compound']}**" if pred["actual_pit"] else "stayed out"
    return (
        f"### Model: {rec}  (p = {pred['pit_probability']:.2f})\n"
        f"- On **{pred['compound']}**, tyre age **{pred['tyre_age']}** laps\n"
        f"- If pitting, recommended compound: **{pred['recommended_compound']}**\n\n"
        f"### Actually: the team **{actual}**"
    )


def build_demo():  # pragma: no cover - UI wiring
    import gradio as gr

    bundle = load_bundle()
    years = available_years()
    default_year = years[-1]
    default_circuits = circuits_for(default_year)
    default_circuit = "Monza" if "Monza" in default_circuits else default_circuits[0]

    with gr.Blocks(title="F1 Tire Strategy") as demo:
        gr.Markdown(
            "# F1 Tire Strategy\nPick a race and driver, scrub through the laps, and "
            "compare the model's recommendation with what the team actually did."
        )
        with gr.Row():
            year = gr.Dropdown(years, value=default_year, label="Season")
            circuit = gr.Dropdown(default_circuits, value=default_circuit, label="Circuit")
            driver = gr.Dropdown([], label="Driver")
            load_btn = gr.Button("Load race", variant="primary")
        lap = gr.Slider(1, 78, step=1, value=1, label="Lap")
        out = gr.Markdown()
        plot = gr.Plot()

        def _on_year(yr):
            choices = circuits_for(yr)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def _load(yr, ci):
            drivers = drivers_in(yr, ci)
            return gr.update(choices=drivers, value=drivers[0] if drivers else None)

        def _predict(yr, ci, dr, lp):
            if not dr:
                return "_Load a race and pick a driver._", None
            pred = lap_prediction(bundle, yr, ci, dr, int(lp))
            return _format(pred), probability_curve(bundle, yr, ci, dr)

        year.change(_on_year, [year], [circuit])
        load_btn.click(_load, [year, circuit], [driver])
        for ctrl in (driver, lap):
            ctrl.change(_predict, [year, circuit, driver, lap], [out, plot])

    return demo


if __name__ == "__main__":  # pragma: no cover
    build_demo().launch()
