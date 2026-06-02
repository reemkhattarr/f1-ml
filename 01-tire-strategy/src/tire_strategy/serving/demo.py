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

from tire_strategy.config import DATA_DIR
from tire_strategy.data import load_dataset
from tire_strategy.serving.models import ModelBundle, load_bundle
from tire_strategy.serving.recommend import recommend_compound

plt.switch_backend("Agg")

_RESULTS_PATH = DATA_DIR / "processed" / "results.parquet"


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    return load_dataset()


@lru_cache(maxsize=1)
def _results() -> pd.DataFrame | None:
    return pd.read_parquet(_RESULTS_PATH) if _RESULTS_PATH.exists() else None


def race_results(year: int, circuit: str) -> pd.DataFrame:
    """Final classification for a race, ordered by finishing position."""
    res = _results()
    if res is None:
        return pd.DataFrame()
    df = cast(pd.DataFrame, res[(res["year"] == year) & (res["circuit"] == circuit)]).copy()
    df["_pos"] = cast(pd.Series, pd.to_numeric(df["Position"], errors="coerce"))
    df = df.sort_values("_pos")
    df["Position"] = df["_pos"].astype("Int64")
    if "GridPosition" in df.columns:
        df["Grid"] = cast(pd.Series, pd.to_numeric(df["GridPosition"], errors="coerce")).astype(
            "Int64"
        )
    rename = {"Abbreviation": "Driver", "TeamName": "Team"}
    keep = ["Position", "Abbreviation", "TeamName", "Grid", "Status", "Points"]
    return cast(pd.DataFrame, df[[c for c in keep if c in df.columns]]).rename(columns=rename)


def race_notes(year: int, circuit: str) -> str:
    """A short, auto-derived summary of events that shaped the race's strategy."""
    table = race_table(year, circuit)
    if table.empty:
        return ""
    total_laps = int(table["lap_number"].max())
    by_lap = table.groupby("lap_number")
    sc_laps = int(by_lap["under_caution"].apply(lambda s: bool(s.fillna(False).any())).sum())
    red = bool(table["flag_red"].fillna(False).any()) if "flag_red" in table else False
    wet_by_lap = cast(pd.Series, by_lap["rainfall"].apply(lambda s: bool(s.fillna(False).any())))

    notes = [f"**{circuit} {year}** — {total_laps} laps."]
    notes.append(
        f"🟡 Safety car / VSC on ~{sc_laps} lap(s) — a cheap-stop window."
        if sc_laps
        else "🟢 Green flag-to-flag (no safety car)."
    )
    if red:
        notes.append("🔴 Race was red-flagged (free tyre change).")
    # Surface mid-race condition changes — a major strategy driver.
    if not bool(wet_by_lap.any()):
        notes.append("☀️ Dry throughout.")
    elif bool(wet_by_lap.all()):
        notes.append("🌧️ Wet throughout.")
    elif not bool(wet_by_lap.iloc[0]):
        first_pos = int(wet_by_lap.to_numpy().argmax())  # first True
        first_wet = int(wet_by_lap.index.to_numpy()[first_pos])
        notes.append(f"🌧️ Rain arrived mid-race (~lap {first_wet}) — a big strategy swing.")
    else:
        wet_laps = int(wet_by_lap.to_numpy().sum())
        notes.append(
            f"🌦️ Mixed conditions — track dried during the race (wet for ~{wet_laps} laps)."
        )

    res = _results()
    if res is not None:
        rr = cast(pd.DataFrame, res[(res["year"] == year) & (res["circuit"] == circuit)])
        status = cast(pd.Series, rr["Status"]).astype(str)
        dsq = rr.loc[status.str.contains("Disq", case=False, na=False), "Abbreviation"]
        if len(dsq):
            notes.append(f"⚠️ Disqualified: {', '.join(dsq.astype(str))}.")
    return "  \n".join(notes)


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
    rainfall = bool(row["rainfall"].fillna(False).astype(bool).iloc[0])
    compound = recommend_compound(bundle.compound_model, row, rainfall=rainfall)
    actual_compound = row["label_compound"].iloc[0]

    return {
        "pit_probability": pit_proba,
        "pit_recommended": pit_proba >= bundle.threshold,
        "recommended_compound": compound.compound,
        "compound_source": compound.source,
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


# F1 tyre-compound colours (slicks + wets) for the strategy timeline.
_COMPOUND_COLORS = {
    "SOFT": "#e8413b",
    "MEDIUM": "#f5d23a",
    "HARD": "#dcdcdc",
    "INTERMEDIATE": "#43a047",
    "WET": "#1e6fd6",
    "UNKNOWN": "#999999",
}


def strategy_timeline(year: int, circuit: str, driver: str):
    """The driver's actual tyre strategy: each stint as a compound-coloured bar
    across the laps, pit stops at the joins, and any wet laps banded on top."""
    table = race_table(year, circuit)
    sub = cast(pd.DataFrame, table[table["driver"] == driver]).sort_values("lap_number")
    if sub.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 1.9))
    seen: set[str] = set()
    for _, grp in sub.groupby("stint"):
        comp = str(grp["compound"].iloc[0])
        start, end = int(grp["lap_number"].min()), int(grp["lap_number"].max())
        ax.barh(
            0,
            end - start + 1,
            left=start - 0.5,
            height=0.6,
            color=_COMPOUND_COLORS.get(comp, "#999999"),
            edgecolor="black",
            linewidth=0.6,
            label=comp if comp not in seen else None,
        )
        seen.add(comp)
        ax.text((start + end) / 2, 0, comp[0], ha="center", va="center", fontsize=9)

    rain_laps = sub.loc[sub["rainfall"].fillna(False).astype(bool), "lap_number"].tolist()
    for i, lap in enumerate(rain_laps):
        ax.axvspan(
            lap - 0.5,
            lap + 0.5,
            ymin=0.82,
            ymax=0.98,
            color="#1e6fd6",
            alpha=0.5,
            label="rain" if i == 0 else None,
        )

    ax.set_ylim(-0.6, 1.0)
    ax.set_yticks([])
    ax.set_xlabel("Lap")
    ax.set_title(f"{driver} tyre strategy — {circuit} {year}")
    ax.legend(loc="upper right", fontsize=7, ncol=6)
    fig.tight_layout()
    return fig


def _format(pred: dict) -> str:
    if not pred:
        return "_No data for this lap._"
    rec = "🟢 **PIT**" if pred["pit_recommended"] else "⚪ **STAY OUT**"
    actual = f"pitted for **{pred['actual_compound']}**" if pred["actual_pit"] else "stayed out"
    compound_note = " _(rain rule)_" if pred.get("compound_source") == "wet-rule" else ""
    return (
        f"### Model: {rec}  (p = {pred['pit_probability']:.2f})\n"
        f"- On **{pred['compound']}**, tyre age **{pred['tyre_age']}** laps\n"
        f"- If pitting, recommended compound: **{pred['recommended_compound']}**{compound_note}\n\n"
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
        notes = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=2):
                lap = gr.Slider(1, 78, step=1, value=1, label="Lap")
                out = gr.Markdown()
                plot = gr.Plot()
                timeline = gr.Plot()
            with gr.Column(scale=1):
                gr.Markdown("### Final classification")
                results = gr.Dataframe(interactive=False, wrap=True, max_height=520)

        def _on_year(yr):
            choices = circuits_for(yr)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def _load(yr, ci):
            drivers = drivers_in(yr, ci)
            return (
                gr.update(choices=drivers, value=drivers[0] if drivers else None),
                race_notes(yr, ci),
                race_results(yr, ci),
            )

        def _predict(yr, ci, dr, lp):
            if not dr:
                return "_Load a race and pick a driver._", None, None
            pred = lap_prediction(bundle, yr, ci, dr, int(lp))
            return (
                _format(pred),
                probability_curve(bundle, yr, ci, dr),
                strategy_timeline(yr, ci, dr),
            )

        year.change(_on_year, [year], [circuit])
        load_btn.click(_load, [year, circuit], [driver, notes, results])
        for ctrl in (driver, lap):
            ctrl.change(_predict, [year, circuit, driver, lap], [out, plot, timeline])

    return demo


if __name__ == "__main__":  # pragma: no cover
    build_demo().launch()
