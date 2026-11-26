"""Fit the strategy engine's per-circuit parameters from raw race laps.

Everything the planner and advisor need, measured from real races:

- **degradation** (s/lap of tyre age) per circuit and compound, from a regression
  on every clean lap inside each driver's race (see :mod:`.pace`), pulled toward
  the all-circuit fit when a circuit has little data;
- **compound pace offsets** (how much faster a fresh soft is than a fresh medium):
  races measure these poorly because drivers manage pace to the strategy, so the
  engine uses a standard estimate moved halfway toward what the latest season's
  races show, and keeps soft faster than medium faster than hard;
- **pit loss** per circuit, from in-lap + out-lap time above the driver's green pace;
- **tyre life**: the 90th percentile of real stint lengths per circuit and compound,
  used as the longest stint the planner will propose;
- **safety-car odds**: share of races with a safety car or VSC, and how often one
  starts per lap, shrunk toward the all-circuit average when a circuit has few races;
- race distance and typical track temperature.

Recent data counts most: the all-circuit fit uses the latest season with enough
races, and each circuit its two latest races, so a new regulation era (2026) takes
over as soon as it has been raced. Assumptions rather than measurements are
labelled as such in the output: the standard compound gaps, and how much cheaper a
stop is under a safety car or a VSC.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from f1_foundation.circuits import canonical_circuit
from tire_strategy.config import DATA_DIR
from tire_strategy.data.features import FUEL_S_PER_LAP, SLICK_COMPOUNDS
from tire_strategy.simulator.data import clean_slick_mask
from tire_strategy.simulator.pitloss import DEFAULT_PIT_LOSS, race_pit_losses
from tire_strategy.strategy.pace import MIN_STEP, PaceFit, fit_pace

PARAMS_PATH: Path = DATA_DIR / "processed" / "strategy_params.json"

SC_PIT_FACTOR = 0.5  # assumption: a stop under a safety car costs about half
VSC_PIT_FACTOR = 0.65  # assumption: under a VSC the field slows less, so less saving
_SHRINK = 3.0  # pseudo-races pulling a circuit's safety-car odds to the average
# Assumption: standard race-pace gap of a fresh soft / hard to a fresh medium (s/lap).
STANDARD_OFFSETS = {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.3}
OFFSET_DATA_WEIGHT = 0.5  # how far the measured gaps move the standard estimate
CIRCUIT_PRIOR_LAPS = 400.0  # a circuit's wear estimate counts the all-circuit fit as this many laps
_MIN_SEASON_RACES = 8
# Races where Pirelli capped stint length for safety: the history there reflects the
# rule, not free choice, and the cap is announced race by race.
CIRCUIT_NOTES = {
    "Lusail": "Pirelli has capped stint length here before (18 laps in 2023, 25 in 2025), "
    "forcing extra stops. If Pirelli announces a cap this year, enter it as the stint cap "
    "in the planner.",
}


def clean_laps(race_laps: pd.DataFrame) -> pd.DataFrame:
    """Representative slick laps: green flag, timed, no pit laps, no lap 1, no outliers."""
    mask = clean_slick_mask(race_laps.reset_index(drop=True)).to_numpy() & (
        race_laps["lap_number"].to_numpy() > 1
    )
    df = race_laps.reset_index(drop=True).loc[mask].rename(columns={"tyre_life": "tyre_age"})
    df = df[df["tyre_age"].notna() & df["laps_remaining"].notna()].copy()
    if df.empty:
        return df
    df["tyre_age"] = df["tyre_age"].astype(int)
    df["circuit"] = df["circuit"].map(canonical_circuit)
    # Traffic, incidents and slow laps: drop anything 7% above the driver's median.
    med = df.groupby(["year", "round", "driver"])["lap_time_s"].transform("median")
    df = df[df["lap_time_s"] <= med * 1.07].copy()
    df["_corrected"] = df["lap_time_s"] - FUEL_S_PER_LAP * df["laps_remaining"].astype(float)
    return df


def _recent(df: pd.DataFrame, seasons: int = 2) -> pd.DataFrame:
    """Rows from the latest ``seasons`` seasons present in ``df``."""
    if df.empty:
        return df
    years = sorted(df["year"].unique())[-seasons:]
    return cast(pd.DataFrame, df[df["year"].isin(years)])


def _stint_lengths(race_laps: pd.DataFrame) -> pd.DataFrame:
    """Length of every completed dry stint (not the one a driver retired in)."""
    df = cast(pd.DataFrame, race_laps[race_laps["compound"].isin(SLICK_COMPOUNDS)])
    g = df.groupby(["year", "round", "circuit", "driver", "stint", "compound"])
    out = g.agg(laps=("lap_number", "count"), last=("lap_number", "max")).reset_index()
    finish = cast(pd.Series, race_laps.groupby(["year", "round"])["lap_number"].max())
    out = out.merge(finish.rename("race_laps"), on=["year", "round"])
    pit_laps = cast(pd.DataFrame, race_laps[race_laps["is_in_lap"].astype(bool)])
    ended_in_pit = cast(pd.DataFrame, pit_laps[["year", "round", "driver", "lap_number"]])
    ended_in_pit = ended_in_pit.rename(columns={"lap_number": "last"})
    ended_in_pit["pitted"] = True
    out = out.merge(ended_in_pit, on=["year", "round", "driver", "last"], how="left")
    complete = out["pitted"].eq(True) | (out["last"] == out["race_laps"])
    out = cast(pd.DataFrame, out[complete]).copy()
    out["circuit"] = out["circuit"].map(canonical_circuit)
    return cast(pd.DataFrame, out)


def _caution_stats(race_laps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, race in race_laps.groupby(["year", "round"]):
        year, rnd = cast(tuple[Any, Any], key)
        r = cast(pd.DataFrame, race)
        lead = r.groupby("lap_number").first()
        sc = lead["flag_sc"].fillna(False).astype(bool)
        vsc = lead["flag_vsc"].fillna(False).astype(bool)
        caution = sc | vsc
        starts = int((caution & ~caution.shift(1, fill_value=False)).sum())
        rows.append(
            {
                "year": int(year),
                "round": int(rnd),
                "circuit": canonical_circuit(r["circuit"].iloc[0]),
                "laps": int(r["lap_number"].max()),
                "any_sc": bool(sc.any()),
                "any_caution": bool(caution.any()),
                "starts": starts,
                "track_temp": float(
                    cast(pd.Series, pd.to_numeric(r["track_temp"], errors="coerce")).median()
                ),
                "rain_share": float(r["rainfall"].fillna(False).astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _latest_races(df: pd.DataFrame, n: int) -> pd.DataFrame:
    keys = cast(pd.DataFrame, df[["year", "round"]].drop_duplicates())
    keys = keys.sort_values(by=["year", "round"]).tail(n)
    return cast(pd.DataFrame, df.merge(keys, on=["year", "round"]))


def _global_pace(clean: pd.DataFrame) -> tuple[PaceFit, int]:
    """All-circuit fit on the latest season with enough races (falls back to all data)."""
    races = clean[["year", "round"]].drop_duplicates()
    counts = races.groupby("year").size()
    eligible = [int(cast(int, y)) for y, n in counts.items() if n >= _MIN_SEASON_RACES]
    season = max(eligible) if eligible else None
    data = cast(pd.DataFrame, clean[clean["year"] == season]) if season else clean
    return fit_pace(data, prior_laps=50.0), season or 0


def _blend_offsets(measured: dict[str, float]) -> dict[str, float]:
    w = OFFSET_DATA_WEIGHT
    soft = (1 - w) * STANDARD_OFFSETS["SOFT"] + w * measured["SOFT"]
    hard = (1 - w) * STANDARD_OFFSETS["HARD"] + w * measured["HARD"]
    return {
        "SOFT": round(min(soft, -MIN_STEP), 3),
        "MEDIUM": 0.0,
        "HARD": round(max(hard, MIN_STEP), 3),
    }


def _ordered_life(life: dict[str, int]) -> dict[str, int]:
    """A softer tyre never runs longer than a harder one."""
    out = dict(life)
    out["SOFT"] = min(out["SOFT"], out["MEDIUM"])
    out["HARD"] = max(out["HARD"], out["MEDIUM"])
    return out


def fit_strategy_params(race_laps: pd.DataFrame) -> dict[str, Any]:
    """Fit all circuit parameters; returns a JSON-serialisable dict."""
    race_laps = race_laps.copy()
    race_laps["circuit"] = race_laps["circuit"].map(canonical_circuit)
    clean = clean_laps(race_laps)
    lengths = _stint_lengths(race_laps)
    caution = _caution_stats(race_laps)

    losses: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for key, race in race_laps.groupby(["year", "round"]):
        year = cast(tuple[Any, Any], key)[0]
        r = cast(pd.DataFrame, race)
        for loss in race_pit_losses(r):
            losses[str(r["circuit"].iloc[0])].append((int(year), loss))

    # ---- all-circuit defaults ----
    global_fit, pace_season = _global_pace(clean)
    offsets = _blend_offsets(global_fit.offsets if global_fit.laps else STANDARD_OFFSETS)
    prior = global_fit.coef.copy()
    prior[1] = offsets["SOFT"]
    prior[2] = offsets["HARD"]
    # Circuits estimate trend and wear; the compound gaps stay the all-circuit ones.
    strength = np.array([CIRCUIT_PRIOR_LAPS, 1e9, 1e9, *([CIRCUIT_PRIOR_LAPS] * 3)])
    default_deg = dict(global_fit.wear)
    recent_lengths = _recent(lengths)
    default_life: dict[str, int] = {
        str(c): int(max(8, round(float(np.percentile(g["laps"], 90)))))
        for c, g in recent_lengths.groupby("compound")
    }
    for c, fallback in (("SOFT", 22), ("MEDIUM", 32), ("HARD", 42)):
        default_life.setdefault(c, fallback)
    default_life = _ordered_life(default_life)
    all_losses = [v for vals in losses.values() for _, v in vals]
    default_pit = round(float(np.median(all_losses)), 1) if all_losses else DEFAULT_PIT_LOSS
    p_caution_all = float(caution["any_caution"].mean()) if not caution.empty else 0.5
    p_sc_all = float(caution["any_sc"].mean()) if not caution.empty else 0.35
    hazard_all = (
        float(caution["starts"].sum() / caution["laps"].sum()) if not caution.empty else 0.012
    )

    circuits: dict[str, Any] = {}
    for circuit in sorted(caution["circuit"].unique()) if not caution.empty else []:
        cc_laps = (
            cast(pd.DataFrame, clean[clean["circuit"] == circuit]) if not clean.empty else clean
        )
        fit = (
            fit_pace(_latest_races(cc_laps, 2), prior=prior, prior_laps=strength)
            if len(cc_laps)
            else None
        )
        deg = dict(fit.wear) if fit and fit.laps else dict(default_deg)
        cl = _recent(cast(pd.DataFrame, lengths[lengths["circuit"] == circuit]))
        life = dict(default_life)
        for c, g in cl.groupby("compound"):
            if len(g) >= 6:
                life[str(c)] = int(max(8, round(float(np.percentile(g["laps"], 90)))))
        life = _ordered_life(life)
        circuit_losses = losses.get(circuit, [])
        if circuit_losses:
            latest = sorted({y for y, _ in circuit_losses})[-2:]
            vals = [v for y, v in circuit_losses if y in latest]
        else:
            vals = []
        pit_loss = round(float(np.median(vals)), 1) if len(vals) >= 5 else default_pit
        cc = cast(pd.DataFrame, caution[caution["circuit"] == circuit])
        n = len(cc)
        p_caution = (cc["any_caution"].sum() + _SHRINK * p_caution_all) / (n + _SHRINK)
        p_sc = (cc["any_sc"].sum() + _SHRINK * p_sc_all) / (n + _SHRINK)
        hazard = (cc["starts"].sum() + _SHRINK * hazard_all * 60) / (
            cc["laps"].sum() + _SHRINK * 60
        )
        latest_race = cc.sort_values(["year", "round"]).iloc[-1]
        circuits[circuit] = {
            "race_laps": int(latest_race["laps"]),
            "degradation": deg,
            "degradation_source": f"{fit.laps} laps, {fit.drivers} driver-races"
            if fit and fit.laps
            else "all circuits",
            "offsets": dict(offsets),
            "offsets_source": "standard estimate adjusted by data",
            "trend_s_per_lap": fit.trend if fit and fit.laps else global_fit.trend,
            "limits_applied": fit.clamped if fit else [],
            "tyre_life": life,
            "pit_loss": pit_loss,
            "pit_loss_source": f"{len(vals)} stops" if len(vals) >= 5 else "all circuits",
            "p_caution": round(float(p_caution), 3),
            "p_safety_car": round(float(p_sc), 3),
            "caution_per_lap": round(float(hazard), 4),
            "races": int(n),
            "seasons": sorted(int(y) for y in cc["year"].unique()),
            "track_temp": round(float(cc["track_temp"].median()), 1)
            if bool(cc["track_temp"].notna().any())
            else None,
            "wet_races": int((cc["rain_share"] > 0.3).sum()),
        }
        if circuit in CIRCUIT_NOTES:
            circuits[circuit]["note"] = CIRCUIT_NOTES[circuit]

    return {
        "version": 2,
        "fuel_s_per_lap": FUEL_S_PER_LAP,
        "track_evolution_s_per_lap": global_fit.trend,
        "sc_pit_factor": SC_PIT_FACTOR,
        "vsc_pit_factor": VSC_PIT_FACTOR,
        "pace": {
            "season": pace_season,
            "laps": global_fit.laps,
            "measured_offsets": global_fit.offsets,
            "standard_offsets": STANDARD_OFFSETS,
            "limits_applied": global_fit.clamped,
        },
        "assumptions": {
            "offsets": "Compound pace gaps: a standard estimate (soft -0.4 s, hard +0.3 s "
            "vs medium) moved halfway toward what the latest season's races show.",
            "sc_pit_factor": "A stop under a safety car costs about half a green-flag stop.",
            "vsc_pit_factor": "A stop under a VSC costs about 65% of a green-flag stop.",
        },
        "seasons": sorted(int(y) for y in race_laps["year"].unique()),
        "default": {
            "degradation": default_deg,
            "offsets": offsets,
            "tyre_life": default_life,
            "pit_loss": default_pit,
            "p_caution": round(p_caution_all, 3),
            "p_safety_car": round(p_sc_all, 3),
            "caution_per_lap": round(hazard_all, 4),
        },
        "circuits": circuits,
    }


def check_params(params: dict[str, Any]) -> list[str]:
    """Physical sanity checks; returns the problems found (empty when all is well)."""
    problems = []
    for name, c in [("default", params["default"]), *params["circuits"].items()]:
        off, deg, life = c["offsets"], c["degradation"], c["tyre_life"]
        if not off["SOFT"] < off["MEDIUM"] < off["HARD"]:
            problems.append(f"{name}: compounds not ordered soft < medium < hard in pace")
        if any(v <= 0 or v > 0.6 for v in deg.values()):
            problems.append(f"{name}: tyre wear outside (0, 0.6] s/lap")
        if not life["SOFT"] <= life["MEDIUM"] <= life["HARD"]:
            problems.append(f"{name}: a softer tyre lasts longer than a harder one")
        if not 10 <= c["pit_loss"] <= 45:
            problems.append(f"{name}: pit loss {c['pit_loss']} s is implausible")
    return problems


def save_params(params: dict[str, Any], path: Path = PARAMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=1, ensure_ascii=False))


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> int:
    import logging

    from tire_strategy.data.dataset import RACE_LAPS_PATH

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    params = fit_strategy_params(pd.read_parquet(RACE_LAPS_PATH))
    problems = check_params(params)
    for problem in problems:
        print(f"PROBLEM: {problem}")
    save_params(params)
    print(f"Strategy parameters for {len(params['circuits'])} circuits -> {PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
