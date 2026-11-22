"""Fit the strategy engine's per-circuit parameters from raw race laps.

Everything the planner and advisor need, measured from real races:

- **degradation** (s/lap of tyre age) per circuit and compound, from within-stint
  slopes of fuel-corrected lap time (the car is constant inside a stint);
- **compound pace offsets** (how much faster a fresh soft is than a fresh medium),
  from the same driver's different stints in the same race, corrected for the
  track getting faster as rubber goes down;
- **pit loss** per circuit, from in-lap + out-lap time above the driver's green pace;
- **tyre life**: the 90th percentile of real stint lengths per circuit and compound,
  used as the longest stint the planner will propose;
- **safety-car odds**: share of races with a safety car or VSC, and how often one
  starts per lap, shrunk toward the all-circuit average when a circuit has few races;
- race distance and typical track temperature.

Recent seasons count most: per circuit the fit uses the latest two seasons with
data there, so a new regulation era (2026) takes over as soon as it has been raced.
Two values are assumptions rather than measurements and are labelled as such in the
output: how much cheaper a stop is under a safety car or a VSC.
"""

from __future__ import annotations

import json
import math
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

PARAMS_PATH: Path = DATA_DIR / "processed" / "strategy_params.json"

SC_PIT_FACTOR = 0.5  # assumption: a stop under a safety car costs about half
VSC_PIT_FACTOR = 0.65  # assumption: under a VSC the field slows less, so less saving
_MIN_AGES = 5
_MIN_STINTS = 8  # stint slopes needed to trust a circuit/compound degradation rate
_MIN_PAIRS = 4  # driver stint pairs needed to trust a circuit compound offset
_SHRINK = 3.0  # pseudo-races pulling a circuit's safety-car odds to the average
_DEG_RANGE = (-0.05, 0.6)  # plausible s/lap of tyre age
_OFFSET_RANGE = (-2.5, 2.5)


def _stint_fit(stint: pd.DataFrame) -> tuple[float, float] | None:
    """(intercept at age 1, slope) of fuel-corrected time vs tyre age, via medians."""
    usable = cast(pd.DataFrame, stint[stint["tyre_age"] >= 2])
    medians = usable.groupby("tyre_age")["_corrected"].median()
    if len(medians) < _MIN_AGES:
        return None
    x = medians.index.to_numpy(float)
    slope, intercept = np.polyfit(x, medians.to_numpy(float), 1)
    if not (_DEG_RANGE[0] < slope < _DEG_RANGE[1]):
        return None
    return float(intercept + slope * 1.0), float(slope)


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


def _stint_table(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in clean.groupby(["year", "round", "driver", "stint"], sort=False):
        year, rnd, driver, stint = cast(tuple[Any, Any, Any, Any], key)
        g = cast(pd.DataFrame, group)
        fit = _stint_fit(g)
        if fit is None:
            continue
        rows.append(
            {
                "year": int(year),
                "round": int(rnd),
                "driver": driver,
                "stint": int(stint),
                "circuit": g["circuit"].iloc[0],
                "compound": g["compound"].iloc[0],
                "intercept": fit[0],
                "slope": fit[1],
                "start_lap": int(g["lap_number"].min() - g["tyre_age"].min() + 1),
            }
        )
    return pd.DataFrame(rows)


def _evolution_rate(stints: pd.DataFrame) -> float:
    """Track gets faster per race lap (s/lap), from same-compound stint pairs."""
    rates = []
    for _, g in stints.groupby(["year", "round", "driver"]):
        recs: list[Any] = list(cast(pd.DataFrame, g).sort_values("start_lap").itertuples())
        for i, a in enumerate(recs):
            for b in recs[i + 1 :]:
                gap = b.start_lap - a.start_lap
                if a.compound == b.compound and gap >= 8:
                    rates.append((b.intercept - a.intercept) / gap)
    if len(rates) < 10:
        return 0.0
    return float(np.clip(np.median(rates), -0.08, 0.0))


def _offset_pairs(stints: pd.DataFrame, evolution: float) -> pd.DataFrame:
    """Per driver/race, the pace gap between two compounds, evolution-corrected."""
    rows = []
    for _, g in stints.groupby(["year", "round", "driver"]):
        recs: list[Any] = list(cast(pd.DataFrame, g).itertuples())
        for i, a in enumerate(recs):
            for b in recs[i + 1 :]:
                if a.compound == b.compound:
                    continue
                diff = (b.intercept - a.intercept) - evolution * (b.start_lap - a.start_lap)
                rows.append(
                    {"year": a.year, "circuit": a.circuit, "a": a.compound, "b": b.compound,
                     "diff": diff}
                )  # fmt: skip
    return pd.DataFrame(rows, columns=pd.Index(["year", "circuit", "a", "b", "diff"]))


def _solve_offsets(pairs: pd.DataFrame) -> dict[str, float] | None:
    """Offsets vs MEDIUM from pairwise gaps (least squares on the pair medians)."""
    idx = {"SOFT": 0, "HARD": 1}
    rows, rhs, weights = [], [], []
    for key, g in pairs.groupby(["a", "b"]):
        a, b = cast(tuple[str, str], key)
        if len(g) < _MIN_PAIRS:
            continue
        row = [0.0, 0.0]
        if b in idx:
            row[idx[b]] += 1
        if a in idx:
            row[idx[a]] -= 1
        rows.append(row)
        rhs.append(float(g["diff"].median()))
        weights.append(math.sqrt(len(g)))
    if not rows:
        return None
    A = np.array(rows) * np.array(weights)[:, None]
    y = np.array(rhs) * np.array(weights)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    rank = np.linalg.matrix_rank(A)
    if rank < 2:
        return None
    soft, hard = (float(np.clip(v, *_OFFSET_RANGE)) for v in sol)
    return {"SOFT": round(soft, 3), "MEDIUM": 0.0, "HARD": round(hard, 3)}


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


def fit_strategy_params(race_laps: pd.DataFrame) -> dict[str, Any]:
    """Fit all circuit parameters; returns a JSON-serialisable dict."""
    race_laps = race_laps.copy()
    race_laps["circuit"] = race_laps["circuit"].map(canonical_circuit)
    clean = clean_laps(race_laps)
    stints = _stint_table(clean)
    evolution = _evolution_rate(stints) if not stints.empty else 0.0
    if not stints.empty:
        # Inside a stint the track also gets faster; take that out of the tyre slope.
        stints["slope"] = stints["slope"] - evolution
    pairs = _offset_pairs(stints, evolution) if not stints.empty else pd.DataFrame()
    lengths = _stint_lengths(race_laps)
    caution = _caution_stats(race_laps)

    losses: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for key, race in race_laps.groupby(["year", "round"]):
        year = cast(tuple[Any, Any], key)[0]
        r = cast(pd.DataFrame, race)
        for loss in race_pit_losses(r):
            losses[str(r["circuit"].iloc[0])].append((int(year), loss))

    # ---- all-circuit defaults (latest two seasons) ----
    recent_stints = _recent(stints)
    default_deg = {
        c: round(float(g["slope"].median()), 4) for c, g in recent_stints.groupby("compound")
    }
    for c, fallback in (("SOFT", 0.09), ("MEDIUM", 0.06), ("HARD", 0.05)):
        default_deg.setdefault(c, fallback)
    default_offsets = _solve_offsets(_recent(pairs)) if not pairs.empty else None
    default_offsets = default_offsets or {"SOFT": -0.5, "MEDIUM": 0.0, "HARD": 0.35}
    recent_lengths = _recent(lengths)
    default_life = {
        c: int(max(8, round(float(np.percentile(g["laps"], 90)))))
        for c, g in recent_lengths.groupby("compound")
    }
    for c, fallback in (("SOFT", 22), ("MEDIUM", 32), ("HARD", 42)):
        default_life.setdefault(c, fallback)
    all_losses = [v for vals in losses.values() for _, v in vals]
    default_pit = round(float(np.median(all_losses)), 1) if all_losses else DEFAULT_PIT_LOSS
    p_caution_all = float(caution["any_caution"].mean()) if not caution.empty else 0.5
    p_sc_all = float(caution["any_sc"].mean()) if not caution.empty else 0.35
    hazard_all = (
        float(caution["starts"].sum() / caution["laps"].sum()) if not caution.empty else 0.012
    )

    circuits: dict[str, Any] = {}
    for circuit in sorted(caution["circuit"].unique()) if not caution.empty else []:
        cs = (
            _recent(cast(pd.DataFrame, stints[stints["circuit"] == circuit]))
            if not stints.empty
            else stints
        )
        deg = dict(default_deg)
        deg_src = dict.fromkeys(deg, "all circuits")
        for c, g in cs.groupby("compound") if not cs.empty else []:
            if len(g) >= _MIN_STINTS:
                deg[str(c)] = round(float(g["slope"].median()), 4)
                deg_src[str(c)] = f"{len(g)} stints"
        cp = (
            _recent(cast(pd.DataFrame, pairs[pairs["circuit"] == circuit]))
            if not pairs.empty
            else pairs
        )
        offsets = _solve_offsets(cp) if not cp.empty else None
        offsets_src = f"{len(cp)} stint pairs" if offsets else "all circuits"
        offsets = offsets or dict(default_offsets)
        cl = _recent(cast(pd.DataFrame, lengths[lengths["circuit"] == circuit]))
        life = dict(default_life)
        for c, g in cl.groupby("compound"):
            if len(g) >= 6:
                life[str(c)] = int(max(8, round(float(np.percentile(g["laps"], 90)))))
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
            "degradation_source": deg_src,
            "offsets": offsets,
            "offsets_source": offsets_src,
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

    return {
        "version": 1,
        "fuel_s_per_lap": FUEL_S_PER_LAP,
        "track_evolution_s_per_lap": round(evolution, 4),
        "sc_pit_factor": SC_PIT_FACTOR,
        "vsc_pit_factor": VSC_PIT_FACTOR,
        "assumptions": {
            "sc_pit_factor": "A stop under a safety car costs about half a green-flag stop.",
            "vsc_pit_factor": "A stop under a VSC costs about 65% of a green-flag stop.",
        },
        "seasons": sorted(int(y) for y in race_laps["year"].unique()),
        "default": {
            "degradation": default_deg,
            "offsets": default_offsets,
            "tyre_life": default_life,
            "pit_loss": default_pit,
            "p_caution": round(p_caution_all, 3),
            "p_safety_car": round(p_sc_all, 3),
            "caution_per_lap": round(hazard_all, 4),
        },
        "circuits": circuits,
    }


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
    save_params(params)
    print(f"Strategy parameters for {len(params['circuits'])} circuits -> {PARAMS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
