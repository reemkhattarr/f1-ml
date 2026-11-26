"""How good are the engine's plans? Backtest past races and log live predictions.

Backtest: for every past race, fit the engine only on races run *before* it (so it
cannot peek), plan the race, and compare with what the top ten actually did:

- **stops**: did the engine pick the number of stops most of the top ten used?
- **first stop**: how far was its first pit lap from the top ten's median first
  stop (among drivers who used that number of stops), and was that median inside
  the engine's pit window?

Races where three or more of the top ten stopped under a safety car, VSC or red flag are scored too
but also reported apart: the engine plans a green-flag race, so the "clean" races
are the fairest test.

Two simple baselines give the scores context: "always one stop", and "the same
number of stops as the most recent race at this circuit". Wet races are skipped
(the engine plans dry races).

Live log: each week the next race's plan is saved *before* the race; once the race
has run it is scored the same way. That is the record no backtest can fake.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, cast

import pandas as pd

from f1_foundation.circuits import canonical_circuit
from tire_strategy.config import DATA_DIR
from tire_strategy.strategy.engine import circuit_params, race_plan
from tire_strategy.strategy.params import fit_strategy_params

logger = logging.getLogger(__name__)
PREDICTIONS_PATH: Path = DATA_DIR / "processed" / "predictions.json"
_MIN_PRIOR_RACES = 8


def actual_strategies(race: pd.DataFrame, top: int = 10) -> list[dict[str, Any]] | None:
    """Stops and pit laps of the top finishers; ``None`` for a wet race."""
    total = int(race["lap_number"].max())
    if race["rainfall"].fillna(False).astype(bool).mean() > 0.05:
        return None
    last = cast(pd.DataFrame, race.sort_values("lap_number")).groupby("driver").tail(1)
    finishers = cast(pd.DataFrame, last[last["lap_number"] >= total - 1])
    finishers = finishers.sort_values(["lap_number", "position"], ascending=[False, True])
    flags = [c for c in ("flag_sc", "flag_vsc", "flag_red") if c in race]
    lead = cast(pd.DataFrame, race.groupby("lap_number")[flags].max())
    flagged = cast(pd.Series, lead.fillna(False).astype(bool).any(axis=1))
    # A stop made under a safety car, VSC or red flag (that lap or the next) was
    # shaped by the race, not planned.
    flag_laps = [int(cast(int, n)) for n, on in flagged.items() if on]
    caution = set(flag_laps) | {n - 1 for n in flag_laps}
    out = []
    for code in finishers["driver"].head(top):
        laps = cast(pd.DataFrame, race[race["driver"] == code]).sort_values("lap_number")
        if bool(laps["compound"].isin(["INTERMEDIATE", "WET"]).any()):
            return None  # intermediates or wets were needed: not a dry race
        pits = [int(x) for x in laps.loc[laps["is_in_lap"].astype(bool), "lap_number"]]
        pits = [p for p in pits if 1 < p < total]
        out.append({"driver": str(code), "stops": len(pits), "pit_laps": pits,
                    "caution_pits": sum(p in caution for p in pits)})  # fmt: skip
    return out or None


def score(plan: dict[str, Any], actual: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare one plan with what the top finishers did."""
    stops = Counter(a["stops"] for a in actual)
    modal = stops.most_common(1)[0][0]
    firsts = [a["pit_laps"][0] for a in actual if a["stops"] == modal and a["pit_laps"]]
    actual_first = median(firsts) if firsts else None
    pred_first = plan["pit_laps"][0] if plan["pit_laps"] else None
    window = plan.get("windows", [[None, None]])[0] if plan["pit_laps"] else [None, None]
    return {
        # Shaped by the race when at least three of the top ten stopped under caution.
        "caution_shaped": sum(bool(a.get("caution_pits")) for a in actual) >= 3,
        "predicted": plan["label"],
        "predicted_stops": plan["stops"],
        "predicted_first_stop": pred_first,
        "window": window,
        "actual_stops": modal,
        "actual_share": round(stops[modal] / len(actual), 2),
        "actual_first_stop": actual_first,
        "stops_match": plan["stops"] == modal,
        "same_stops_share": round(stops[plan["stops"]] / len(actual), 2),
        "first_stop_error": None
        if pred_first is None or actual_first is None
        else round(float(pred_first - actual_first), 1),
        "in_window": None
        if actual_first is None or window[0] is None
        else bool(window[0] <= actual_first <= window[1]),
    }


def _races(race_laps: pd.DataFrame) -> list[tuple[int, int, str]]:
    keys = cast(pd.DataFrame, race_laps[["year", "round", "circuit"]].drop_duplicates())
    keys = keys.sort_values(by=["year", "round"])
    return [(int(y), int(r), canonical_circuit(c)) for y, r, c in keys.itertuples(index=False)]


def backtest(race_laps: pd.DataFrame, since: int = 2024) -> dict[str, Any]:
    """Score the engine on every dry race from ``since``, fitted only on earlier races."""
    race_laps = race_laps.copy()
    race_laps["circuit"] = race_laps["circuit"].map(canonical_circuit)
    order = _races(race_laps)
    rows: list[dict[str, Any]] = []
    last_stops: dict[str, int] = {}
    for i, (year, rnd, circuit) in enumerate(order):
        race = cast(
            pd.DataFrame, race_laps[(race_laps["year"] == year) & (race_laps["round"] == rnd)]
        )
        actual = actual_strategies(race)
        if actual is not None and year >= since and i >= _MIN_PRIOR_RACES:
            before = race_laps.merge(
                pd.DataFrame(order[:i], columns=pd.Index(["year", "round", "circuit"]))
            )
            params = fit_strategy_params(before)
            p = circuit_params(params, circuit)
            laps = int(race["lap_number"].max())
            plan = race_plan(p, laps)["best"]
            row = {"year": year, "round": rnd, "circuit": circuit, "laps": laps,
                   "known_circuit": p["known"], **score(plan, actual)}  # fmt: skip
            modal = row["actual_stops"]
            row["baseline_one_stop"] = modal == 1
            row["baseline_last_time"] = (
                last_stops[circuit] == modal if circuit in last_stops else None
            )
            rows.append(row)
            logger.info("Backtest %s %s %s: %s vs %s stops", year, rnd, circuit,
                        row["predicted"], modal)  # fmt: skip
        if actual is not None:
            last_stops[circuit] = Counter(a["stops"] for a in actual).most_common(1)[0][0]
    return {"races": rows, "summary": summarise(rows)}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"races": 0}
    clean = [r for r in rows if not r["caution_shaped"]]
    return {**_rates(rows), "clean": _rates(clean)}


def _rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    errors = [abs(r["first_stop_error"]) for r in rows if r.get("first_stop_error") is not None]
    return {
        "races": len(rows),
        "stops_match": rate("stops_match"),
        "baseline_one_stop": rate("baseline_one_stop"),
        "baseline_last_time": rate("baseline_last_time"),
        "in_window": rate("in_window"),
        "first_stop_error_median": round(median(errors), 1) if errors else None,
    }


# ------------------------------------------------------------------ live predictions
def load_predictions(path: Path = PREDICTIONS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def update_predictions(
    log: list[dict[str, Any]],
    params: dict[str, Any],
    calendar: dict[str, Any] | None,
    race_laps: pd.DataFrame | None,
    now: str,
) -> list[dict[str, Any]]:
    """Add the next race's plan (once, before the race) and score finished ones."""
    log = [dict(e) for e in log]
    nxt = (calendar or {}).get("next")
    if nxt and not any(e["year"] == nxt["year"] and e["round"] == nxt["round"] for e in log):
        p = circuit_params(params, nxt["circuit"])
        laps = p["race_laps"]
        if laps:
            plan = race_plan(p, laps)["best"]
            keep = ("label", "stops", "pit_laps", "windows")
            log.append({"year": nxt["year"], "round": nxt["round"], "circuit": nxt["circuit"],
                        "race_date": nxt["date"], "made_at": now, "laps": laps,
                        "plan": {k: plan[k] for k in keep}})  # fmt: skip
    if race_laps is not None:
        for e in log:
            if "result" in e:
                continue
            race = cast(
                pd.DataFrame,
                race_laps[(race_laps["year"] == e["year"]) & (race_laps["round"] == e["round"])],
            )
            if race.empty:
                continue
            actual = actual_strategies(race)
            e["result"] = {"wet": True} if actual is None else score(e["plan"], actual)
    return sorted(log, key=lambda e: (e["year"], e["round"]))
