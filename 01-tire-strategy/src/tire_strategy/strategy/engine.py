"""Strategy planner and pit-wall advisor.

The race is put on one clock. Every lap on compound ``c`` at tyre age ``a`` costs

    offset[c] + degradation[c] * a          (seconds, relative to a fresh medium)

and every stop costs the circuit's pit loss. Fuel burn and the car's own pace are
the same whatever the strategy, so they cancel and are left out. A strategy's cost
is the sum over its laps plus its stops; the best strategy is the cheapest one that
respects the rules (at least two different dry compounds) and the tyres' real life
(no stint longer than teams have actually run that compound at that circuit).

For each compound sequence (up to three stops) the best stint lengths come from a
small dynamic programme, so the answer is exact for this cost model, not a guess.

The same algorithm is implemented in ``site/engine.js`` so the website gives the
same answers as Python; ``tests/test_engine_parity.py`` checks that they agree.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Any

DRY: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
WET: tuple[str, ...] = ("INTERMEDIATE", "WET")
# Running wet tyres on a dry track: an assumption, used only to price staying out.
_WET_ON_DRY = {"offset": 3.0, "degradation": 0.12, "life": 30}
_INF = math.inf
MARGIN_S = 0.5  # below this, box and stay out are called a coin flip
WINDOW_S = 2.0  # a stop lap is "in the window" if it costs at most this much more


def circuit_params(params: dict[str, Any], circuit: str | None) -> dict[str, Any]:
    """Parameters for one circuit, falling back to the all-circuit defaults."""
    base = params["default"]
    c = params["circuits"].get(circuit or "", {})
    return {
        "circuit": circuit,
        "known": bool(c),
        "race_laps": c.get("race_laps"),
        "degradation": {**base["degradation"], **c.get("degradation", {})},
        "offsets": {**base["offsets"], **c.get("offsets", {})},
        "tyre_life": {**base["tyre_life"], **c.get("tyre_life", {})},
        "pit_loss": c.get("pit_loss", base["pit_loss"]),
        "p_caution": c.get("p_caution", base["p_caution"]),
        "p_safety_car": c.get("p_safety_car", base["p_safety_car"]),
        "caution_per_lap": c.get("caution_per_lap", base["caution_per_lap"]),
        "sc_pit_factor": params.get("sc_pit_factor", 0.5),
        "vsc_pit_factor": params.get("vsc_pit_factor", 0.65),
    }


def _compound(p: dict[str, Any], c: str) -> tuple[float, float, int]:
    if c in WET:
        return _WET_ON_DRY["offset"], _WET_ON_DRY["degradation"], _WET_ON_DRY["life"]
    return float(p["offsets"][c]), float(p["degradation"][c]), int(p["tyre_life"][c])


def stint_cost(p: dict[str, Any], compound: str, laps: int, age0: int = 0) -> float:
    """Seconds lost over a stint of ``laps`` laps starting at tyre age ``age0``."""
    off, deg, _ = _compound(p, compound)
    return laps * off + deg * (laps * age0 + laps * (laps - 1) / 2)


def lap_cost(p: dict[str, Any], compound: str, age: int) -> float:
    off, deg, _ = _compound(p, compound)
    return off + deg * age


def _best_lengths(
    p: dict[str, Any], seq: Sequence[str], total: int, first_age: int
) -> tuple[float, list[int]] | None:
    """Cheapest stint lengths for a compound sequence covering ``total`` laps."""
    k = len(seq)
    caps = []
    for i, c in enumerate(seq):
        age0 = first_age if i == 0 else 0
        caps.append(min(total, _compound(p, c)[2] - age0))
    if any(cap < 1 for cap in caps) or sum(caps) < total or total < k:
        return None
    # f[i][n]: cheapest cost of the first i stints covering n laps.
    f = [[_INF] * (total + 1) for _ in range(k + 1)]
    choice = [[0] * (total + 1) for _ in range(k + 1)]
    f[0][0] = 0.0
    for i, c in enumerate(seq):
        age0 = first_age if i == 0 else 0
        row, nxt = f[i], f[i + 1]
        for n in range(total + 1):
            base = row[n]
            if base == _INF:
                continue
            for length in range(1, min(caps[i], total - n) + 1):
                cost = base + stint_cost(p, c, length, age0)
                if cost < nxt[n + length] - 1e-12:
                    nxt[n + length] = cost
                    choice[i + 1][n + length] = length
    if f[k][total] == _INF:
        return None
    lengths, n = [], total
    for i in range(k, 0, -1):
        length = choice[i][n]
        lengths.append(length)
        n -= length
    return f[k][total], lengths[::-1]


def _sequences(
    compounds: Sequence[str], max_stops: int, first: str | None
) -> Iterable[tuple[str, ...]]:
    for stops in range(max_stops + 1):
        if first is None:
            yield from itertools.product(compounds, repeat=stops + 1)
        else:
            for rest in itertools.product(compounds, repeat=stops):
                yield (first, *rest)


def _rule_ok(seq: Sequence[str], used: Iterable[str], rule: bool) -> bool:
    if not rule:
        return True
    dry = {c for c in (*used, *seq) if c in DRY}
    return len(dry) >= 2 or any(c in WET for c in (*used, *seq))


def plan(
    p: dict[str, Any],
    total_laps: int,
    *,
    start_lap: int = 1,
    current: tuple[str, int] | None = None,
    used: Iterable[str] = (),
    compounds: Sequence[str] = DRY,
    two_compound_rule: bool = True,
    max_stops: int = 3,
    pit_loss: float | None = None,
) -> list[dict[str, Any]]:
    """All feasible strategies from ``start_lap`` to the flag, cheapest first.

    ``current`` = (compound, tyre age) continues the tyre already on the car (the
    advisor's "stay out" case); without it the first stint starts on a fresh set.
    """
    remaining = total_laps - start_lap + 1
    if remaining < 1:
        return []
    loss = float(p["pit_loss"] if pit_loss is None else pit_loss)
    used = tuple(used)
    first, first_age = current if current else (None, 0)
    options = []
    for seq in _sequences(compounds, max_stops, first):
        if not _rule_ok(seq, used, two_compound_rule):
            continue
        best = _best_lengths(p, seq, remaining, first_age)
        if best is None:
            continue
        cost, lengths = best
        stops = len(seq) - 1
        stints, lap = [], start_lap
        for c, n in zip(seq, lengths, strict=True):
            stints.append({"compound": c, "laps": n, "start_lap": lap, "end_lap": lap + n - 1})
            lap += n
        options.append(
            {
                "stops": stops,
                "stints": stints,
                "pit_laps": [s["end_lap"] for s in stints[:-1]],
                "tyre_cost": round(cost, 2),
                "pit_cost": round(stops * loss, 2),
                "cost": round(cost + stops * loss, 2),
            }
        )
    options.sort(key=lambda o: (o["cost"], o["stops"]))
    if options:
        best_cost = options[0]["cost"]
        for o in options:
            o["delta"] = round(o["cost"] - best_cost, 2)
    return options


def pit_windows(p: dict[str, Any], option: dict[str, Any], window_s: float = WINDOW_S) -> list:
    """For each stop, the range of laps that costs at most ``window_s`` more."""
    stints = option["stints"]
    windows = []
    for i in range(len(stints) - 1):
        a, b = stints[i], stints[i + 1]
        age_a = a.get("age0", 0)
        base = stint_cost(p, a["compound"], a["laps"], age_a) + stint_cost(
            p, b["compound"], b["laps"]
        )
        lo = hi = a["end_lap"]
        for direction in (-1, 1):
            shift = 1
            while True:
                la, lb = a["laps"] + direction * shift, b["laps"] - direction * shift
                if la < 1 or lb < 1:
                    break
                if (
                    la + age_a > _compound(p, a["compound"])[2]
                    or lb > _compound(p, b["compound"])[2]
                ):
                    break
                cost = stint_cost(p, a["compound"], la, age_a) + stint_cost(p, b["compound"], lb)
                if cost - base > window_s:
                    break
                if direction < 0:
                    lo = a["end_lap"] - shift
                else:
                    hi = a["end_lap"] + shift
                shift += 1
        windows.append([lo, hi])
    return windows


def best_by_stops(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for o in options:
        seen.setdefault(o["stops"], o)
    return [seen[k] for k in sorted(seen)]


def _scaled(p: dict[str, Any], factor: float) -> dict[str, Any]:
    q = dict(p)
    q["degradation"] = {c: v * factor for c, v in p["degradation"].items()}
    return q


def _label(option: dict[str, Any]) -> str:
    return "-".join(s["compound"][0] for s in option["stints"])


def race_plan(
    p: dict[str, Any],
    total_laps: int,
    *,
    compounds: Sequence[str] = DRY,
    two_compound_rule: bool = True,
    max_stops: int = 3,
) -> dict[str, Any]:
    """Pre-race plan: best strategy per stop count, pit windows and sensitivity."""
    options = plan(
        p, total_laps, compounds=compounds, two_compound_rule=two_compound_rule,
        max_stops=max_stops,
    )  # fmt: skip
    if not options:
        return {"options": [], "best": None}
    top = best_by_stops(options)
    for o in top:
        o["windows"] = pit_windows(p, o)
        o["label"] = _label(o)
    sensitivity = {}
    for name, factor in (("degradation_low", 0.75), ("degradation_high", 1.25)):
        alt = plan(
            _scaled(p, factor), total_laps, compounds=compounds,
            two_compound_rule=two_compound_rule, max_stops=max_stops,
        )  # fmt: skip
        sensitivity[name] = {"label": _label(alt[0]), "stops": alt[0]["stops"]} if alt else None
    alternatives = list(options[:8])
    for o in alternatives:
        o.setdefault("label", _label(o))
    return {"best": options[0], "by_stops": top, "top": alternatives,
            "sensitivity": sensitivity}  # fmt: skip


def advise(
    p: dict[str, Any],
    *,
    total_laps: int,
    lap: int,
    compound: str,
    tyre_age: int,
    used: Iterable[str] = (),
    caution: str = "none",
    rain: bool = False,
    gap_ahead: float | None = None,
    gap_behind: float | None = None,
    compounds: Sequence[str] = DRY,
    two_compound_rule: bool = True,
) -> dict[str, Any]:
    """Box now or stay out? Compares the best race from here for each choice.

    ``lap`` is the lap about to start; ``tyre_age`` the laps already done on the
    current tyre. ``caution`` is "none", "sc" or "vsc" (a stop is cheaper then).
    """
    remaining = total_laps - lap + 1
    if remaining < 1:
        raise ValueError("The race is already over at that lap.")
    used = {*used, compound}
    factor = {"sc": p["sc_pit_factor"], "vsc": p["vsc_pit_factor"]}.get(caution, 1.0)
    pit_now = round(float(p["pit_loss"]) * factor, 2)
    notes: list[str] = []

    if rain:
        on_slicks = compound in DRY
        return {
            "decision": "box" if on_slicks else "stay",
            "compound": "INTERMEDIATE" if on_slicks else None,
            "source": "wet rule",
            "delta": None,
            "pit_loss_now": pit_now,
            "stay": None,
            "box": [],
            "notes": [
                "Rain: on slicks, fit intermediates (the rule teams follow almost always)."
                if on_slicks
                else "Rain: stay on wet-weather tyres."
            ],
        }

    stay_opts = plan(
        p, total_laps, start_lap=lap, current=(compound, tyre_age), used=used,
        compounds=compounds, two_compound_rule=two_compound_rule,
    )  # fmt: skip
    stay = stay_opts[0] if stay_opts else None
    box = []
    for c in compounds:
        opts = plan(
            p, total_laps, start_lap=lap, current=(c, 0), used=used | {c},
            compounds=compounds, two_compound_rule=two_compound_rule,
        )  # fmt: skip
        if opts:
            o = dict(opts[0])
            o["cost"] = round(o["cost"] + pit_now, 2)
            o["pit_cost"] = round(o["pit_cost"] + pit_now, 2)
            o["compound"] = c
            box.append(o)
    box.sort(key=lambda o: o["cost"])
    best_box = box[0] if box else None

    if stay is None and best_box is None:
        raise ValueError("No legal strategy from this state.")
    if stay is None:
        decision, delta = "box", None
        notes.append("This tyre cannot reach the flag or the rules require another compound.")
    elif best_box is None:
        decision, delta = "stay", None
    else:
        delta = round(stay["cost"] - best_box["cost"], 2)  # > 0: boxing is faster
        decision = "box" if delta > MARGIN_S else "stay" if delta < -MARGIN_S else "either"

    if caution in ("sc", "vsc") and best_box is not None:
        saved = round(float(p["pit_loss"]) - pit_now, 1)
        notes.append(f"{caution.upper()} now: a stop costs {pit_now:.1f}s instead of "
                     f"{p['pit_loss']:.1f}s ({saved:.1f}s cheaper).")  # fmt: skip
    if best_box is not None and gap_behind is not None and gap_behind < pit_now:
        notes.append(f"Boxing loses about {pit_now:.0f}s: you would rejoin behind the car "
                     f"now {gap_behind:.1f}s behind you.")  # fmt: skip
    if best_box is not None and gap_ahead is not None and caution == "none":
        gain = lap_cost(p, compound, tyre_age) - lap_cost(p, best_box["compound"], 0)
        if gain > 0 and 2 * gain >= gap_ahead:
            notes.append(f"Undercut on: fresh {best_box['compound'].lower()}s are about "
                         f"{gain:.1f}s a lap faster and the car ahead is {gap_ahead:.1f}s away.")  # fmt: skip  # noqa: E501
    if stay is not None and stay["pit_laps"]:
        notes.append(f"If you stay out, the best next stop is lap {stay['pit_laps'][0]}.")

    return {
        "decision": decision,
        "compound": best_box["compound"] if best_box and decision != "stay" else None,
        "source": "strategy engine",
        "delta": delta,
        "pit_loss_now": pit_now,
        "stay": stay,
        "box": box,
        "notes": notes,
    }
