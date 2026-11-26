"""Tyre pace and wear from race laps: one regression inside each driver's race.

For every clean green-flag lap of a driver in a race:

    lap_time = car_and_driver + trend * lap + offset[compound] + wear[compound] * tyre_age

- ``car_and_driver`` is a separate constant for every driver in every race (a fixed
  effect), so differences between cars never leak into the tyre numbers.
- ``trend`` is the race-long drift in lap time: fuel burning off and the track
  rubbering in. It is estimated, not assumed. Comparing a driver's stints without
  it makes whichever stint comes later look faster, which is what broke the first
  version (medium after hard looked 0.2 s quicker, hard after medium only 0.09 s
  slower).
- ``offset`` is each compound's pace on a fresh set relative to the medium, and
  ``wear`` how much it slows per lap of age. Offsets are identified by drivers who
  ran more than one compound; wear and trend are separated because stints start at
  different points of the race.

The fixed effects are removed by subtracting each driver-race's mean (within
transformation). Laps far off the fit (traffic, mistakes) are dropped and the fit
repeated. A circuit's estimate is pulled toward a prior (the all-circuit fit) in
proportion to how little data it has, so a circuit with one race cannot produce a
wild number. Physical limits are applied last: wear is positive and a softer
compound is faster than a harder one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")
# Columns of the design matrix: trend, offset S, offset H, wear S, wear M, wear H.
_NAMES = ("trend", "off_SOFT", "off_HARD", "wear_SOFT", "wear_MEDIUM", "wear_HARD")
MIN_WEAR = 0.005  # s/lap: a tyre never gets faster with age
MIN_STEP = 0.05  # s/lap: a softer fresh tyre is at least this much faster
# Weak physical prior for the all-circuit fit (only matters with almost no data).
GENERIC_PRIOR = np.array([-0.06, -0.4, 0.3, 0.09, 0.06, 0.045])


@dataclass
class PaceFit:
    offsets: dict[str, float]
    wear: dict[str, float]
    trend: float
    laps: int
    drivers: int  # driver-races that contributed
    multi_compound: int  # driver-races that ran 2+ compounds (identify offsets)
    clamped: list[str] = field(default_factory=list)
    coef: np.ndarray = field(default_factory=lambda: GENERIC_PRIOR.copy(), repr=False)


def _design(laps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comp = laps["compound"].astype(str).to_numpy()
    age = laps["tyre_age"].to_numpy(float)
    lap = laps["lap_number"].to_numpy(float)
    X = np.column_stack(
        [
            lap,
            (comp == "SOFT").astype(float),
            (comp == "HARD").astype(float),
            age * (comp == "SOFT"),
            age * (comp == "MEDIUM"),
            age * (comp == "HARD"),
        ]
    )
    groups = (
        laps["year"].astype(str)
        + "_"
        + laps["round"].astype(str)
        + "_"
        + laps["driver"].astype(str)
    )
    codes = pd.factorize(groups)[0]
    return X, laps["lap_time_s"].to_numpy(float), codes


def _demean(a: np.ndarray, codes: np.ndarray) -> np.ndarray:
    n = codes.max() + 1
    counts = np.bincount(codes, minlength=n).astype(float)
    if a.ndim == 1:
        return a - (np.bincount(codes, weights=a, minlength=n) / counts)[codes]
    means = np.column_stack(
        [np.bincount(codes, weights=a[:, j], minlength=n) / counts for j in range(a.shape[1])]
    )
    return a - means[codes]


def fit_pace(
    laps: pd.DataFrame,
    prior: np.ndarray | None = None,
    prior_laps: float | np.ndarray = 400.0,
) -> PaceFit:
    """Fit offsets, wear and trend on clean laps (see module docstring).

    ``prior`` (coefficients in ``_NAMES`` order) acts like ``prior_laps`` extra laps
    of evidence (one number, or one per coefficient): with thousands of laps the data
    decides, with a few hundred the estimate sits between data and prior.
    """
    prior = GENERIC_PRIOR if prior is None else prior
    strength = np.broadcast_to(np.asarray(prior_laps, float), prior.shape)
    laps = cast(pd.DataFrame, laps[laps["compound"].isin(COMPOUNDS)])
    if len(laps) < 30:
        return _finish(prior.copy(), laps)
    X, y, codes = _design(laps)
    Xd, yd = _demean(X, codes), _demean(y, codes)
    keep = np.ones(len(yd), bool)
    coef = prior.copy()
    for _ in range(2):  # fit, drop outliers, refit
        A, b = Xd[keep], yd[keep]
        scale = np.maximum((A**2).mean(axis=0), 1e-9)
        lam = np.diag(strength * scale)
        coef = np.linalg.solve(A.T @ A + lam, A.T @ b + lam @ prior)
        resid = yd - Xd @ coef
        mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) * 1.4826
        keep = np.abs(resid) <= max(3 * mad, 0.3)
    return _finish(coef, cast(pd.DataFrame, laps.iloc[np.flatnonzero(keep)]))


def _monotone(values: list[float]) -> list[float]:
    """Pool adjacent violators: make values non-increasing (soft >= medium >= hard)."""
    blocks = [[v, 1] for v in values]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            size = blocks[i][1] + blocks[i + 1][1]
            blocks[i : i + 2] = [[total / size, size]]
            i = max(i - 1, 0)
        else:
            i += 1
    return [v for v, n in blocks for _ in range(int(n))]


def _finish(coef: np.ndarray, laps: pd.DataFrame) -> PaceFit:
    raw: dict[str, float] = dict(zip(_NAMES, coef.tolist(), strict=True))
    clamped: list[str] = []
    measured = [raw[f"wear_{c}"] for c in COMPOUNDS]
    ordered = _monotone(measured)  # a softer tyre wears at least as fast
    if any(abs(a - b) > 1e-9 for a, b in zip(measured, ordered, strict=True)):
        clamped.append("wear order")
    wear = {}
    for c, w in zip(COMPOUNDS, ordered, strict=True):
        if w < MIN_WEAR:
            clamped.append(f"{c.lower()} wear")
            w = MIN_WEAR
        wear[c] = round(w, 4)
    soft, hard = raw["off_SOFT"], raw["off_HARD"]
    if soft > -MIN_STEP:
        clamped.append("soft pace")
        soft = -MIN_STEP
    if hard < MIN_STEP:
        clamped.append("hard pace")
        hard = MIN_STEP
    if len(laps):
        per = laps.groupby(["year", "round", "driver"])["compound"].nunique()
        drivers, multi = len(per), int((per >= 2).sum())
    else:
        drivers = multi = 0
    return PaceFit(
        offsets={"SOFT": round(soft, 3), "MEDIUM": 0.0, "HARD": round(hard, 3)},
        wear=wear,
        trend=round(raw["trend"], 4),
        laps=len(laps),
        drivers=drivers,
        multi_compound=multi,
        clamped=clamped,
        coef=coef,
    )
