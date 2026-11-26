"""Race data from the OpenF1 API, in the same shape as :func:`f1_foundation.load_race`.

Why a second source: F1's live-timing server (which FastF1 reads) refuses requests
from cloud machines such as GitHub's runners, so the scheduled cloud jobs stopped
receiving new races. OpenF1 (https://openf1.org) republishes the same timing feed
through a public API that cloud machines can reach, so the cloud can keep the
dataset current without anyone running code locally.

The output columns match ``load_race`` so labels, features and the strategy engine
work unchanged. Differences worth knowing:

- ``is_accurate`` is True for every timed lap and ``deleted`` is always False
  (OpenF1 has no per-lap accuracy or track-limits flag); the downstream cleaning
  already drops in/out laps, caution laps and outliers.
- Track status comes from race-control messages rather than FastF1's per-lap
  status string, so safety-car laps run from the "deployed" lap to the "in this
  lap" / "ending" lap.
- ``position`` is the order cars crossed the line on each lap.

The free API allows a few requests per second, so the client spaces requests and
retries politely, and caches finished sessions on disk.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from f1_foundation.circuits import canonical_circuit
from f1_foundation.config import DATA_DIR

logger = logging.getLogger(__name__)

API_URL = "https://api.openf1.org/v1"
CACHE_DIR: Path = DATA_DIR / "cache" / "openf1"
_SC_SPAN_FALLBACK = 3  # laps a safety car is assumed to last if its end is not reported

Record = dict[str, Any]


# --------------------------------------------------------------------------- client
@dataclass
class OpenF1Client:
    """Small polite client: spaced requests, retries on 429/5xx, on-disk cache."""

    base_url: str = API_URL
    min_interval_s: float = 2.1
    max_retries: int = 6
    cache_dir: Path | None = CACHE_DIR
    _last_call: float = field(default=0.0, repr=False)

    def get(self, endpoint: str, cache: bool = True, **params: object) -> list[Record]:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}/{endpoint}?{query}"
        path = self._cache_path(url) if cache else None
        if path is not None and path.exists():
            return json.loads(path.read_text())

        delay = 5.0
        for attempt in range(self.max_retries):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:  # OpenF1 answers 404 when there are no rows
                    data = []
                    break
                if exc.code not in (429, 500, 502, 503, 504) or attempt == self.max_retries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries - 1:
                    raise
            logger.info("OpenF1 busy (%s), retrying in %.0fs", endpoint, delay)
            time.sleep(delay)
            delay = min(delay * 2, 60)
        else:  # pragma: no cover - loop always breaks or raises
            data = []

        if not isinstance(data, list):
            data = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data))
        return cast(list[Record], data)

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{sha1(url.encode()).hexdigest()}.json"


# ------------------------------------------------------------------------- calendar
@dataclass(frozen=True)
class RaceSession:
    year: int
    round: int
    session_key: int
    circuit: str
    date_start: datetime
    date_end: datetime | None


def _parse_time(value: object) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def calendar_from_sessions(sessions: list[Record], year: int) -> list[RaceSession]:
    """Grand-prix races of a season, numbered like the official calendar.

    Cancelled races are skipped before numbering, so rounds match FastF1 (e.g. in
    2026 Miami is round 4 after Bahrain and Saudi Arabia were cancelled).
    """
    races = [
        s
        for s in sessions
        if s.get("session_name") == "Race" and not s.get("is_cancelled") and s.get("date_start")
    ]
    races.sort(key=lambda s: str(s["date_start"]))
    out: list[RaceSession] = []
    for i, s in enumerate(races, start=1):
        start = _parse_time(s["date_start"])
        assert start is not None
        out.append(
            RaceSession(
                year=year,
                round=i,
                session_key=int(s["session_key"]),
                circuit=canonical_circuit(s.get("location") or s.get("circuit_short_name")),
                date_start=start,
                date_end=_parse_time(s.get("date_end")),
            )
        )
    return out


def race_calendar(year: int, client: OpenF1Client | None = None) -> list[RaceSession]:
    """All (non-cancelled) races of a season, including ones not run yet."""
    client = client or OpenF1Client()
    # The calendar can change during a season, so it is never served from cache.
    return calendar_from_sessions(
        client.get("sessions", cache=False, year=year, session_name="Race"), year
    )


def completed_races(
    seasons: tuple[int, ...], client: OpenF1Client | None = None, now: datetime | None = None
) -> list[RaceSession]:
    """Races that have finished (with a two-hour margin for the data to land)."""
    now = now or datetime.now(UTC)
    done: list[RaceSession] = []
    for year in seasons:
        for race in race_calendar(year, client):
            end = race.date_end or race.date_start
            if (now - end).total_seconds() > 2 * 3600:
                done.append(race)
    return done


# ----------------------------------------------------------------------- transforms
def _safety_car_laps(race_control: list[Record]) -> tuple[set[int], set[int], set[int]]:
    """Laps run under the safety car, the virtual safety car, and red flags."""
    sc: set[int] = set()
    vsc: set[int] = set()
    red: set[int] = set()
    open_sc: int | None = None
    open_vsc: int | None = None
    events = sorted(race_control, key=lambda m: str(m.get("date", "")))
    for msg in events:
        lap = msg.get("lap_number")
        if lap is None:
            continue
        lap = int(lap)
        text = str(msg.get("message", "")).upper()
        category = str(msg.get("category", ""))
        if category == "Flag" and str(msg.get("flag", "")).upper() == "RED":
            red.add(lap)
        if category != "SafetyCar":
            continue
        if "VIRTUAL" in text:
            if "DEPLOYED" in text:
                open_vsc = lap
            elif "ENDING" in text and open_vsc is not None:
                vsc.update(range(open_vsc, lap + 1))
                open_vsc = None
        elif "DEPLOYED" in text:
            open_sc = lap
        elif ("IN THIS LAP" in text or "ENDING" in text) and open_sc is not None:
            sc.update(range(open_sc, lap + 1))
            open_sc = None
    if open_sc is not None:
        sc.update(range(open_sc, open_sc + _SC_SPAN_FALLBACK))
    if open_vsc is not None:
        vsc.update(range(open_vsc, open_vsc + 2))
    return sc, vsc, red


def _yellow_laps(race_control: list[Record]) -> set[int]:
    return {
        int(m["lap_number"])
        for m in race_control
        if m.get("lap_number") is not None
        and str(m.get("flag", "")).upper() in ("YELLOW", "DOUBLE YELLOW")
    }


def race_table(
    laps: list[Record],
    stints: list[Record],
    pits: list[Record],
    race_control: list[Record],
    weather: list[Record],
    drivers: list[Record],
    *,
    year: int,
    round_number: int,
    circuit: str,
) -> pd.DataFrame:
    """Build the per-driver-per-lap race table from raw OpenF1 payloads."""
    if not laps:
        raise ValueError("No laps in this session")
    df = pd.DataFrame(laps)
    df = cast(pd.DataFrame, df[df["lap_number"].notna()]).copy()
    df["driver_number"] = df["driver_number"].astype(int)
    df["lap_number"] = df["lap_number"].astype(int)
    df["date_start"] = pd.to_datetime(df["date_start"], utc=True, format="ISO8601")
    df = cast(pd.DataFrame, df.sort_values(["driver_number", "lap_number"])).reset_index(drop=True)
    # Some laps come without a start time (often lap 1 or a lap with a timing gap):
    # take it from the previous lap's start plus its duration when that is known.
    missing = cast(pd.Series, df["date_start"].isna())
    if bool(missing.any()):
        seconds = cast(pd.Series, pd.to_numeric(df["lap_duration"], errors="coerce"))
        duration = cast(pd.Series, pd.to_timedelta(seconds, unit="s"))
        by_driver = df.groupby("driver_number")
        prev_end = by_driver["date_start"].shift(1) + duration.groupby(df["driver_number"]).shift(1)
        df.loc[missing, "date_start"] = prev_end[missing]

    lap_time = cast(pd.Series, pd.to_numeric(df["lap_duration"], errors="coerce"))
    next_start = df.groupby("driver_number")["date_start"].shift(-1)
    crossing = next_start.where(
        next_start.notna(), df["date_start"] + pd.to_timedelta(lap_time.to_numpy(), unit="s")
    )
    t0 = df["date_start"].min()
    df["time_s"] = (crossing - t0).dt.total_seconds()
    df["lap_start_time_s"] = (df["date_start"] - t0).dt.total_seconds()
    # Fill missing lap times (often lap 1) from consecutive line crossings.
    gap = cast(pd.Series, df.groupby("driver_number")["time_s"].diff())
    df["lap_time_s"] = lap_time.where(lap_time.notna(), gap)
    first = df["lap_number"] == 1
    df.loc[first & lap_time.isna(), "lap_time_s"] = np.nan  # standing start: no usable time

    # Order across the line on each lap.
    df["position"] = df.groupby("lap_number")["time_s"].rank(method="first").astype("Int64")

    # Tyres: map every lap onto its stint.
    df["stint"] = pd.NA
    df["compound"] = pd.NA
    df["tyre_life"] = pd.NA
    df["fresh_tyre"] = pd.NA
    max_lap = int(df["lap_number"].max())
    for s in stints:
        if s.get("driver_number") is None or s.get("lap_start") is None:
            continue
        start = int(s["lap_start"])
        end = int(s["lap_end"]) if s.get("lap_end") is not None else max_lap
        mask = (df["driver_number"] == int(s["driver_number"])) & df["lap_number"].between(
            start, end
        )
        age0 = int(s.get("tyre_age_at_start") or 0)
        df.loc[mask, "stint"] = int(s.get("stint_number") or 0)
        df.loc[mask, "compound"] = str(s.get("compound") or "UNKNOWN").upper()
        df.loc[mask, "tyre_life"] = age0 + (df.loc[mask, "lap_number"] - start) + 1
        df.loc[mask, "fresh_tyre"] = age0 == 0

    in_laps = {
        (int(p["driver_number"]), int(p["lap_number"]))
        for p in pits
        if p.get("driver_number") is not None and p.get("lap_number") is not None
    }
    keys = list(zip(df["driver_number"], df["lap_number"], strict=True))
    df["is_in_lap"] = [k in in_laps for k in keys]
    df["is_out_lap"] = df["is_pit_out_lap"].fillna(False).astype(bool)

    sc, vsc, red = _safety_car_laps(race_control)
    yellow = _yellow_laps(race_control)
    df["flag_sc"] = df["lap_number"].isin(list(sc))
    df["flag_vsc"] = df["lap_number"].isin(list(vsc))
    df["flag_red"] = df["lap_number"].isin(list(red))
    df["flag_yellow"] = df["lap_number"].isin(list(yellow))
    df["under_caution"] = df["flag_sc"] | df["flag_vsc"]

    # Weather: nearest earlier sample for each lap start.
    if weather:
        w = pd.DataFrame(weather)
        w["date"] = pd.to_datetime(w["date"], utc=True, format="ISO8601")
        w = w.sort_values("date")
        timed = cast(pd.DataFrame, df.loc[df["date_start"].notna(), ["date_start"]])
        merged = pd.merge_asof(
            timed.reset_index().sort_values("date_start"),
            w,
            left_on="date_start",
            right_on="date",
            direction="nearest",
        ).set_index("index")
        merged = merged.reindex(df.index)  # laps with no time at all get no weather
        df["air_temp"] = pd.to_numeric(merged["air_temperature"], errors="coerce")
        df["track_temp"] = pd.to_numeric(merged["track_temperature"], errors="coerce")
        rain = cast(pd.Series, pd.to_numeric(merged["rainfall"], errors="coerce"))
        df["rainfall"] = rain.fillna(0) > 0
        df["humidity"] = pd.to_numeric(merged.get("humidity"), errors="coerce")
        df["pressure"] = pd.to_numeric(merged.get("pressure"), errors="coerce")
        df["wind_speed"] = pd.to_numeric(merged.get("wind_speed"), errors="coerce")
        df["wind_direction"] = pd.to_numeric(merged.get("wind_direction"), errors="coerce")
    else:
        for col in ("air_temp", "track_temp", "humidity", "pressure", "wind_speed"):
            df[col] = np.nan
        df["wind_direction"] = np.nan
        df["rainfall"] = False

    info = {int(d["driver_number"]): d for d in drivers if d.get("driver_number") is not None}
    df["driver"] = [str(info.get(n, {}).get("name_acronym") or n) for n in df["driver_number"]]
    df["team"] = [str(info.get(n, {}).get("team_name") or "UNKNOWN") for n in df["driver_number"]]

    total_laps = max_lap
    out = pd.DataFrame(
        {
            "year": int(year),
            "round": int(round_number),
            "circuit": canonical_circuit(circuit),
            "event_name": f"{canonical_circuit(circuit)} Grand Prix",
            "session": "R",
            "driver": df["driver"].astype("string"),
            "driver_number": df["driver_number"].astype(str).astype("string"),
            "team": df["team"].astype("string"),
            "lap_number": df["lap_number"].astype("Int64"),
            "stint": cast(pd.Series, pd.to_numeric(df["stint"], errors="coerce")).astype("Int64"),
            "compound": df["compound"].astype("string"),
            "tyre_life": cast(pd.Series, pd.to_numeric(df["tyre_life"], errors="coerce")).astype(
                "Int64"
            ),
            "fresh_tyre": df["fresh_tyre"].astype("boolean"),
            "position": df["position"],
            "total_laps": total_laps,
            "laps_remaining": (total_laps - df["lap_number"]).astype("Int64"),
            "lap_time_s": df["lap_time_s"].astype(float),
            "lap_start_time_s": df["lap_start_time_s"].astype(float),
            "time_s": df["time_s"].astype(float),
            "air_temp": df["air_temp"],
            "track_temp": df["track_temp"],
            "rainfall": df["rainfall"].astype("boolean"),
            "humidity": df["humidity"],
            "pressure": df["pressure"],
            "wind_speed": df["wind_speed"],
            "wind_direction": df["wind_direction"],
            "is_in_lap": df["is_in_lap"].astype(bool),
            "is_out_lap": df["is_out_lap"].astype(bool),
            "flag_yellow": df["flag_yellow"].astype("boolean"),
            "flag_sc": df["flag_sc"].astype("boolean"),
            "flag_vsc": df["flag_vsc"].astype("boolean"),
            "flag_red": df["flag_red"].astype("boolean"),
            "under_caution": df["under_caution"].astype("boolean"),
            "is_accurate": pd.Series(df["lap_time_s"].notna(), dtype="boolean"),
            "deleted": pd.Series(False, index=df.index, dtype="boolean"),
        }
    )
    return out.reset_index(drop=True)


def results_table(results: list[Record], drivers: list[Record]) -> pd.DataFrame:
    """Classification in the shape of :func:`f1_foundation.ingest.load_results`."""
    info = {int(d["driver_number"]): d for d in drivers if d.get("driver_number") is not None}
    rows = []
    for r in results:
        n = int(r["driver_number"])
        out_of_race = bool(r.get("dnf") or r.get("dns") or r.get("dsq"))
        pos = r.get("position")
        rows.append(
            {
                "driver": str(info.get(n, {}).get("name_acronym") or n),
                "status": "DNF" if out_of_race else "Finished",
                "classified_position": "R" if out_of_race else str(pos),
                "classified": not out_of_race and pos is not None,
                "finish_position": pos,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=pd.Index(
            ["driver", "status", "classified_position", "classified", "finish_position"]
        ),
    )
    finish = cast(pd.Series, pd.to_numeric(out["finish_position"], errors="coerce"))
    out["finish_position"] = finish.astype("Int64")
    return out


# ---------------------------------------------------------------------- high level
def _session(race: RaceSession | int, year: int | None, client: OpenF1Client) -> RaceSession:
    if isinstance(race, RaceSession):
        return race
    if year is None:
        raise ValueError("year is required when race is a round number")
    for candidate in race_calendar(year, client):
        if candidate.round == int(race):
            return candidate
    raise ValueError(f"No race {year} round {race} in OpenF1")


def load_race_openf1(
    year: int, race: RaceSession | int, client: OpenF1Client | None = None
) -> pd.DataFrame:
    """One race as a per-driver-per-lap table (same columns as ``load_race``)."""
    client = client or OpenF1Client()
    s = _session(race, year, client)
    key = s.session_key
    return race_table(
        laps=client.get("laps", session_key=key),
        stints=client.get("stints", session_key=key),
        pits=client.get("pit", session_key=key),
        race_control=client.get("race_control", session_key=key),
        weather=client.get("weather", session_key=key),
        drivers=client.get("drivers", session_key=key),
        year=s.year,
        round_number=s.round,
        circuit=s.circuit,
    )


def load_results_openf1(
    year: int, race: RaceSession | int, client: OpenF1Client | None = None
) -> pd.DataFrame:
    client = client or OpenF1Client()
    s = _session(race, year, client)
    return results_table(
        client.get("session_result", session_key=s.session_key),
        client.get("drivers", session_key=s.session_key),
    )
