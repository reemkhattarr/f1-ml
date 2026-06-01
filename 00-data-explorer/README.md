# Project 00 — F1 Data Explorer

The data-foundation of the portfolio: a GUI to **browse every field** in the
datasets the ML projects are built on. Pick a dataset, filter by season / circuit /
driver, and inspect all columns.

**🔴 Live:** https://huggingface.co/spaces/reemkhattar/f1-data-explorer

## What it shows

| Dataset | Rows | What's in it |
|---|---|---|
| Lap features + labels (main) | ~122k laps | the engineered model inputs + strategy labels, per driver-lap |
| Clean stint laps | ~60k laps | clean slick laps **with lap times** |
| Telemetry features | ~27k laps | per-lap driving-aggression / tyre-load aggregates |

## Why "00"

It sits *before* the four ML projects (`01`–`04`) because it's about
**seeing the raw material** — the F1 timing data — rather than modelling it. It's
the first thing a visitor should open to understand what the data looks like.

## Implementation

The app reuses the shared `tire_strategy` package (it reads the parquets that
project `01` builds):

- App: [`01-tire-strategy/src/tire_strategy/serving/explorer.py`](../01-tire-strategy/src/tire_strategy/serving/explorer.py)
- Deploy scaffold: [`01-tire-strategy/deploy/explorer_space/`](../01-tire-strategy/deploy/explorer_space/) + `build_explorer_space.py`

Run locally: `just explorer`. Rebuild + redeploy the Space: `just build-explorer-space && just deploy-explorer`.

## Not yet visualized (FastF1 has much more)

The explorer currently shows our *processed* parquets. FastF1 also exposes — and we
could add — raw telemetry traces (speed/throttle/brake/gear + X/Y track position),
sector times and speed-trap speeds, full weather, session results, qualifying and
practice sessions, and race-control messages. The raw telemetry traces are the only
heavy addition; the rest are cheap to add for already-cached races.
