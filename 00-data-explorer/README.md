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

## Structure — the shared foundation

`00` is a real workspace member, the package **`f1_foundation`**, that every
project depends on (and that depends on none of them — the dependency arrow always
points `01–04 → 00`):

```
00-data-explorer/src/f1_foundation/
├── config.py         # SEED + the shared FastF1 cache and data/ paths
├── ingest.py         # raw FastF1 → clean per-lap tables (load_race, load_results)
├── explorer.py       # the Gradio explorer app
└── explorer_data.py  # the all-sessions raw-data builder
```

The FastF1 cache and **all** processed parquets live in one shared, gitignored
`data/` at the workspace root, so every project writes there and the explorer
browses everything by filename — without importing any project's code. A project
parquet is shown as a tab only when its file is present, so future projects 02–04
appear automatically as they produce data.

> Extracted from `01` once a clean seam existed (see `IMPLEMENTATION.md` §13). `01`
> now imports ingestion from the foundation (`from f1_foundation import load_race`).

## Implementation

- App: [`src/f1_foundation/explorer.py`](src/f1_foundation/explorer.py)
- Raw-data builder: [`src/f1_foundation/explorer_data.py`](src/f1_foundation/explorer_data.py)
- Deploy scaffold: [`deploy/explorer_space/`](deploy/explorer_space/) + `build_explorer_space.py`

Run locally: `just explorer`. Rebuild + redeploy the Space: `just build-explorer-space && just deploy-explorer`.

## Not yet visualized (FastF1 has much more)

The explorer currently shows our *processed* parquets. FastF1 also exposes — and we
could add — raw telemetry traces (speed/throttle/brake/gear + X/Y track position),
sector times and speed-trap speeds, full weather, session results, qualifying and
practice sessions, and race-control messages. The raw telemetry traces are the only
heavy addition; the rest are cheap to add for already-cached races.
