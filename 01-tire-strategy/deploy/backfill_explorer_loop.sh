#!/usr/bin/env bash
# Autonomous explorer backfill: FastF1 caps at ~500 calls/hour, so the cold
# all-sessions build runs in hourly batches. Each round adds a batch of sessions
# (the builder halts cleanly on the rate limit), rebuilds + redeploys the explorer
# Space, then waits for the limit to reset and repeats — until a round completes
# without hitting the limit (i.e. coverage is done).
#
#   bash 01-tire-strategy/deploy/backfill_explorer_loop.sh
#
# Safe to stop/restart at any time: the FastF1 cache + the resumable builder mean
# no progress is lost.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

MAX_ROUNDS="${1:-15}"
SLEEP_SECS="${2:-3700}"

for round in $(seq 1 "$MAX_ROUNDS"); do
  echo "=== backfill round $round/$MAX_ROUNDS ==="
  out="$(uv run python -m tire_strategy.serving.explorer_data 2>&1)"
  echo "$out" | grep -E "Added|rate limit|Wrote" || true

  # Rebuild + redeploy the explorer Space with whatever coverage we now have.
  uv run python 01-tire-strategy/deploy/build_explorer_space.py >/dev/null 2>&1
  uv run hf upload reemkhattar/f1-data-explorer \
    01-tire-strategy/deploy/explorer_space_build . --repo-type space >/dev/null 2>&1 \
    && echo "redeployed explorer (round $round)" || echo "deploy failed (round $round)"

  if ! echo "$out" | grep -q "rate limit reached"; then
    echo "=== backfill complete (no rate limit this round) ==="
    exit 0
  fi
  echo "rate-limited; sleeping ${SLEEP_SECS}s before the next batch..."
  sleep "$SLEEP_SECS"
done
echo "=== reached MAX_ROUNDS=$MAX_ROUNDS; re-run to continue if needed ==="
