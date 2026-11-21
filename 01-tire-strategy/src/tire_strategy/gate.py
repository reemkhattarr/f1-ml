"""Quality gate: decide whether a newly trained model may replace the live one.

Run after ``train`` in CI, before anything is published::

    python -m tire_strategy.gate artifacts/metadata.json live_metadata.json

The comparison uses the **grouped cross-validation cost** (``cv_model_cost_mean``):
it is computed across every race in the dataset, so it stays comparable when the
holdout season changes (the temporal test set does not — 2025 and 2026 races are
different problems). The gate passes when the candidate

1. beats both baselines (never pit, rule-based) in cross-validation, and
2. is not worse than the live model by more than ``TOLERANCE``.

The tolerance absorbs ordinary noise from new races entering the pool (the CV cost
standard deviation is about 0.01). If there is no live model yet, only rule 1
applies. Exit code 0 = publish, 1 = keep the live model.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

TOLERANCE = 0.02
COST = "cv_model_cost_mean"
BASELINES = ("cv_majority_cost_mean", "cv_rule_cost_mean")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]


def check(candidate: dict, live: dict | None, tolerance: float = TOLERANCE) -> GateResult:
    """Compare two ``metadata.json`` payloads (see ``train.py``)."""
    metrics = candidate.get("metrics", {})
    reasons: list[str] = []
    if COST not in metrics:
        return GateResult(False, [f"candidate has no {COST}; did training finish?"])

    cost = float(metrics[COST])
    passed = True
    for name in BASELINES:
        if name in metrics and cost >= float(metrics[name]):
            passed = False
            reasons.append(f"does not beat {name} ({cost:.4f} >= {float(metrics[name]):.4f})")

    live_cost = (live or {}).get("metrics", {}).get(COST)
    if live_cost is None:
        reasons.append("no live model to compare against")
    elif cost > float(live_cost) + tolerance:
        passed = False
        reasons.append(f"worse than live: {cost:.4f} vs {float(live_cost):.4f} (+{tolerance})")
    else:
        reasons.append(f"CV cost {cost:.4f} vs live {float(live_cost):.4f}")
    return GateResult(passed, reasons)


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() and path.stat().st_size else None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m tire_strategy.gate CANDIDATE_METADATA LIVE_METADATA")
        return 2
    candidate = _load(Path(argv[0]))
    if candidate is None:
        print(f"::error::no candidate metadata at {argv[0]}")
        return 1
    result = check(candidate, _load(Path(argv[1])))
    for reason in result.reasons:
        print(("" if result.passed else "::error::") + reason)
    print("GATE PASSED" if result.passed else "GATE FAILED: keeping the live model")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
