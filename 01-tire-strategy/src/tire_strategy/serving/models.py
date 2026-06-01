"""Loading the trained model artifacts for serving.

Kept free of any web-framework imports so it can be unit-tested and reused
outside FastAPI (e.g. by the Gradio demo).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from tire_strategy.config import ARTIFACTS_DIR


@dataclass
class ModelBundle:
    """Everything needed to serve a prediction."""

    pit_model: Any  # calibrated pit-timing pipeline (has predict_proba)
    compound_model: Any  # compound-choice pipeline (has predict_proba + classes_)
    threshold: float  # cost-optimised pit decision threshold
    slick_compounds: list[str]


def load_bundle(artifacts_dir: Path = ARTIFACTS_DIR) -> ModelBundle:
    """Load the trained models + metadata from ``artifacts_dir``.

    Raises ``FileNotFoundError`` if the artifacts are missing (model not trained
    yet); callers translate that into a 503.
    """
    metadata_path = artifacts_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No trained model at {artifacts_dir}; run `just train` first.")

    metadata = json.loads(metadata_path.read_text())
    return ModelBundle(
        pit_model=joblib.load(artifacts_dir / "pit_model.joblib"),
        compound_model=joblib.load(artifacts_dir / "compound_model.joblib"),
        threshold=float(metadata["pit_threshold"]),
        slick_compounds=list(metadata["slick_compounds"]),
    )
