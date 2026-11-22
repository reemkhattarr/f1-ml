"""Export the trained models to plain JSON so a web page can run them.

The pit-timing model is a LightGBM pipeline (one-hot circuit/compound, then
boosted trees) wrapped in isotonic calibration; the compound model is a multiclass
LightGBM pipeline. Both reduce to: build a feature vector, walk every tree, add up
the leaf values, squash (sigmoid / softmax), and for the pit model map through the
calibration curve. This module writes exactly that, in a compact flat-array format
per tree, and :func:`predict_pit` / :func:`predict_compound` evaluate it in Python
so tests can check the JSON against the real models (``site/model.js`` is the
browser twin).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _flatten(tree: dict[str, Any]) -> dict[str, list]:
    """LightGBM nested tree -> flat arrays (children < 0 are leaves: ~index)."""
    feat: list[int] = []
    thr: list[float] = []
    miss: list[int] = []
    dleft: list[int] = []
    left: list[int] = []
    right: list[int] = []
    leaves: list[float] = []

    def walk(node: dict[str, Any]) -> int:
        if "leaf_value" in node or "split_index" not in node:
            leaves.append(round(float(node.get("leaf_value", 0.0)), 7))
            return ~(len(leaves) - 1)
        i = len(feat)
        feat.append(int(node["split_feature"]))
        thr.append(float(node["threshold"]))
        miss.append(_MISSING.get(str(node.get("missing_type", "None")), 0))
        dleft.append(1 if node.get("default_left", True) else 0)
        left.append(0)
        right.append(0)
        left[i] = walk(node["left_child"])
        right[i] = walk(node["right_child"])
        return i

    root = walk(tree["tree_structure"])
    return {"root": [root], "f": feat, "t": thr, "m": miss, "d": dleft, "l": left, "r": right,
            "v": leaves}  # fmt: skip


def _inputs(pipeline: Any) -> list[dict[str, Any]]:
    """Describe each model input column: a numeric field or a one-hot category."""
    pre = pipeline.named_steps["pre"]
    names = list(pre.get_feature_names_out())
    encoder = pre.named_transformers_["categorical"]
    onehot: dict[str, tuple[str, str]] = {}
    for col, cats in zip(encoder.feature_names_in_, encoder.categories_, strict=True):
        for value in cats:
            onehot[f"{col}_{value}"] = (str(col), str(value))
    out = []
    for name in names:
        if name in onehot:
            col, value = onehot[name]
            out.append({"cat": col, "value": value})
        else:
            out.append({"num": name})
    return out


def _booster_json(pipeline: Any) -> dict[str, Any]:
    clf = pipeline.named_steps["clf"]
    dump = clf.booster_.dump_model()
    return {
        "num_class": int(dump.get("num_class", 1)),
        "inputs": _inputs(pipeline),
        "trees": [_flatten(t) for t in dump["tree_info"]],
    }


def export_models(pit_model: Any, compound_model: Any, threshold: float) -> dict[str, Any]:
    """Both models as JSON-serialisable dicts."""
    base = pit_model.calibrated_classifiers_[0]
    pipeline = base.estimator
    pipeline = getattr(pipeline, "estimator", pipeline)  # unwrap FrozenEstimator
    iso = base.calibrators[0]
    pit = _booster_json(pipeline)
    pit["calibration"] = {
        "x": [round(float(v), 7) for v in iso.X_thresholds_],
        "y": [round(float(v), 7) for v in iso.y_thresholds_],
    }
    pit["threshold"] = float(threshold)
    compound = _booster_json(compound_model)
    compound["classes"] = [str(c) for c in compound_model.named_steps["clf"].classes_]
    return {"pit": pit, "compound": compound}


# ----------------------------------------------------------------- Python evaluator
def _vector(model: dict[str, Any], row: dict[str, Any]) -> list[float]:
    vec = []
    for spec in model["inputs"]:
        if "cat" in spec:
            vec.append(1.0 if str(row.get(spec["cat"])) == spec["value"] else 0.0)
        else:
            v = row.get(spec["num"])
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vec.append(math.nan)
            else:
                vec.append(float(v))
    return vec


def _tree_value(tree: dict[str, list], x: list[float]) -> float:
    node = tree["root"][0]
    while node >= 0:
        v = x[tree["f"][node]]
        missing = tree["m"][node]
        if math.isnan(v) and missing != 2:
            v = 0.0
        if (missing == 2 and math.isnan(v)) or (missing == 1 and abs(v) <= 1e-35):
            node = tree["l"][node] if tree["d"][node] else tree["r"][node]
        else:
            node = tree["l"][node] if v <= tree["t"][node] else tree["r"][node]
    return tree["v"][~node]


def _raw(model: dict[str, Any], row: dict[str, Any]) -> list[float]:
    x = _vector(model, row)
    k = model["num_class"]
    sums = [0.0] * k
    for i, tree in enumerate(model["trees"]):
        sums[i % k] += _tree_value(tree, x)
    return sums


def predict_pit(models: dict[str, Any], row: dict[str, Any]) -> float:
    """Calibrated probability of a pit stop within the model's horizon."""
    pit = models["pit"]
    p = 1.0 / (1.0 + math.exp(-_raw(pit, row)[0]))
    cal = pit["calibration"]
    return float(np.interp(p, cal["x"], cal["y"]))


def predict_compound(models: dict[str, Any], row: dict[str, Any]) -> dict[str, float]:
    comp = models["compound"]
    raw = _raw(comp, row)
    m = max(raw)
    exp = [math.exp(v - m) for v in raw]
    total = sum(exp)
    return {c: e / total for c, e in zip(comp["classes"], exp, strict=True)}
