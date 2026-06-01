"""Hugging Face Space entrypoint for the F1 Data Explorer."""

import sys

sys.path.insert(0, "01-tire-strategy/src")

from tire_strategy.serving.explorer import build_explorer

build_explorer().launch()
