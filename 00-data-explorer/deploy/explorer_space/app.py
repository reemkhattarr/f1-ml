"""Hugging Face Space entrypoint for the F1 Data Explorer."""

import sys

sys.path.insert(0, "00-data-explorer/src")

from f1_foundation.explorer import build_explorer, launch_kwargs

build_explorer().launch(**launch_kwargs())
