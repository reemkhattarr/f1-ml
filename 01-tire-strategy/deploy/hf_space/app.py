"""Hugging Face Space entrypoint for the F1 Tire Strategy demo.

The assembled Space mirrors the workspace layout (`00-data-explorer/...` and
`01-tire-strategy/...`, with a shared `data/`) so both packages' path config
resolves unchanged. See ../build_space.py.
"""

import sys

sys.path.insert(0, "00-data-explorer/src")
sys.path.insert(0, "01-tire-strategy/src")

from tire_strategy.serving.demo import build_demo

build_demo().launch()
