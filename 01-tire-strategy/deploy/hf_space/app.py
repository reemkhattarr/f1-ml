"""Hugging Face Space entrypoint for the F1 Tire Strategy demo.

The assembled Space mirrors the repo layout (`01-tire-strategy/...`) so the
package's path config resolves unchanged. See ../build_space.py.
"""

import sys

sys.path.insert(0, "01-tire-strategy/src")

from tire_strategy.serving.demo import build_demo

build_demo().launch()
