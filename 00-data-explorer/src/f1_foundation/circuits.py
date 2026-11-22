"""One name per circuit, whichever data source it came from.

FastF1 and OpenF1 both name circuits by the event location, but a few venues have
two spellings across seasons or sources ("Miami" / "Miami Gardens", "Monaco" /
"Monte Carlo", "Yas Island" / "Yas Marina"). Models one-hot encode the circuit and
the strategy engine keys its parameters by it, so every name goes through
:func:`canonical_circuit` first.
"""

from __future__ import annotations

_ALIASES: dict[str, str] = {
    "Miami Gardens": "Miami",
    "Monte Carlo": "Monaco",
    "Yas Marina": "Yas Island",
    "Yas Marina Circuit": "Yas Island",
    "Montreal": "Montréal",
    "Sao Paulo": "São Paulo",
    "Interlagos": "São Paulo",
    "Singapore": "Marina Bay",
    "Catalunya": "Barcelona",
    "Hungaroring": "Budapest",
}


def canonical_circuit(name: object) -> str:
    """Return the canonical circuit name (unknown names pass through unchanged)."""
    text = str(name).strip()
    return _ALIASES.get(text, text)
