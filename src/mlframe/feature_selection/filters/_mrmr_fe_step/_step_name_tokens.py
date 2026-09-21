"""Reading raw-variable references out of an engineered feature's name.

The FE step decides whether a candidate is already covered by a surviving feature by looking at which raw variables each name mentions. Two
blocks of ``_step_score`` needed that and each grew its own byte-identical copy, down to a separately aliased ``import re``, differing only in
which gate map the second closed over. One pair of functions serves both, with the pattern compiled once at import rather than leaning on
``re``'s internal cache on every name.
"""

from __future__ import annotations

import re

# A bare single-token raw variable (``a``, ``x12``), NOT a substring of a surrounding function or feature name: the lookarounds are what keep
# ``log`` in ``log(c)`` from reading as a variable ``l`` followed by ``og``.
_BARE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z](?:[a-z]?\d+)?)(?![A-Za-z0-9_])")


def bare_tokens(name: str) -> set:
    """The bare single-token raw-variable references in ``name``."""
    return set(_BARE_TOKEN_RE.findall(name))


def gate_cols_in(name: str, gate_map) -> list:
    """The gate-composite source columns whose own name appears as a substring of ``name``."""
    return [gate_col for gate_col in gate_map if gate_col in name]
