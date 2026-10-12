"""On-by-default kernel switches of the feature-engineering GPU paths.

Each fused-kernel switch is read per call from its own environment variable and is on unless the variable is set to one of
``0 false off no`` (any case, surrounding whitespace ignored). Any other value, including an empty string, leaves it on.
"""

from __future__ import annotations

import os

__all__ = ["fused_switch_enabled"]

_OFF_VALUES = ("0", "false", "off", "no")


def fused_switch_enabled(name: str) -> bool:
    """Whether the on-by-default kernel switch ``name`` is on for this call."""
    return os.environ.get(name, "1").strip().lower() not in _OFF_VALUES
