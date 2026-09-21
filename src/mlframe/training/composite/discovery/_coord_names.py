"""Name test for spatial-coordinate columns, used by the auto-base spatial-coord demoter.

Correlation alone cannot tell an X/Y/Z triplet from a group of synonyms: ``desc_len``/``desc_words``/``desc_lines`` or
``budget_amount``/``budget_per_position`` correlate as tightly as coordinates do. A production run demoted 11-14 such
text-length and budget features per target (including the top-MI ``budget_amount``) as "spatial coords". Synonym groups
are the dedup step's job, which keeps one representative; the spatial demoter now also requires coordinate-like names.
"""

from __future__ import annotations

import re

_COORD_TOKENS = frozenset(
    {
        "x", "y", "z",
        "lat", "latitude", "lon", "lng", "longitude",
        "easting", "northing", "utm",
        "elevation", "altitude", "depth", "tvd", "md",
        "coord", "coords", "coordinate", "coordinates",
    }
)

_CAMEL_SPLIT = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_TOKEN_SPLIT = re.compile(r"[^0-9A-Za-z]+")
_TRAILING_DIGITS = re.compile(r"\d+$")


def is_coordinate_like_name(name: str) -> bool:
    """True when any token of ``name`` (split on non-alphanumerics and camelCase, trailing digits dropped) is a coordinate word."""
    for token in _TOKEN_SPLIT.split(_CAMEL_SPLIT.sub("_", str(name))):
        token = _TRAILING_DIGITS.sub("", token).lower()
        if token in _COORD_TOKENS:
            return True
    return False
