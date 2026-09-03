"""One canonical form for a dtype string, shared by every cache key that has to survive a backend swap.

This lived in two places: ``training.core._phase_train_one_target._canonicalise_dtype`` and
``feature_selection.filters._mrmr_fingerprints._canonicalise_dtype_str``. The second carried a comment saying it
was duplicated on purpose, to avoid an import cycle (mrmr -> training) -- a real constraint, but one solved by
putting the function somewhere BOTH can import rather than by keeping two copies of it.

Two copies of a cache-key rule is a bad kind of duplication: the copies do not fail loudly when they drift, they
just start hashing the same frame to two different keys, and the cache silently stops hitting. This module imports
nothing from mlframe, so it is a leaf for any caller.
"""

from __future__ import annotations

from typing import Any

# Aliases that mean one thing on disk and several things across polars / pandas / numpy. Everything else falls
# through unchanged, so an unknown dtype still produces a stable key rather than being collapsed into a neighbour.
_ALIASES = {
    "boolean": "b",
    "bool": "b",
    "utf8": "s",
    "string": "s",
    "object": "s",
    "str": "s",
    "categorical": "c",
    "category": "c",
}
# Width-carrying families: the canonical form keeps the width, so int64 and int32 stay distinct keys.
# ``uint`` and ``int`` cannot be confused for each other here (``"uint8"`` does not start with ``"int"``), so the
# order of this table is presentational only.
_PREFIXES = (("uint", "u"), ("int", "i"), ("float", "f"))


def canonicalise_dtype(dt: Any) -> str:
    """Canonical short form of a dtype: ``Int64``/``int64`` -> ``i64``, ``Utf8``/``object`` -> ``s``, and so on.

    The same on-disk dtype yields the same string whether the frame is polars or pandas, so an identity-cache
    entry written by one backend is found by the other.
    """
    s = str(dt).strip().lower()
    for prefix, short in _PREFIXES:
        if s.startswith(prefix):
            return short + s[len(prefix) :]
    return _ALIASES.get(s, s)


__all__ = ["canonicalise_dtype"]
