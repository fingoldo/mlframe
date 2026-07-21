"""
Runtime capability detection for polars-ds.

Detection is a plain ``hasattr(Blueprint, "tfidf")``-style presence check, NOT a try-invoke probe on a
synthetic frame -- a method existing on the class does not guarantee it succeeds at call time for every
dtype/platform combination (it may still raise ``NotImplementedError`` or similar). Callers of
:class:`PolarsNativeDispatcher` must still be prepared to handle a runtime failure even when ``has(op)``
returns ``True``; this module only tells you the op EXISTS, not that it WORKS on your specific data.
Positive results are cached for process lifetime; negative results (e.g. polars-ds not installed) are NOT
cached, so a mid-session ``pip install`` is picked up on the next call.

The :class:`PolarsNativeDispatcher` consumes this and routes handler
operations to polars-ds when available, sklearn fallback otherwise.

Public surface:
  * :func:`detect_polars_ds_capabilities` -- returns the cached set
    of capability strings (e.g. ``"blueprint.scale"``, ``"blueprint.impute"``).
  * :class:`PolarsNativeDispatcher` -- ``has(op)`` + ``get_version()``.
  * :func:`reset_capability_cache` -- testing-only reset.

Capability string convention: ``"<namespace>.<op>"``. ``blueprint.X``
for ``Blueprint`` methods, ``pds.X`` for top-level ``polars_ds`` API.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)


# Operations we currently know how to use OR plan to use. Each entry is a plain capability-name string,
# checked via hasattr on the Blueprint class / pds module (see the module docstring for why this is a
# presence check, not a guarantee the op succeeds at call time).
#
# Keep this in sync with the dispatcher consumers.

_BLUEPRINT_OPS = (
    "scale",
    "robust_scale",
    "ordinal_encode",
    "one_hot_encode",
    "impute",
    "conditional_impute",
    "linear_impute",
    "winsorize",
    "polynomial_features",
    "target_encode",
    "woe_encode",
    "iv_encode",
    "rank_hot_encode",
    "kbins_discretize",
    "tfidf",
    "hashing_encode",
    "select_by_std",
    "drop_outliers",
    "int_to_float",
)

_PDS_TOPLEVEL_OPS = (
    "target_encode_oof",
    "target_encode_bayes",
    "woe_encode_oof",
    "tfidf",
    "hashing_vectorize",
    "kbins_discretize",
)


_CACHED_CAPS: Optional[FrozenSet[str]] = None
_CACHED_VERSION: Optional[str] = None


def _polars_ds_available() -> bool:
    """Cheap presence check via ``importlib.util.find_spec`` -- does NOT execute the module body, so a
    user-controlled ``polars_ds.py`` shadow package can't run arbitrary code at FHC construction time.
    """
    return importlib.util.find_spec("polars_ds") is not None


def detect_polars_ds_capabilities() -> FrozenSet[str]:
    """Return the cached set of polars-ds capabilities.

    First call probes; subsequent calls return cached result.
    Negative results -- e.g. polars-ds not installed -- are NOT
    cached (so a re-install picks up next call). Positive results
    cached for process lifetime.
    """
    global _CACHED_CAPS, _CACHED_VERSION

    if _CACHED_CAPS is not None:
        return _CACHED_CAPS

    if not _polars_ds_available():
        # Don't cache absence -- user might pip install during the
        # session and we want to pick that up next time.
        return frozenset()

    try:
        import polars_ds as pds
        from polars_ds.pipeline import Blueprint
    except ImportError:
        return frozenset()

    caps = {f"polars_ds:{getattr(pds, '__version__', 'unknown')}"}

    for op in _BLUEPRINT_OPS:
        if hasattr(Blueprint, op):
            caps.add(f"blueprint.{op}")

    for op in _PDS_TOPLEVEL_OPS:
        if hasattr(pds, op):
            caps.add(f"pds.{op}")

    _CACHED_CAPS = frozenset(caps)
    _CACHED_VERSION = getattr(pds, "__version__", "unknown")

    logger.info(
        "[fhc] polars-ds %s detected with %d capabilities",
        _CACHED_VERSION, len(caps) - 1,  # subtract the version sentinel
    )
    return _CACHED_CAPS


def reset_capability_cache() -> None:
    """Testing-only: clear the cached capability set so subsequent
    calls re-probe. Used by tests that monkey-patch polars-ds APIs."""
    global _CACHED_CAPS, _CACHED_VERSION
    _CACHED_CAPS = None
    _CACHED_VERSION = None


class PolarsNativeDispatcher:
    """Routes handler operations to polars-ds when available, sklearn
    fallback otherwise.

    Instantiate per-FHC (consumes ``backend.prefer_polarsds``):

        dispatcher = PolarsNativeDispatcher(prefer_polarsds=True)
        if dispatcher.has("blueprint.tfidf"):
            ...
        else:
            ...
    """

    def __init__(self, prefer_polarsds: bool = True):
        self._prefer = prefer_polarsds
        self._caps = detect_polars_ds_capabilities() if prefer_polarsds else frozenset()

    def has(self, op: str) -> bool:
        """Returns True iff ``op`` is in the capability set AND
        ``prefer_polarsds=True``."""
        return op in self._caps

    def get_version(self) -> Optional[str]:
        """Return the detected ``polars_ds`` version string, or ``None`` if polars-ds is unavailable / capability detection was skipped."""
        for c in self._caps:
            if c.startswith("polars_ds:"):
                return c.split(":", 1)[1]
        return None

    def __repr__(self) -> str:
        return f"PolarsNativeDispatcher(prefer={self._prefer}, " f"version={self.get_version()!r}, caps={len(self._caps)})"


__all__ = [
    "detect_polars_ds_capabilities",
    "reset_capability_cache",
    "PolarsNativeDispatcher",
]
