"""Draining the memo caches before a timed cell, and proving the drain happened.

MRMR memoizes whole fits by content hash, in two independent places: a class-level ``_FIT_CACHE`` LRU on
the ``MRMR`` class and a module-level ``_MRMR_IDENTITY_FP_CACHE`` of identity fingerprints that can
short-circuit a fit before it starts. Both are process-wide and neither is reset between fits.

This benchmark's own design guarantees a hit. Arms alternate inside a cell against the SAME training
frame, and the roster runs more than one MRMR-family arm; a second fit on identical content returns
instantly from the memo. The measured number is then not "what this arm costs" but "what a dictionary
lookup costs", and because the memo replays the true selection the RESULT is still correct -- so nothing
downstream looks wrong. A wrong timing that produces a right answer is the failure mode a cost axis is
least able to survive.

So the drain is asserted rather than attempted. ``drain_memo_caches`` reports whether it could VERIFY the
caches are empty afterwards, and a cell that cannot verify records ``memo_drained=False``, which
invalidates its timing rather than silently publishing it. The public ``clear_fit_cache()`` alone is not
enough: it drains only the first of the two caches, and the identity fingerprints are what let a fit be
skipped entirely.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = ["MemoDrain", "drain_memo_caches", "assert_memo_drained"]

#: Import path of the module owning both caches. Probed in ``sys.modules`` rather than imported, because
#: importing it pulls the numba and sklearn subgraph, and a cell whose roster contains no MRMR arm has
#: nothing to drain: an unloaded module cannot hold a memoized fit.
_MRMR_MODULE = "mlframe.feature_selection.filters.mrmr._mrmr_class"


@dataclass(frozen=True)
class MemoDrain:
    """The outcome of one drain attempt, as a record a cell can carry into its results."""

    attempted: bool
    verified: bool
    fit_entries_cleared: Optional[int]
    identity_entries_cleared: Optional[int]
    reason: str

    def as_dict(self) -> Dict[str, Any]:
        """Return the record shape stored on a cell."""
        return {
            "memo_drained": bool(self.verified),
            "memo_drain_attempted": bool(self.attempted),
            "memo_fit_entries_cleared": self.fit_entries_cleared,
            "memo_identity_entries_cleared": self.identity_entries_cleared,
            "memo_drain_reason": self.reason,
        }


def _residual_sizes(module: Any) -> Dict[str, Optional[int]]:
    """Return how many entries each cache still holds, or ``None`` where the cache cannot be inspected."""
    sizes: Dict[str, Optional[int]] = {}
    try:
        sizes["fit"] = len(module.MRMR._FIT_CACHE)
    except Exception as exc:  # an uninspectable cache is unknown, never assumed empty
        logger.warning("cannot size the MRMR fit cache after draining it: %s", exc)
        sizes["fit"] = None
    try:
        sizes["identity"] = len(module._MRMR_IDENTITY_FP_CACHE)
    except Exception as exc:
        logger.warning("cannot size the MRMR identity fingerprint cache after draining it: %s", exc)
        sizes["identity"] = None
    return sizes


def drain_memo_caches() -> MemoDrain:
    """Drain both MRMR memo caches and report whether emptiness could be verified afterwards.

    Returns:
        A :class:`MemoDrain`. ``verified`` is true only when both caches were observed EMPTY after the
        drain, or when the module owning them was never imported in this process. Anything else -- a
        failed clear, a cache that cannot be sized, a residual entry -- is false, and a false value is
        what invalidates the cell's timing.
    """
    module = sys.modules.get(_MRMR_MODULE)
    if module is None:
        # Nothing has fitted an MRMR in this process, so no memo exists to hit. This is a verified state,
        # not an unknown one: the caches live on objects that do not exist yet.
        return MemoDrain(attempted=False, verified=True, fit_entries_cleared=0, identity_entries_cleared=0, reason="the mrmr module is not loaded in this process")

    fit_cleared: Optional[int] = None
    identity_cleared: Optional[int] = None
    try:
        # The public classmethod takes the same canonical lock every cache read and write site uses, so
        # it is the only safe way to drain the fit cache while another thread could be mid-fit.
        fit_cleared = int(module.MRMR.clear_fit_cache())
    except Exception as exc:
        logger.debug("drain_memo_caches: %s", exc, exc_info=True)
        return MemoDrain(attempted=True, verified=False, fit_entries_cleared=None, identity_entries_cleared=None, reason=f"clear_fit_cache() failed: {type(exc).__name__}: {exc}")

    try:
        lock = getattr(module, "_MRMR_IDENTITY_FP_LOCK", None)
        cache = module._MRMR_IDENTITY_FP_CACHE
        if lock is None:
            identity_cleared = len(cache)
            cache.clear()
        else:
            with lock:
                identity_cleared = len(cache)
                cache.clear()
    except Exception as exc:
        logger.debug("drain_memo_caches: %s", exc, exc_info=True)
        return MemoDrain(attempted=True, verified=False, fit_entries_cleared=fit_cleared, identity_entries_cleared=None, reason=f"identity fingerprint cache could not be cleared: {type(exc).__name__}: {exc}")

    residual = _residual_sizes(module)
    if residual["fit"] is None or residual["identity"] is None:
        return MemoDrain(attempted=True, verified=False, fit_entries_cleared=fit_cleared, identity_entries_cleared=identity_cleared, reason="a cache could not be sized after the drain, so emptiness is unproven")
    if residual["fit"] or residual["identity"]:
        return MemoDrain(
            attempted=True,
            verified=False,
            fit_entries_cleared=fit_cleared,
            identity_entries_cleared=identity_cleared,
            reason=f"entries survived the drain: fit={residual['fit']}, identity={residual['identity']}",
        )
    return MemoDrain(attempted=True, verified=True, fit_entries_cleared=fit_cleared, identity_entries_cleared=identity_cleared, reason="both caches observed empty after the drain")


def assert_memo_drained() -> MemoDrain:
    """Drain the caches and RAISE when emptiness cannot be verified.

    For callers that are measuring rather than recording -- an ablation, a timing harness -- where an
    unverifiable drain means the number about to be produced is meaningless and should not be produced at
    all.

    Raises:
        RuntimeError: When the drain could not be verified.
    """
    drain = drain_memo_caches()
    if not drain.verified:
        raise RuntimeError(f"refusing to time a fit whose memo state is unknown: {drain.reason}")
    return drain
