"""Gains alignment and shared-FE-subsample helpers carved verbatim out of ``_fit_impl_core``."""

from __future__ import annotations

import logging

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _gains_paired_with_output(self, predictors_log) -> "np.ndarray | None":
    """Greedy gains in ``get_feature_names_out()`` order: raw support by column index, then advertised engineered recipes by name.

    A raw column the screen picked on its own has a single-index log entry; an engineered column the screen scored is logged under its
    recipe name. Anything without an entry (a raw re-added by a retention pass, an engineered column appended after the screen) gets 0.0.
    Returns ``None`` when ``support_`` is unavailable, so the caller keeps its positional fallback.
    """
    support = getattr(self, "support_", None)
    if support is None:
        return None
    support = np.asarray(support)
    if support.dtype == bool:
        support = np.flatnonzero(support)
    by_index: dict = {}
    by_name: dict = {}
    for entry in predictors_log:
        indices = tuple(entry.get("indices", ()) or ())
        gain = float(entry.get("gain", 0.0))
        if len(indices) == 1:
            by_index.setdefault(int(indices[0]), gain)
        name = entry.get("name")
        if name is not None:
            by_name.setdefault(str(name), gain)
    raw = [by_index.get(int(i), 0.0) for i in support.tolist()]
    recipes = getattr(self, "_engineered_recipes_", None) or []
    advertised = [r for r in recipes if r.extra.get("chain_lookups") is not None or not r.extra.get("requires_refit_for_replay")]
    engineered = [by_name.get(str(r.name), 0.0) for r in advertised]
    return np.asarray(raw + engineered, dtype=np.float64)


def _align_mrmr_gains(self) -> None:
    """Trim/pad ``self.mrmr_gains_`` to exactly ``self.n_features_`` (the ``len(mrmr_gains_) == n_features_``
    public contract). ``mrmr_gains_`` is the greedy log; the final feature count diverges (shorter on a
    degenerate/redundancy/cap/UAED trim, longer when FE/retention appended features the greedy never scored).
    Must be called as the VERY LAST fit step, after every ``n_features_`` mutation - including the group-aware
    final demotion, which drops zero-within-group engineered recipes and lowers ``n_features_`` but does not
    touch ``mrmr_gains_``. Idempotent (byte-identical when already aligned). Best-effort."""
    try:
        _g = getattr(self, "mrmr_gains_", None)
        _nf_final = int(getattr(self, "n_features_", 0) or 0)
        _log = getattr(self, "_predictors_log_", None) or ()
        _repaired = _gains_paired_with_output(self, _log) if _log else None
        if _repaired is not None:
            if _repaired.shape[0] > _nf_final:
                _repaired = _repaired[:_nf_final]
            elif _repaired.shape[0] < _nf_final:
                _repaired = np.concatenate([_repaired, np.zeros(_nf_final - _repaired.shape[0], dtype=np.float64)])
            self.mrmr_gains_ = _repaired
        elif _g is not None and _nf_final >= 0 and _g.shape[0] != _nf_final:
            if _g.shape[0] > _nf_final:
                self.mrmr_gains_ = _g[:_nf_final]
            else:
                self.mrmr_gains_ = np.concatenate([_g, np.zeros(_nf_final - _g.shape[0], dtype=np.float64)])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("mrmr: mrmr_gains_ finalisation failed: %r", e, exc_info=True)


def _resolve_screen_shared_idx(self, data, target_indices):
    """Resolve the fit's one shared FE row draw for the screen (None at small n, meaning full-n screening)."""
    # Resolve the fit's ONE shared row draw BEFORE the screen so the order-1 relevance sweep + FDR
    # floor score on it (screen is the first consumer -> caches the draw -> the FE step reuses the
    # SAME rows). None at small n -> full-n screen, unchanged.
    try:
        from .._fe_sufficient_summary import _get_shared_fe_subsample_idx

        _screen_shared_idx = _get_shared_fe_subsample_idx(self, np.asarray(data[:, int(target_indices[0])]), len(data))
    except Exception as _sub_exc:
        # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
        log_throttle(
            logger,
            "mrmr_fit_shared_fe_subsample_failed",
            logging.WARNING,
            "mrmr: shared FE subsample resolution failed; screening at FULL n: %r",
            _sub_exc,
            exc_info=True,
        )
        _screen_shared_idx = None
    return _screen_shared_idx
