"""Fit-skip signature and the process-wide fit cache used at the start of ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from typing import Any

import numpy as np
# --- end imports ---


def _fit_signature(self, y, X):
    """Signature of this fit (shapes, X/y content hashes, column names, pre-override ctor params) for the in-object
    same-inputs skip, plus the X and y content hashes it folds in."""
    from mlframe.feature_selection.filters.mrmr.shared import (
        full_y_content_hash as _full_y_content_hash,
        full_x_content_hash as _full_x_content_hash,
        hashable_params_signature as _hashable_params_signature,
    )

    _y_hash_for_sig = _full_y_content_hash(y)
    # Fold column-name tuple so two same-shape frames with different column orders / names don't
    # share a fast-path slot.
    _x_cols_sig = None
    if hasattr(X, "columns"):
        try:
            _x_cols_sig = tuple(str(c) for c in X.columns)
        except Exception as exc:
            logger.debug("mrmr: columns-signature hash failed; treating as unknown (forces a cache miss): %r", exc, exc_info=True)
            _x_cols_sig = None
    # Fold X content hash into
    # the shortcut signature. Pre-fix the signature was
    # ``(X.shape, y.shape, y_hash, x_cols)`` - X CONTENT was absent.
    # Refitting the same MRMR instance on a different-content X with
    # identical shape + column names + y silently replayed the prior
    # fit, returning stale ``support_``. Affects sklearn CV with
    # clone=False, partial_fit-style retraining loops, and rolling-
    # window online retraining where shape+column-names+y are
    # constant. The companion ``_FIT_CACHE`` path below already folded
    # ``_full_x_content_hash`` - asymmetric guarantees between the two
    # cache layers. Fold X content hash here so both layers agree.
    _x_hash_for_sig = _full_x_content_hash(X)
    # Fold the selector's OWN parameter signature into the in-object skip signature.
    # Pre-fix the signature was ``(X.shape, y.shape, y_hash, x_hash, x_cols)`` - SELECTOR PARAMS were
    # absent: refitting the same MRMR instance with changed settings (via ``set_params`` or direct
    # attribute assignment, e.g. ``selector.n_features_to_select = 3``) on identical data silently
    # replayed the prior fit, returning a selection computed under the OLD params. Same asymmetric-
    # guarantees bug class as the X-content fix above: the process-wide ``_FIT_CACHE``
    # below already folds ``_hashable_params_signature`` while this layer did not. ``get_params``
    # introspects ``__init__`` arg names and reads CURRENT attribute values at fit time, so params
    # changed after a previous fit are captured on the next ``fit`` call. ``deep=True`` additionally
    # expands nested ``get_params``-bearing objects (``param__subparam``) so in-place mutation of a
    # nested estimator/config also invalidates the skip. On any ``get_params`` failure we fall back
    # to a per-call unique token (identity equality) => never matches => conservative full refit.
    #
    # PRE-OVERRIDE snapshot preferred (bug fix, 05_concurrency_and_statistics.md, found while testing
    # ): ``fit``'s outer wrapper (``_fit_body`` in ``_mrmr_class.py``) applies several
    # TRANSIENT mid-fit overrides (cluster_aggregate_enable, fast-search profile knobs, default-screen-
    # subsample, ...) to ctor-param-named attributes BEFORE calling into ``_fit_impl`` here, then
    # restores them in its ``finally``. Reading ``self.get_params()`` fresh AT THIS POINT would capture
    # those TRANSIENT values instead of the stable, user-visible ctor state, permanently breaking the
    # same-content-skip signature match on every SUBSEQUENT identical fit() for any config where an
    # override actually fires (e.g. the DEFAULT ``cluster_aggregate_enable=True``) - the stored
    # signature would never again match a freshly (post-restore) computed one. ``_pre_fit_ctor_params_
    # snapshot_`` is captured once, pre-override, at the very top of ``_fit_body``; fall back to a live
    # read only if it's absent (a caller invoking ``_fit_impl`` directly, bypassing the wrapper).
    _self_params_sig: Any
    try:
        _pre_fit_snapshot = getattr(self, "_pre_fit_ctor_params_snapshot_", None)
        _self_params_sig = _hashable_params_signature(_pre_fit_snapshot if _pre_fit_snapshot is not None else self.get_params(deep=True))
    except Exception as exc:
        logger.debug("mrmr: ctor-params signature hash failed; using a unique sentinel (forces a cache miss): %r", exc, exc_info=True)
        _self_params_sig = object()
    signature = (X.shape, y.shape, _y_hash_for_sig, _x_hash_for_sig, _x_cols_sig, _self_params_sig)
    return _x_hash_for_sig, _y_hash_for_sig, signature


def _fit_cache_key(self, X, y, _y_hash_for_sig, groups):
    """Content-based key of this fit in the process-wide ``MRMR._FIT_CACHE``; None when either content hash is empty or
    the key cannot be built (the fit then skips the cache rather than risk a wrong replay)."""
    from mlframe.feature_selection.filters.mrmr.shared import (
        content_array_signature as _content_array_signature,
        full_x_content_hash as _full_x_content_hash,
        hashable_params_signature as _hashable_params_signature,
        target_name_signature as _target_name_signature,
    )

    try:
        # deep=True, as the in-object skip signature uses: a nested estimator mutated in place must invalidate this cache too.
        _params_sig = _hashable_params_signature(self.get_params(deep=True))
        _x_sig = _content_array_signature(X)
        _y_sig = _content_array_signature(y)
        # Two targets with statistically-similar sampled cells collide on _y_sig / _x_sig alone and replay one another's support_. Fold full blake2b hashes over BOTH X and y plus the target name to
        # disambiguate; either empty hash => skip cache (don't risk a wrong replay). Symmetric X/y guarantee closes A1#8: the prior 1024-strided X sample alone left a window where a
        # column-wise outlier clip preserving the sampled positions silently replayed the unclipped fit.
        _y_name = _target_name_signature(y)
        # Reuse _y_hash_for_sig computed above; recomputing on 1M-row y costs ~0.5ms per fit and was paid twice pre-fix (A1#15).
        _y_full_hash = _y_hash_for_sig
        _x_full_hash = _full_x_content_hash(X)
        # Under group_aware_mi the relevance MI depends on the GROUP assignment, so two fits on the SAME X/y with
        # DIFFERENT groups must NOT replay one another. Fold a groups content signature into the key (only when
        # group-aware, so the group-naive path stays byte-identical). group_aware_mi itself is already in _params_sig.
        _groups_sig = None
        if getattr(self, "group_aware_mi", False) and groups is not None:
            _groups_sig = _content_array_signature(np.asarray(groups))
        if not _y_full_hash or not _x_full_hash:
            _cache_key = None
        else:
            _cache_key = (_x_sig, _y_sig, _y_name, _y_full_hash, _x_full_hash, _params_sig, _groups_sig)
    except Exception as exc:
        logger.debug("mrmr: fit-cache key construction failed; skipping cache lookup for this fit: %r", exc, exc_info=True)
        _cache_key = None
    return _cache_key


def _same_inputs_skip(self, signature, x_hash):
    """True when refitting on inputs and params identical to the last fit may be skipped (the fitted state already matches).
    An empty X hash (uncacheable) never skips, mirroring the _FIT_CACHE rule."""
    skip = getattr(self, "skip_retraining_on_same_content", None)
    if skip is None:
        skip = getattr(self, "skip_retraining_on_same_shape", True)
    if not (skip and signature == self.signature and x_hash):
        return False
    if self.verbose:
        logger.info("Skipping retraining on the same inputs signature %s", signature)
    return True


def _replay_from_fit_cache(self, cache_key):
    """Replay a prior fit's state from ``MRMR._FIT_CACHE`` on a key hit; True when replayed.

    After sklearn.base.clone() the cloned MRMR has no fitted state so the in-object signature shortcut never fires; this
    process-wide cache catches it. The replay READ of the cached instance's attributes stays inside the SAME locked
    critical section as the lookup: if the cached instance is itself concurrently being re-fit (a shared estimator in a
    service), an unlocked replay could read a torn mix of reset and fitted attributes. Locking the lookup+replay span makes
    the replay see a consistent snapshot either fully before or fully after the concurrent fit's own (also locked) writes.
    """
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import _MRMR_FIT_CACHE_LOCK
    from mlframe.feature_selection.filters.mrmr import MRMR
    from mlframe.feature_selection.filters.mrmr.shared import replay_fitted_state as _replay_fitted_state

    if cache_key is None:
        return False
    with _MRMR_FIT_CACHE_LOCK:
        if cache_key not in MRMR._FIT_CACHE:
            return False
        cached = MRMR._FIT_CACHE[cache_key]
        MRMR._FIT_CACHE.move_to_end(cache_key)
        replayed = _replay_fitted_state(self, cached)
    if self.verbose:
        logger.info("MRMR.fit: _FIT_CACHE hit -- replayed %d fitted attrs from prior fit, skipping cat-FE + permutation.", replayed)
    return True
