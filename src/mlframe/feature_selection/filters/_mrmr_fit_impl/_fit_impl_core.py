"""``MRMR._fit_impl`` main fit body for ``mlframe.feature_selection.filters.mrmr``.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. ``_fit_impl`` is bound back onto the ``MRMR`` class at the
parent's module bottom, so call sites that invoke ``self._fit_impl(...)``
continue to work unchanged.

Heavy lifting: signature/cache key build, content-hash short-circuit,
sub-sample loop, FE-step orchestration, MI ranking and the per-fold
selection. Many helpers (logger, signature hashing, target coercion)
live in the parent and are imported lazily inside this body to avoid the
``mrmr -> _mrmr_fit_impl -> mrmr`` import cycle.
"""
from __future__ import annotations

import logging
import os
import threading
import warnings
from collections import defaultdict
from timeit import default_timer as timer

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Guards every read-then-mutate sequence on the process-wide ``MRMR._FIT_CACHE`` (lookup + ``move_to_end`` on
# a hit; ``__setitem__`` + ``move_to_end`` + LRU/byte-cap ``popitem`` on store). Concurrent fits - multi-target
# discovery, joblib-threading callers, web-service workers - otherwise race ``popitem``/``__setitem__``/
# ``move_to_end`` on the same OrderedDict and can raise KeyError or evict the wrong entry. RLock so a wrapped
# region may safely re-enter. The companion ``_MRMR_IDENTITY_FP_CACHE`` already had its own lock; this closes the
# same gap for the fit cache. Exposed on the ``MRMR`` class (idempotently, inside the fit body) as
# ``_FIT_CACHE_LOCK`` so any other holder of the cache can take the same canonical lock.
_MRMR_FIT_CACHE_LOCK = threading.RLock()


# _pgn_raw_budget re-exported from the ``_assign_support`` sub-split (Tier F) for backward-compat
# direct imports (e.g. tests/feature_selection/filters/test_mrmr_pgn_engineered_budget.py) that
# import it from this module's own namespace.
from ._assign_support import _pgn_raw_budget  # noqa: F401 - re-exported facade name, imported directly by tests/test_mrmr_pgn_engineered_budget.py

# Above this many bytes of nullable-column data, densify masked columns one-per-``assign`` instead of all at once
# so peak extra RAM stays ~one float64 column rather than ~2x the whole nullable subset (100GB-frame safe).
_NULLABLE_DENSIFY_EAGER_MAX_BYTES = 2 * 1024**3


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
        if _g is not None and _nf_final >= 0 and _g.shape[0] != _nf_final:
            if _g.shape[0] > _nf_final:
                self.mrmr_gains_ = _g[:_nf_final]
            else:
                self.mrmr_gains_ = np.concatenate([_g, np.zeros(_nf_final - _g.shape[0], dtype=np.float64)])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("mrmr: mrmr_gains_ finalisation failed: %r", e, exc_info=True)

"""MRMR._fit_impl main fit body.

The irreducible single function _fit_impl (bound onto the MRMR class
at the mrmr package facade) lives here verbatim. It is LOC-budget exempt:
one giant function cannot be split without distorting the fit control flow.
Its many lazy in-body from ..X import ... imports break the
mrmr -> _mrmr_fit_impl -> mrmr cycle; the small free helpers it calls live
in the sibling _helpers.py.
"""


from ._helpers import _dispatch_default_scorer, _mrmr_cache_bytes_total, fe_decide_on_subsample

def _fit_impl(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | pd.Series | np.ndarray, groups: pd.Series | np.ndarray = None, **fit_params):
    """We run N selections on data subsets, and pick only features that appear in all selections"""
    # Lazy import: ``.mrmr`` re-imports this module at its module bottom for
    # method binding -> any top-level ``from .mrmr import ...`` here would
    # create a hard import cycle that ``tests/test_meta/test_no_import_cycles.py``
    # flags. Python's module cache makes repeat imports cheap.
    from ..mrmr import (
        MRMR,
        _content_array_signature,
        _full_y_content_hash,
        _full_x_content_hash,
        _hashable_params_signature,
        _replay_fitted_state,
        _target_name_signature,
        _target_to_numpy_values,
        RFECV,
        CatBoostClassifier,
        categorize_dataset,
        compute_probabilistic_multiclass_error,
        create_binary_transformations,
        create_unary_transformations,
        screen_predictors,
        sort_dict_by_value,
    )
    # Publish the canonical fit-cache lock on the class so any other holder of ``_FIT_CACHE`` shares it. Idempotent:
    # only set on first fit, never re-bound (re-binding would split the lock identity under concurrent fits).
    if getattr(MRMR, "_FIT_CACHE_LOCK", None) is None:
        MRMR._FIT_CACHE_LOCK = _MRMR_FIT_CACHE_LOCK
    # include_numeric NaN guard: snapshot raw NaN/inf-bearing NUMERIC columns at the VERY START of fit, before
    # _validate_inputs / categorize / any GPU-discretisation path can impute X. include_numeric must skip a column
    # the user supplied with NaN - its quantile-edge transform replay has no NaN bin, so a NaN test value would
    # silently clip to the top bin (train/serve skew). Captured here so a downstream in-place impute (e.g. the GPU
    # categorize path that is active when the harness sets CUDA_PATH) cannot erase the NaN before the candidate
    # scan and defeat the guard.
    _include_numeric_input_nan_cols = set()
    # Hoisted ONCE (y is never reassigned in _fit_impl): the as-numpy target was re-materialised
    # 53x across the FE/screen stages. Same array (read-only consumers); behavior-preserving.
    _y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    # Per-column boolean NaN mask snapshot at fit entry, before any in-place impute (the include_numeric / binned_numeric_agg cat-FE path GPU-categorizes
    # and imputes X in place when CUDA_PATH is set). The missingness-FE family (is_missing__/missingness_count/missingness_pattern) derives its signal
    # from where the input was NaN; it runs AFTER that impute, so it must read this snapshot, not the live (now-finite) X, or the signal is silently erased.
    _fit_entry_nan_mask = {}
    # Both consumers of this snapshot are opt-in and default OFF (missingness-FE family below, and cat-FE's
    # include_numeric branch far downstream): skip the per-column float64-cast + isfinite scan entirely when
    # neither will ever read it, rather than paying it on every fit regardless. Mirrors each consumer's own gate
    # exactly, so this can never diverge into a false skip.
    _cat_fe_cfg_probe = getattr(self, "cat_fe_config", None)
    _will_use_include_numeric_nan_guard = bool(_cat_fe_cfg_probe is not None and getattr(_cat_fe_cfg_probe, "enable", True) and getattr(_cat_fe_cfg_probe, "include_numeric", False))
    _will_use_missingness_fe = (
        bool(getattr(self, "fe_missingness_indicator_enable", False))
        or bool(getattr(self, "fe_missingness_count_enable", False))
        or bool(getattr(self, "fe_missingness_pattern_enable", False))
    )
    if hasattr(X, "columns") and (_will_use_include_numeric_nan_guard or _will_use_missingness_fe):
        for _c in list(X.columns):
            try:
                _cv = X[_c]
                _cv_np = np.asarray(_cv.to_numpy() if hasattr(_cv, "to_numpy") else _cv, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            _nan_mask_c = ~np.isfinite(_cv_np)
            if _nan_mask_c.any():
                _include_numeric_input_nan_cols.add(_c)
                _fit_entry_nan_mask[_c] = _nan_mask_c
    X = self._validate_inputs(X, y)

    # Large-n regression adaptive-quantization gate. The 180-cell campaign showed fixed 20-bin quantile beats MDLP 15/15 on reg n=100k
    # (holdout +0.116 / F1 +0.242) but LOSES at reg n=20k and on classification, so it is gated to the detected (regression AND n>=threshold)
    # regime, and only when the user left both quantization params at defaults. getattr defaults keep this replay-safe on pre-flip pickles.
    if getattr(self, "adaptive_nbins_large_n_reg", False) and getattr(self, "nbins_strategy", None) == "mdlp" and int(getattr(self, "quantization_nbins", 10)) == 10:
        _n_rows_gate = int(X.shape[0]) if hasattr(X, "shape") else 0
        _thr = int(getattr(self, "adaptive_nbins_large_n_reg_threshold", 50_000))
        if _n_rows_gate >= _thr:
            _explicit_tt_gate = getattr(self, "target_type", None)
            if _explicit_tt_gate is not None:
                _tt_str_gate = str(_explicit_tt_gate).lower()
                _is_reg_gate = not ("classif" in _tt_str_gate or _tt_str_gate in ("binary", "multiclass", "multilabel"))
            else:
                _y_arr_gate = np.asarray(y)
                _n_unique_gate = len(np.unique(_y_arr_gate))
                _ratio_gate = len(_y_arr_gate) / max(1, _n_unique_gate)
                _is_float_gate = _y_arr_gate.dtype.kind == "f"
                _is_classification_gate = (not _is_float_gate) and _ratio_gate > 100 and _n_unique_gate <= 64
                _is_reg_gate = not _is_classification_gate
            if _is_reg_gate:
                self.nbins_strategy = None
                self.quantization_nbins = int(getattr(self, "adaptive_nbins_large_n_reg_nbins", 20))
                self._adaptive_nbins_large_n_reg_fired_ = True

    # ----------------------------------------------------------------------------------------------------------------------------
    # Compute inputs/outputs signature
    # ----------------------------------------------------------------------------------------------------------------------------

    # Shape-only signature was too loose: un-cloned MRMR fit on target A, then re-fit on target B with
    # identical (n_rows, n_cols) shape, replayed A's support_ verbatim. Fold the y content hash in.
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
    # 2026-06-10 fix: fold the selector's OWN parameter signature into the in-object skip signature.
    # Pre-fix the signature was ``(X.shape, y.shape, y_hash, x_hash, x_cols)`` - SELECTOR PARAMS were
    # absent: refitting the same MRMR instance with changed settings (via ``set_params`` or direct
    # attribute assignment, e.g. ``selector.n_features_to_select = 3``) on identical data silently
    # replayed the prior fit, returning a selection computed under the OLD params. Same asymmetric-
    # guarantees bug class as the 2026-05-30 X-content fix above: the process-wide ``_FIT_CACHE``
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
    if getattr(self, "skip_retraining_on_same_content", None) if getattr(self, "skip_retraining_on_same_content", None) is not None else getattr(self, "skip_retraining_on_same_shape", True):
        # Empty X hash (uncacheable) => fall through to full fit to
        # avoid risking a wrong replay, mirroring the _FIT_CACHE rule
        # at line 144 below.
        if signature == self.signature and _x_hash_for_sig:
            if self.verbose:
                logger.info("Skipping retraining on the same inputs signature %s", signature)
            return self

    # Process-wide ``_FIT_CACHE`` hit. After sklearn.base.clone() the cloned MRMR has no fitted state so
    # the signature==signature shortcut above never fires. Content-based key (id-based missed every hit
    # because the suite copies X between iterations - different id() but identical content);
    # _content_array_signature returns shape+dtype+10 sampled values, cheap O(1) and statistically unique
    # enough to avoid false positives on real data. Falls through to full fit on any error or miss.
    _cache_key = None
    try:
        _params_sig = _hashable_params_signature(self.get_params(deep=False))
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
    _cached = None
    _replayed = None
    if _cache_key is not None:
        # concurrency audit: the replay READ of ``_cached``'s attributes must stay inside the
        # SAME locked critical section as the lookup, not run after the ``with`` block exits. If the
        # cached instance is itself concurrently being re-fit() (same object, same cache key - a shared/
        # reused estimator in a service), an unlocked replay could read a torn mix of attributes: some
        # already reset for the new in-flight fit, some still holding the old fitted values. Locking the
        # whole lookup+replay span makes the replay see a consistent snapshot either fully before or
        # fully after the concurrent fit's own (also-locked, see the store site below) attribute writes.
        with _MRMR_FIT_CACHE_LOCK:
            if _cache_key in MRMR._FIT_CACHE:
                _cached = MRMR._FIT_CACHE[_cache_key]
                MRMR._FIT_CACHE.move_to_end(_cache_key)
                _replayed = _replay_fitted_state(self, _cached)
    if _cached is not None:
        if self.verbose:
            logger.info(
                "MRMR.fit: _FIT_CACHE hit -- replayed %d fitted attrs " "from prior fit, skipping cat-FE + permutation.",
                _replayed,
            )
        return self

    # ---------------------------------------------------------------------------------------------------------------
    # Inits
    # ---------------------------------------------------------------------------------------------------------------

    # Outer FE-loop runtime-budget guard. screen_predictors honours self.max_runtime_mins on its own; here we additionally
    # short-circuit between FE iterations so a long FE step that finished after the budget elapsed doesn't trigger another.
    start_time = timer()
    ran_out_of_time = False

    # Carry an absolute deadline to the OPTIONAL enrichment FE generators (orth / extra-basis / pair-cross,
    # plus hermite / wavelet / hinge / binned_numeric_agg / pairwise_modular / conditional_gate /
    # cat_interactions / target_encoding) so a single wide-frame enrichment pass that starts before the
    # budget is spent still aborts its per-column / per-pair loop at the deadline instead of running tens
    # of seconds past a tiny max_runtime_mins. Enrichment-only: the core screen / greedy MI is never gated,
    # so an aborted pass still leaves a usable partial selection. Cleared in MRMR.fit's finally (the outer
    # call-site boundary, in _mrmr_class.py) - NOT here, since this function has no single exit point.
    from .._fe_deadline import set_fe_deadline as _set_fe_deadline
    _set_fe_deadline((start_time + self.max_runtime_mins * 60.0) if self.max_runtime_mins is not None else None)

    def _fe_budget_ok() -> bool:
        """Between-FE-step wall-clock gate: True while remaining time under ``max_runtime_mins`` is unspent (or unset)."""
        # Pre-FE univariate generators (extra-basis, wavelet, dispersion, ...) run once before the FE loop and the
        # between-step guard below cannot bound a single long stage; gate each heavy default-ON stage on the remaining
        # wall-clock so an oversized fit handed a small max_runtime_mins aborts within a small multiple of the budget.
        if self.max_runtime_mins is None:
            return True
        return bool((timer() - start_time) / 60.0 < self.max_runtime_mins)

    def _fe_family_on(flag: str, default: bool = False) -> bool:
        """True iff the family's own ``fe_*_enable`` flag is set AND this fit has an FE budget.

        ``fe_max_steps=0`` is the "no feature engineering at all" contract, and it is unconditional: a family
        flag can only ENABLE a family within that budget, never buy its way past it. Reading the budget from
        ``self`` (not the local) so the helper is safe to call from anywhere in the fit.

        Previously only the hybrid-orth / univariate-basis pair honoured this; every other family fired at
        ``fe_max_steps=0``, which made "no FE" mean "no FE except the ~30 default-ON families" and silently
        engineered columns into fits that had explicitly asked for none.
        """
        return bool(getattr(self, flag, default)) and int(getattr(self, "fe_max_steps", 0) or 0) > 0

    dtype = self.dtype

    parallel_kwargs = self._effective_parallel_kwargs()
    n_jobs = self._effective_n_jobs()
    verbose = self.verbose

    prefetch_factor = 4

    fe_max_steps = self.fe_max_steps
    fe_npermutations = self.fe_npermutations
    fe_unary_preset = self.fe_unary_preset
    fe_binary_preset = self.fe_binary_preset
    fe_max_pair_features = self.fe_max_pair_features

    fe_min_nonzero_confidence = self.fe_min_nonzero_confidence
    fe_min_pair_mi = self.fe_min_pair_mi
    fe_min_pair_mi_prevalence = self.fe_min_pair_mi_prevalence
    fe_min_engineered_mi_prevalence = self.fe_min_engineered_mi_prevalence
    fe_good_to_best_feature_mi_threshold = self.fe_good_to_best_feature_mi_threshold
    fe_max_external_validation_factors = self.fe_max_external_validation_factors
    fe_max_polynoms = self.fe_max_polynoms
    fe_print_best_mis_only = self.fe_print_best_mis_only
    fe_smart_polynom_iters = self.fe_smart_polynom_iters
    fe_smart_polynom_optimization_steps = self.fe_smart_polynom_optimization_steps
    fe_min_polynom_degree = self.fe_min_polynom_degree
    fe_max_polynom_degree = self.fe_max_polynom_degree
    fe_min_polynom_coeff = self.fe_min_polynom_coeff
    fe_max_polynom_coeff = self.fe_max_polynom_coeff

    # Convert numpy array to DataFrame if needed
    # Record a sentinel
    # ``self._feature_names_in_synthesized_`` so ``get_feature_names_out``
    # can distinguish ndarray-fit synthesized placeholders from
    # legitimate DataFrame columns the user happened to name
    # ``feature_<int>``. Pre-fix the detection used
    # ``str(n).startswith("feature_")`` heuristically, which
    # misclassified real columns and silently bypassed the sklearn
    # column-drift contract for any user whose DataFrame happened to
    # use that naming (very common after ``pd.DataFrame(arr)`` + rename).
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self._feature_names_in_synthesized_ = True
    else:
        self._feature_names_in_synthesized_ = False

    # EMBEDDING / FREE-TEXT PASSTHROUGH. MI discretisation needs scalar (hashable, orderable) cells; embedding-vector columns (object cells = list/ndarray) and
    # long free-text columns violate that and would crash the discretiser or mis-bin into a useless ~N-level categorical. Detect them here and EXCLUDE them from
    # the working frame so the screen / FE / MI never see them, but PASS THEM THROUGH to the transform output unchanged - the learnable-embedding MLP / recurrent
    # network (and the ``_encode_emb_text_fit`` boundary encoder) are the correct consumers. ``feature_names_in_`` (set below from the full pre-narrow column list)
    # still counts them so the sklearn ``n_features_in_`` contract matches the user's input width; the passthrough indices are re-appended to ``support_`` at
    # fit-end. Default ON (a corrective mechanism; the legacy crash/drop was silently wrong); set ``embedding_passthrough=False`` for the legacy behaviour.
    self._passthrough_features_ = []
    if getattr(self, "embedding_passthrough", True) and isinstance(X, pd.DataFrame):
        from .._mrmr_passthrough import detect_passthrough_columns
        _emb_cols, _text_cols = detect_passthrough_columns(
            X,
            detect_embeddings=getattr(self, "embedding_passthrough_detect_embeddings", True),
            detect_text=getattr(self, "embedding_passthrough_detect_text", True),
        )
        _passthrough = list(_emb_cols) + [c for c in _text_cols if c not in _emb_cols]
        if _passthrough:
            self._passthrough_features_ = _passthrough
            # Column-subset selection shares the underlying column buffers (no row copy) - RAM-safe on 100+ GB frames. The original full column order is recovered
            # at fit-end from ``feature_names_in_`` (built from the pre-narrow list below) so the re-appended passthrough indices land at their true positions.
            _keep_cols = [c for c in (X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)) if c not in set(_passthrough)]
            self._passthrough_full_columns_ = X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
            X = X[_keep_cols]
            if verbose:
                logger.info(
                    "MRMR.fit: routing %d non-scalar column(s) THROUGH feature selection unchanged (embeddings=%s, text=%s); they bypass the MI screen and reach the estimator raw.",
                    len(_passthrough), _emb_cols, _text_cols,
                )

    # NULLABLE-DTYPE DENSIFICATION (gaps_fe_masking-09). A pandas masked-array frame (Int64 / Float64 / boolean +
    # pd.NA) is NOT what the screen / FE-pair numba kernels and the ``dtype.kind=="f"`` NaN guard expect:
    # ``DataFrame.to_numpy()`` on a mixed nullable frame yields object cells holding pd.NA (NOT float64+NaN), so
    # numeric FE families (e.g. conditional_gate) silently skip those columns and the SELECTION diverges from the
    # dense-float64 fit. Densify masked numeric / boolean columns to float64 (pd.NA -> NaN, semantically lossless)
    # so every downstream path is dtype-agnostic. Categorical / string extension columns are left untouched for
    # categorize_dataset (their ``dtype.kind`` is 'O' / 'U', not in the masked numeric set). Default ON: a
    # corrective mechanism (the legacy silent column-skip was wrong), no flag.
    if isinstance(X, pd.DataFrame):
        _nullable_num = [c for c in X.columns if pd.api.types.is_extension_array_dtype(X[c].dtype) and getattr(X[c].dtype, "kind", "O") in ("i", "u", "f", "b")]
        if _nullable_num:
            # A single ``assign`` of every nullable column materialises all the float64 arrays before building the
            # frame (peak ~2x the nullable-column bytes); above the threshold densify one column per ``assign`` so
            # each intermediate frame is freed and peak extra RAM stays ~one column. ``assign`` returns a new frame
            # either way, so the caller's frame is never mutated - the densification stays RAM-safe on 100+ GB frames.
            if len(X) * len(_nullable_num) * 8 <= _NULLABLE_DENSIFY_EAGER_MAX_BYTES:
                X = X.assign(**{c: X[c].astype("float64") for c in _nullable_num})
            else:
                for _nc in _nullable_num:
                    X = X.assign(**{_nc: X[_nc].astype("float64")})
            if verbose:
                logger.info(
                    "MRMR.fit: densified %d nullable masked column(s) to float64 (NaN-preserving): %s",
                    len(_nullable_num), _nullable_num[:8],
                )

    # 2026-05-31 Layer 23 — hybrid orthogonal-polynomial + MI-greedy FE.
    # When ``fe_hybrid_orth_enable=True``, generate basis_n(z) columns for each
    # numeric input column and MI-rank against y; append the top-K winners
    # before screening. The hybrid pipeline lives in ``_orthogonal_univariate_fe``
    # and returns EngineeredRecipe objects so transform() can replay each
    # appended column without re-running the MI ranking (deterministic in X,
    # never references y at replay time).
    #
    # The injection happens BEFORE feature_names_in_ is set so the engineered
    # columns are NOT recorded as raw input features; instead they're
    # pre-registered in ``engineered_recipes`` dict (the same dict the FE-step
    # would populate) and the end-of-fit remap routes them through
    # ``self._engineered_recipes_`` automatically.
    self.hybrid_orth_features_ = []
    # Every column the hybrid-orth family APPENDED to the candidate pool, whether or not it later survived
    # selection. ``hybrid_orth_features_`` is intersected with ``support_`` (survivor-only), so when a sibling
    # FE family emits an equivalent column and wins the greedy, the survivor roster goes empty and there is no
    # way left to tell "the stage never fired" from "it fired and lost". This roster answers that directly.
    self.hybrid_orth_candidates_ = []
    # ADAPTIVE-FREQUENCY Fourier: names of the held-out-validated
    # adaptive sin/cos columns the extra-basis stage emitted. Used by the
    # support-finalisation ADAPTIVE-PROTECTION block to re-add any the MRMR
    # screen dropped. Always present (empty when no adaptive freq detected) so
    # transform / pickle / clone never trip on a missing attribute.
    self._adaptive_fourier_features_ = []
    # HINGE / change-point: names of the held-out-tau-validated
    # hinge legs the change-point stage emitted. Used by the support-
    # finalisation HINGE-PROTECTION block to re-add any the MRMR screen dropped
    # (a single relu leg is MONOTONE -> MI-INVARIANT by the DPI, so the greedy
    # MI screen drops it as redundant with raw x exactly as it drops the adaptive
    # Fourier legs - its value is downstream linear usability, not MI). Always
    # present (empty when hinge off / no kink) so transform / pickle / clone
    # never trip on a missing attribute.
    self._hinge_features_ = []
    # SUFFICIENT-SUMMARY EARLY-STOP verdict. The fitted-attribute mirror of
    # the last sufficient-summary check in the greedy FE loop (a SufficientSummaryVerdict,
    # or None when the early-stop never ran / was disabled). Surfaced so callers can inspect
    # WHY the FE search stopped (residual fraction, max raw MI, maxT floor). Always present
    # so transform / pickle / clone never trip on a missing attribute.
    self.sufficient_summary_ = None
    # Count of FE operator-search iterations actually executed (``_run_fe_step`` calls). The
    # sufficient-summary early-stop reduces this by skipping provably-pointless steps; the
    # biz_value test asserts on it as a DETERMINISTIC work-saved proxy (timing on a contended
    # box is jittery). Always present for transform / pickle / clone.
    self._fe_steps_executed_ = 0
    # PER-GATE FE REJECTION LEDGER (additive): reset the per-fit raw-record list HERE, before
    # ANY FE stage runs (recipe-FE families at L33/L34/L37/L38/L104 + cluster-basis all record
    # via their reject_sink BEFORE the pair-search loop). A later reset would clobber those
    # families' unified-gate abs-MAD floor kills; fe_rejection_ledger_ is built from it at fit-end.
    self._fe_rejection_records_ = []
    # Deferred hinge-leg buffer: the hinge stage detects + held-out-validates the
    # legs early (it needs the raw source columns before pair-FE rewrites them) but
    # DEFERS materialising them into the candidate matrix until support finalisation,
    # so the legs never perturb pair-composite recovery. {name: float64 values} and
    # {name: EngineeredRecipe}. Empty when the operator is off / detects nothing.
    _hinge_deferred_values: dict = {}
    _hinge_deferred_recipes: dict = {}
    _hybrid_orth_pre_recipes: dict = {}
    # Format-agnostic FE seam primitives. CLOSED-FORM families route their DECISION through fe_decide_on_subsample with the
    # NATIVE frame (subsample gather is a small native copy, winners replay on native columns), so a 100+ GB polars frame is
    # never whole-copied. The few OOF / cross-row families that need the full frame gate their pandas materialisation on
    # fe_polars_exceeds (~2 GB, CLAUDE.md eager-conversion rule) and skip above it. Engineered columns append via fe_append_columns.
    from .._fe_frame_ops import fe_to_pandas, fe_append_columns, fe_extract_columns, fe_is_numeric_col, fe_polars_exceeds
    # Snapshot the raw input columns BEFORE any FE stage appends engineered
    # intermediates. The cat_pair / cat_triple auto-detect paths restrict their
    # candidate members to this set so a cross is never built on an engineered
    # column (which cannot be replayed at transform time -> KeyError).
    _raw_input_cols_pre_fe = list(X.columns) if hasattr(X, "columns") else []
    # 2026-06-02 UNIVARIATE-BASIS FE — DEFAULT ON (closes the univariate-
    # nonlinearity gap). The pair-FE path (always on) recovers pair interactions
    # (a*b, a/b, |a-b|) but CANNOT express a single-variable nonlinearity (no
    # pairing makes a clean a**2 / a**3 / |a| out of one column); on a symmetric
    # domain raw ``a`` is uninformative about ``a**2`` (corr ~0), so a univariate
    # quadratic signal was silently MISSED (measured: a**2 corr 0.016, zero
    # engineered features). The orthogonal-basis univariate stage (``a__T2`` ~
    # a**2 etc.) closes that - ``fe_univariate_basis_enable`` (default True)
    # runs JUST the univariate basis FE, uplift-gated via ``min_uplift`` in
    # ``hybrid_orth_mi_fe_with_recipes`` so it is near-no-op when there is no
    # univariate nonlinearity, independent of the heavier pair-CROSS-basis stage
    # which stays behind ``fe_hybrid_orth_enable``. Recovery pinned in
    # ``test_biz_value_mrmr_univariate_basis_fe.py``.
    # fe_max_steps==0 is the documented "no FE at all" contract (see e.g. test_group_aware_mi_mrmr.py's
    # fe_max_steps=0 fixtures): both default-ON families must not fire just because the user never
    # explicitly touched their own enable flag - gate on fe_max_steps>0 too, matching the analogous
    # discrete-structural-operators precedent above (which DOES allow fe_max_steps=0 firing, but only
    # for an operator the caller explicitly opted into via its own flag - neither family here has that
    # explicit-opt-in carve-out, so fe_max_steps=0 disables both unconditionally).
    _hybrid_on = _fe_family_on("fe_hybrid_orth_enable", False) and fe_max_steps > 0
    _univ_basis_on = _fe_family_on("fe_univariate_basis_enable", True) and fe_max_steps > 0
    if (_hybrid_on or _univ_basis_on) and _fe_budget_ok():
        # Polars frames: skip with a warning - hybrid FE pipeline operates on
        # pandas. Native polars support would require a separate code path;
        # not in Layer 23 MVP scope.
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_univariate_fe import (
                hybrid_orth_mi_fe_with_recipes,
                hybrid_orth_mi_pair_fe_with_recipes,
            )

            _y_for_hybrid = _y_np
            # Hybrid MI scoring expects discrete y. Two cases:
            #   (a) Float-encoded discrete labels (0.0/1.0) - safe to cast to int64.
            #   (b) Continuous regression target - truncating to int destroys the
            #       signal (e.g. y in [-2.5, 3.1] all collapses to {-2,-1,0,1,2,3},
            #       6 quasi-balanced bins, MI to any continuous predictor ~0).
            #       Quantile-bin instead so MI scoring sees a meaningful discrete y.
            if _y_for_hybrid.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_hybrid).size)
                if _n_unique <= 32:
                    _y_for_hybrid = _y_for_hybrid.astype(np.int64)
                else:
                    try:
                        _y_for_hybrid = pd.qcut(
                            _y_for_hybrid, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: qcut-based y discretisation failed (heavy ties/NaN); falling back to int-cast: %r", exc, exc_info=True)
                        # qcut can fail when y has heavy ties or NaN. Fall back to
                        # int-cast so the pipeline still runs (signal may degrade
                        # but does not crash the fit).
                        _y_for_hybrid = _y_for_hybrid.astype(np.int64)
            _h_degrees = tuple(int(d) for d in self.fe_hybrid_orth_degrees)
            _h_basis = str(self.fe_hybrid_orth_basis)
            _h_top_k = int(self.fe_hybrid_orth_top_k)
            # The pair-CROSS-basis stage is heavier and only runs under the
            # explicit ``fe_hybrid_orth_enable`` opt-in; the default-on
            # univariate-basis path (``fe_univariate_basis_enable`` only) is
            # univariate-only so it stays cheap + near-no-op (uplift-gated).
            _h_pair_enable = bool(self.fe_hybrid_orth_pair_enable) and _hybrid_on
            _h_pair_max_degree = int(self.fe_hybrid_orth_pair_max_degree)
            # Restrict the source pool to numeric columns the caller passed
            # via factors_names_to_use (when set); otherwise the hybrid
            # pipeline auto-routes to all numeric columns of X.
            _h_cols = None
            if getattr(self, "factors_names_to_use", None):
                _h_cols = [c for c in self.factors_names_to_use if c in X.columns]
            _X_before_hybrid_cols = list(X.columns)
            # 2026-06-01 Layer 85 — default-scorer routing for the L21
            # univariate basis-selection stage. Non-"plug_in" values
            # route the univariate dispatch through one of the alternate
            # scorers (CMIM, JMIM, KSG, copula, dCor, HSIC, TC, lasso,
            # elasticnet, auto, ensemble, meta). Recipes still emit as
            # ``orth_univariate``; only the SELECTION differs. The pair
            # stage (L22) is skipped under non-default routing because
            # the alternate scorers operate on univariate columns only.
            # "plug_in" preserves the master-branch byte-identical
            # behaviour: pair stage runs IFF ``pair_enable=True``.
            _default_scorer = str(getattr(
                self, "fe_hybrid_orth_default_scorer", "plug_in",
            ))
            if _default_scorer == "plug_in":
                if _h_pair_enable:
                    # Decide on the shared FE subsample; winners replayed at full n.
                    X_h, _uni_sc, _cross_sc, _recipes = fe_decide_on_subsample(
                        hybrid_orth_mi_pair_fe_with_recipes,
                        X, _y_for_hybrid,
                        subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                        subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                        shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                        cols=_h_cols,
                        degrees=_h_degrees,
                        basis=_h_basis,
                        top_k=_h_top_k,
                        top_pair_count=_h_top_k,
                        pair_max_degree=_h_pair_max_degree,
                    )
                else:
                    # Decide on the shared FE subsample (native gather, no whole-frame copy); winners replay at full n.
                    X_h, _uni_sc, _recipes = fe_decide_on_subsample(
                        hybrid_orth_mi_fe_with_recipes,
                        X, _y_for_hybrid,
                        subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                        subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                        shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                        cols=_h_cols,
                        degrees=_h_degrees,
                        basis=_h_basis,
                        top_k=_h_top_k,
                    )
            else:
                def _default_scorer_run(_Xs, _ys, **_kw):
                    """Adapt ``_default_scorer`` to the ``fe_decide_on_subsample`` callable signature (subsample-scorer callback)."""
                    return _dispatch_default_scorer(_default_scorer, X=_Xs, y=_ys, **_kw)
                X_h, _uni_sc, _recipes = fe_decide_on_subsample(
                    _default_scorer_run,
                    X, _y_for_hybrid,
                    subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                    subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                    shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                    cols=_h_cols,
                    degrees=_h_degrees,
                    basis=_h_basis,
                    top_k=_h_top_k,
                )
            # Identify appended columns vs the pre-hybrid X.
            _appended = [c for c in X_h.columns if c not in _X_before_hybrid_cols]
            if _appended:
                X = fe_append_columns(X, fe_extract_columns(X_h, _appended))
                self.hybrid_orth_features_ = list(_appended)
                for _r in _recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth: appended %d engineered " "column(s) (univariate + pair): %s",
                        len(_appended),
                        _appended[:8],
                    )
        except Exception as _h_exc:
            logger.warning(
                "MRMR.fit hybrid_orth FE raised %s: %s; continuing " "without hybrid-FE columns.",
                type(_h_exc).__name__,
                _h_exc,
            )
        # 2026-05-31 Layer 32 — extra-basis (B-spline / Fourier) FE stage.
        # Runs only when the master hybrid switch is on AND the user
        # opted in via a non-empty ``fe_hybrid_orth_extra_bases`` tuple.
        # Complementary to the polynomial path: spline catches threshold
        # rules, Fourier catches periodic patterns. Recipes are
        # closed-form (no y), replay safe.
        _extra_bases_cfg = tuple(getattr(self, "fe_hybrid_orth_extra_bases", ()) or ())
        # Defensive guard: the polynomial-stage ``try:`` may have raised
        # before defining ``_y_for_hybrid`` / ``_h_top_k``. Bind safe
        # defaults so the extra-basis stage can still run.
        try:
            _y_for_extra = _y_for_hybrid
        except NameError:
            _y_for_extra = _y_np
            if _y_for_extra.dtype.kind in "fc":
                if int(np.unique(_y_for_extra).size) <= 32:
                    _y_for_extra = _y_for_extra.astype(np.int64)
                else:
                    try:
                        _y_for_extra = pd.qcut(
                            _y_for_extra, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the hybrid-orth extra-basis FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_extra = _y_for_extra.astype(np.int64)
        _top_k_for_extra = int(getattr(self, "fe_hybrid_orth_top_k", 5))
        # Effective extra-basis set. Two independent contributors:
        #   * the EXPLICIT ``fe_hybrid_orth_extra_bases`` config, but only under
        #     the heavy ``fe_hybrid_orth_enable`` master switch (legacy gate - a
        #     user who set the config but not the master expected a no-op);
        #   * the DEFAULT-ON Fourier univariate basis (``fe_univariate_fourier_enable``),
        #     which runs in the univariate path WITHOUT the master switch so a
        #     pure oscillatory signal (sin/cos) is recovered by default. The
        #     extra-basis stage is uplift + multiple-comparison gated downstream,
        #     so adding Fourier is near-no-op when there is no oscillation.
        _univ_fourier_on = _fe_family_on("fe_univariate_fourier_enable", True)
        _eff_extra_bases = tuple(_extra_bases_cfg) if (_extra_bases_cfg and _hybrid_on) else ()
        # The default-on Fourier univariate basis is part of the plug-in univariate dispatch. Under an alternate ``fe_hybrid_orth_default_scorer`` (cmim / jmim / ksg / ...) the routing
        # runs ONLY the univariate basis-selection for that scorer (the pair stage is likewise skipped above); the Fourier extra basis is a plug-in-path addition, so adding it under
        # alternate routing would emit columns the routed scorer never selected and diverge from a direct call to that scorer. Gate it to plug-in routing.
        try:
            _extra_basis_scorer_ok = _default_scorer == "plug_in"
        except NameError:
            _extra_basis_scorer_ok = True
        if _univ_fourier_on and _univ_basis_on and _extra_basis_scorer_ok and "fourier" not in _eff_extra_bases:
            _eff_extra_bases = (*_eff_extra_bases, "fourier")
        if _eff_extra_bases:
            try:
                from .._orthogonal_univariate_fe import (
                    hybrid_orth_extra_basis_fe_with_recipes,
                )

                _fourier_freqs = tuple(float(f) for f in getattr(self, "fe_hybrid_orth_fourier_freqs", (1.0, 2.0)))
                _spline_knots = int(getattr(self, "fe_hybrid_orth_spline_knots", 5))
                _fourier_powers = tuple(int(p) for p in getattr(self, "fe_hybrid_orth_fourier_powers", (1, 2)))
                _X_before_extra_cols = list(X.columns)
                # Build the extra basis (Fourier/spline) on RAW columns only -
                # EXCLUDE the already-appended poly-basis columns (``a__T2`` ...).
                # Running Fourier on an engineered column would produce a NESTED
                # recipe (``a__T2__sin1``) whose transform-replay needs ``a__T2``
                # materialised first; the 1-deep replay path can't order that and
                # raises KeyError('a__T2') at transform time. Keeping the source
                # scope to raw columns keeps every extra-basis recipe 1-deep and
                # replayable (and honours factors_names_to_use when set).
                _already_eng_for_extra = set(self.hybrid_orth_features_ or [])
                if getattr(self, "factors_names_to_use", None):
                    _e_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _already_eng_for_extra]
                else:
                    _e_cols = [c for c in X.columns if c not in _already_eng_for_extra]
                # ADAPTIVE-FREQUENCY Fourier: default ON. The
                # fixed grid {1, 2} misses arbitrary-period oscillations
                # (sin(3.7*x), sin(5.3*x)); the adaptive detector sweeps a
                # coarse z-space grid + local-refines + held-out-validates
                # the dominant frequency per column, n-gated at >= 800 rows
                # (smaller n false-positives a chance frequency). The
                # emitted adaptive sin/cos recipes are tagged adaptive=True
                # and PROTECTED past screening below (a single leg has low
                # marginal MI - phase - so the screen would drop the
                # held-out-validated pair otherwise).
                _fourier_adaptive = bool(getattr(self, "fe_univariate_fourier_adaptive", True))
                _fourier_adaptive_mvc = float(
                    getattr(
                        self,
                        "fe_univariate_fourier_adaptive_min_val_corr",
                        0.15,
                    )
                )
                # ADAPTIVE-CHIRP: second argument-warp path. Runs
                # the same held-out detector on u = sign(z)*z**2 so a growing-
                # frequency chirp (sin(2*pi*f*z**2)) the linear-argument
                # Fourier cannot express is recovered. Emits __qsin/__qcos
                # legs tagged adaptive=True -> captured below + protected past
                # the screen + dedup-exempt exactly like the linear legs.
                _fourier_chirp = bool(getattr(self, "fe_univariate_fourier_chirp", True))
                _fourier_chirp_mvc = float(
                    getattr(
                        self,
                        "fe_univariate_fourier_chirp_min_val_corr",
                        0.15,
                    )
                )
                # Detect frequencies + rank MI on the shared subsample (native gather, no whole-frame copy - the
                # periodogram detector is the dominant orth-FE CPU cost); winners replay at full n via apply_recipe.
                X_e, _e_scores, _e_recipes = fe_decide_on_subsample(
                    hybrid_orth_extra_basis_fe_with_recipes,
                    X, _y_for_extra,
                    subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                    subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                    shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                    cols=_e_cols,
                    extra_bases=_eff_extra_bases,
                    fourier_freqs=_fourier_freqs,
                    fourier_powers=_fourier_powers,
                    spline_knots=_spline_knots,
                    top_k=_top_k_for_extra,
                    fourier_adaptive=_fourier_adaptive,
                    fourier_adaptive_min_val_corr=_fourier_adaptive_mvc,
                    fourier_chirp=_fourier_chirp,
                    fourier_chirp_min_val_corr=_fourier_chirp_mvc,
                    max_adaptive_cols=getattr(self, "fe_univariate_fourier_adaptive_max_cols", None),
                )
                _e_appended = [c for c in X_e.columns if c not in _X_before_extra_cols]
                if _e_appended:
                    X = fe_append_columns(X, fe_extract_columns(X_e, _e_appended))
                    # Extend hybrid_orth_features_ with the extra-basis winners
                    # so the downstream remap / transform pipeline handles them
                    # exactly like the polynomial winners.
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_e_appended)
                    for _r in _e_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    # Capture ADAPTIVE-tagged Fourier feature names so the
                    # support-finalisation block can re-add any the MRMR
                    # screen dropped (held-out-validated, must survive).
                    _adaptive_names = [
                        _r.name for _r in _e_recipes
                        if getattr(_r, "kind", None) == "orth_fourier"
                        and bool(dict(getattr(_r, "extra", {})).get("adaptive", False))
                        and _r.name in set(_e_appended)
                    ]
                    if _adaptive_names:
                        _prev_adaptive = list(getattr(self, "_adaptive_fourier_features_", None) or [])
                        self._adaptive_fourier_features_ = _prev_adaptive + _adaptive_names
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth extra-basis: appended %d " "engineered column(s) (spline/fourier): %s",
                            len(_e_appended),
                            _e_appended[:8],
                        )
            except Exception as _e_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth extra-basis FE raised %s: %s; " "continuing without extra-basis columns.",
                    type(_e_exc).__name__,
                    _e_exc,
                )
    # 2026-06-09 — HINGE / piecewise-linear change-point basis stage.
    # Independent opt-in via ``fe_hinge_enable`` (does NOT require
    # ``fe_hybrid_orth_enable``): captures a SLOPE CHANGE at a data-dependent
    # threshold ``y = a*x + b*max(x-tau,0)`` (pricing tiers / dose-response /
    # saturation) that the catalog cannot - ``numeric_rounding`` is piecewise-
    # CONSTANT, the cubic B-spline rounds off a sharp kink at its fixed quantile
    # knots, and orth-poly needs a high degree + rings (Gibbs) around the kink.
    # The breakpoint ``tau`` is detected by scanning inner-quantile cuts for the
    # max 2-segment-SSE drop, HELD-OUT-validated on the ``%3`` stride slice (the
    # 2-segment fit must beat plain linear OOS) so a chance kink / pure noise
    # admits no hinge. Emitted ``relu(x-tau)`` / ``relu(tau-x)`` legs carry a
    # genuinely different LINEAR shape from raw x, so they clear the standard
    # MI-uplift gate (unlike the MI-invariant isotonic / RankGauss). Recipes
    # (``hinge_basis``) store only ``{tau, side}`` - no y - so replay is the
    # pure function ``np.maximum(x-tau,0)``, leak-free. On a monotone target a
    # hinge can be near-collinear with raw x -> the downstream cross-stage
    # Spearman dedup drops it (no duplicate columns survive).
    if _fe_family_on("fe_hinge_enable", False) and _fe_budget_ok():
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._hinge_basis_fe import hybrid_hinge_fe_with_recipes
            # The hinge detector + admission are REGRESSION-style (2-segment
            # SSE breakpoint search + held-out incremental linear-R^2 gate),
            # so they want the RAW continuous y - NOT the qcut-to-10-bins
            # coercion the MI-based FE stages use. Quantile-binning a
            # monotone slope-change target (y = a*x + b*relu(x-tau)) collapses
            # the saturating top tier into one bin and DESTROYS the very slope
            # change the hinge detects (measured: qcut y -> 0 breakpoints
            # found; raw y -> tau recovered). Raw class codes work for a
            # discrete classification y too (the linear-fit slope detection is
            # scale/shift invariant). y carries no leak: the recipe stores only
            # {tau, side}, never y.
            _y_for_hinge = _y_np
            _y_for_hinge = np.asarray(_y_for_hinge, dtype=np.float64).ravel()
            # Seed pool restricted to RAW source columns: a hinge built on a
            # prior-stage engineered column would create a recipe whose
            # src_name references an engineered column absent at transform
            # time (KeyError on replay). Honour factors_names_to_use.
            _hinge_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _hinge_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hinge_already_appended and fe_is_numeric_col(X, c)]
            else:
                _hinge_cols = [c for c in X.columns if c not in _hinge_already_appended and fe_is_numeric_col(X, c)]
            _hinge_top_k = int(getattr(self, "fe_hinge_top_k", 5))
            _hinge_max_bp = int(getattr(self, "fe_hinge_max_breakpoints", 2))
            _hinge_emit_ind = bool(getattr(self, "fe_hinge_emit_indicator", False))
            _hinge_mvu = float(getattr(self, "fe_hinge_min_heldout_r2_uplift", 0.02))
            _X_before_hinge_cols = list(X.columns)
            X_h, _h_scores, _h_recipes = hybrid_hinge_fe_with_recipes(
                X, _y_for_hinge,
                cols=_hinge_cols,
                max_breakpoints=_hinge_max_bp,
                emit_indicator=_hinge_emit_ind,
                min_heldout_r2_uplift=_hinge_mvu,
                top_k=_hinge_top_k,
            )
            _h_appended = [c for c in X_h.columns if c not in _X_before_hinge_cols]
            if _h_appended:
                # DEFERRED MATERIALISATION: the hinge legs are a
                # TERMINAL univariate linear-usability stage - they must NOT
                # enter the pair-FE / screening candidate matrix, or (a) the
                # pair search consumes a leg as an operand (replacing a clean
                # raw operand with a hinge-transformed one) and (b) a leg's
                # high marginal MI crowds the genuine pair composites out of
                # selection (measured on y=a**2/b+log(c)*sin(d): the legs on
                # b/d displaced div(sqr(a),abs(b)) / mul(log(c),sin(d))). So
                # we do NOT append the legs to X here; we BUFFER the leg values
                # + recipes and materialise + protect them only at support
                # finalisation (after the FE loop has recovered the composites
                # untouched). This keeps the hidden-champion win (a pure
                # slope-change column with no competing composite still gets
                # its leg) without regressing multi-signal pair recovery.
                _hinge_deferred_values = {c: np.asarray(X_h[c].to_numpy(), dtype=np.float64) for c in _h_appended}
                _hinge_deferred_recipes = {_r.name: _r for _r in _h_recipes if _r.name in set(_h_appended)}
                if verbose:
                    logger.info(
                        "MRMR.fit hinge change-point FE: detected %d held-out-" "validated leg(s) (deferred to support finalisation): %s",
                        len(_h_appended),
                        _h_appended[:8],
                    )
        except Exception as _h_exc:
            logger.warning(
                "MRMR.fit hinge change-point FE raised %s: %s; " "continuing without hinge columns.",
                type(_h_exc).__name__,
                _h_exc,
            )
    # 2026-05-31 Layer 56 — TRI-PRODUCT cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable): captures
    # genuine 3-way interactions like 3-way XOR and price*quantity*count
    # that the pair stage cannot. O(seed_k^3 * deg^3) candidate count is
    # bounded by seed_k=4 default. Recipes (``orth_triplet_cross``) replay
    # from X only, no y, leakage-free by construction.
    # The GBM seeder (#6) opens 3-way generation via order-3-floored explicit triples
    # (``_seeded_triplets_names_``); run the triplet stage for those even when the legacy
    # univariate-seeded triplet path (``fe_hybrid_orth_triplet_enable``) is OFF.
    _gbm_seeded_triplet_names = list(getattr(self, "_seeded_triplets_names_", []) or [])
    from ._hybrid_orth_family_variants import _hybrid_orth_family_variants

    X = _hybrid_orth_family_variants(
        self,
        X=X,
        y=y,
        verbose=verbose,
        _y_np=_y_np,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _gbm_seeded_triplet_names=_gbm_seeded_triplet_names,
        _fe_family_on=_fe_family_on,
    )
    # 2026-05-21 revert of Wave 29 P1 polars->pandas coercion. That
    # coercion was added on the premise that downstream ``X[target_name]
    # = y`` mutation assumed pandas and would raise on polars; but the
    # ``_is_polars_input`` branch immediately below (line ~1326) ALREADY
    # handles polars via ``X.with_columns(target_series)``. The Wave 29
    # coercion was a false-positive fix that killed the zero-copy
    # polars promise (test_mrmr_fe_zero_copy_polars regressed -
    # ``pl.DataFrame.to_pandas()`` was called 1x per fit on 100+ GB
    # production frames). Leaving polars frames untouched so the
    # native branch fires.

    # 2026-05-31 Layer 26 — generic MI-greedy FE constructor (sibling to the
    # hybrid orthogonal-polynomial stage above). Same wiring pattern: opt-in
    # via ``fe_mi_greedy_enable=True``, default OFF preserves byte-identical
    # behaviour. The seed pool is the RAW columns of X (NOT the post-hybrid
    # augmented frame) so the two stages can't compound transforms (e.g.
    # ``log(x__He2)``); each constructor explores its own design space and
    # the union of winners is screened by MRMR.
    self.mi_greedy_features_ = []
    _mi_greedy_pre_recipes: dict = {}
    if _fe_family_on("fe_mi_greedy_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._mi_greedy_fe import greedy_mi_fe_construct_with_recipes
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
            # W6: record abs-MAD floor kills in the mi_greedy stage into the
            # FE rejection ledger (pure-record; selection unchanged).
            _mig_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _mig_reject_sink(**_kw):
                """Reject-sink callback for the greedy MI-construction FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_mig_step, **_kw)

            _y_for_mig = _y_np
            if _y_for_mig.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_mig).size)
                if _n_unique <= 32:
                    _y_for_mig = _y_for_mig.astype(np.int64)
                else:
                    try:
                        _y_for_mig = pd.qcut(
                            _y_for_mig, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the MI-greedy FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_mig = _y_for_mig.astype(np.int64)
            # Restrict the MI-greedy seed pool to RAW source columns only
            # (i.e. exclude hybrid-orth-appended columns from the prior
            # stage). Compound transforms like ``log(He2(x))`` would
            # create recipes whose ``src_names`` reference an engineered
            # column that does not exist at transform time - replay
            # would KeyError. Each constructor explores its OWN design
            # space; the union of winners is screened by MRMR.
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _mig_cols = None
            if getattr(self, "factors_names_to_use", None):
                _mig_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _mig_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _X_before_mig_cols = list(X.columns)
            X_mg, _mig_scores, _mig_recipes = fe_decide_on_subsample(
                greedy_mi_fe_construct_with_recipes,
                X, _y_for_mig,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_mig_cols,
                seed_cols_count=int(self.fe_mi_greedy_seed_cols_count),
                top_k=int(self.fe_mi_greedy_top_k),
                include_unary=bool(self.fe_mi_greedy_include_unary),
                include_binary=bool(self.fe_mi_greedy_include_binary),
                reject_sink=_mig_reject_sink,
            )
            _mig_appended = [c for c in X_mg.columns if c not in _X_before_mig_cols]
            if _mig_appended:
                X = fe_append_columns(X, fe_extract_columns(X_mg, _mig_appended))
                self.mi_greedy_features_ = list(_mig_appended)
                for _r in _mig_recipes:
                    _mi_greedy_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit mi_greedy: appended %d engineered " "column(s): %s",
                        len(_mig_appended),
                        _mig_appended[:8],
                    )
        except Exception as _mig_exc:
            logger.warning(
                "MRMR.fit mi_greedy FE raised %s: %s; continuing " "without MI-greedy columns.",
                type(_mig_exc).__name__,
                _mig_exc,
            )

    # 2026-05-31 Layer 60 — CMI-greedy FE constructor (sibling to Layer 26).
    # Ranks the same candidate library by ``CMI(candidate; y | support)``
    # instead of marginal ``MI(candidate; y)`` so duplicate-signal transforms
    # (``log_abs(x)`` + ``square(x)`` both monotone in |x|) cannot all be
    # picked: once one is in the support, the others' CMI collapses near
    # zero. Winners are MERGED into ``mi_greedy_features_`` (same recipe
    # kind ``mi_greedy_transform``) so downstream end-of-fit remap and
    # transform-time replay are shared infrastructure. Seed pool excludes
    # both prior hybrid-orth and prior marginal-MI-greedy engineered cols
    # (same rationale: replay must not reference engineered sources).
    if _fe_family_on("fe_mi_greedy_cmi_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._mi_greedy_cmi_fe import greedy_cmi_fe_construct_with_recipes

            _y_for_cmi = _y_np
            if _y_for_cmi.dtype.kind in "fc":
                _n_unique_cmi = int(np.unique(_y_for_cmi).size)
                if _n_unique_cmi <= 32:
                    _y_for_cmi = _y_for_cmi.astype(np.int64)
                else:
                    try:
                        _y_for_cmi = pd.qcut(
                            _y_for_cmi, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the CMI-greedy FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_cmi = _y_for_cmi.astype(np.int64)
            _eng_already_appended = set(getattr(self, "hybrid_orth_features_", None) or []) | set(self.mi_greedy_features_ or [])
            if getattr(self, "factors_names_to_use", None):
                _cmi_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _eng_already_appended]
            else:
                _cmi_cols = [c for c in X.columns if c not in _eng_already_appended]
            _X_before_cmi_cols = list(X.columns)
            X_cmi, _cmi_scores, _cmi_recipes = fe_decide_on_subsample(
                greedy_cmi_fe_construct_with_recipes,
                X, _y_for_cmi,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_cmi_cols,
                seed_cols_count=int(self.fe_mi_greedy_cmi_seed_cols_count),
                top_k=int(self.fe_mi_greedy_cmi_top_k),
                include_unary=bool(getattr(self, "fe_mi_greedy_include_unary", True)),
                include_binary=bool(getattr(self, "fe_mi_greedy_include_binary", True)),
                min_cmi_gain=float(self.fe_mi_greedy_cmi_min_gain),
                # Was hardcoded to 0xC011 inside
                # greedy_cmi_fe_construct regardless of random_state, correlating the CMI noise-floor
                # permutation across nominally-independent bootstrap/multi-seed replicates.
                seed=int(getattr(self, "random_seed", 0) or 0),
            )
            _cmi_appended = [c for c in X_cmi.columns if c not in _X_before_cmi_cols]
            if _cmi_appended:
                X = fe_append_columns(X, fe_extract_columns(X_cmi, _cmi_appended))
                # Merge into the existing mi_greedy_features_ list so
                # end-of-fit dedup / remap / pickle treat both stages
                # uniformly. Skip names already present (the two stages
                # share the engineered-column namespace; CMI ones that
                # happen to collide with Layer-26 picks are dropped by
                # name-equality here).
                _existing = set(self.mi_greedy_features_ or [])
                for _c in _cmi_appended:
                    if _c not in _existing:
                        self.mi_greedy_features_.append(_c)
                        _existing.add(_c)
                for _r in _cmi_recipes:
                    if _r.name not in _mi_greedy_pre_recipes:
                        _mi_greedy_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit mi_greedy_cmi: appended %d engineered " "column(s): %s",
                        len(_cmi_appended),
                        _cmi_appended[:8],
                    )
        except Exception as _cmi_exc:
            logger.warning(
                "MRMR.fit mi_greedy_cmi FE raised %s: %s; continuing " "without CMI-greedy columns.",
                type(_cmi_exc).__name__,
                _cmi_exc,
            )

    # 2026-05-31 Layer 33 — K-fold target encoding for raw categorical
    # columns. Runs after hybrid + MI-greedy because TE is the standard
    # prod pattern for cardinality > 5 categoricals that the other two
    # stages do not touch. Recipes (kind ``kfold_target_encoded``) carry
    # only the full-data per-category lookup - no y at replay time.
    # Engineered columns route through ``hybrid_orth_features_`` so the
    # end-of-fit remap treats them as engineered features (same routing
    # as Layer 23 / 26 / 32).
    self.kfold_te_features_ = []
    _kfold_te_pre_recipes: dict = {}
    _binned_agg_pre_recipes: dict = {}
    if _fe_family_on("fe_kfold_te_enable", False):
        # K-fold target encoding is an OOF stat (no closed-form subsample-replay), so it needs the full frame: gate the
        # polars->pandas materialisation on size and skip a > ~2 GiB frame rather than whole-copy it (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: fe_kfold_te_enable=True but X is a large polars frame (> ~2 GiB); K-fold target encoding needs a "
                "full-frame OOF decision and is skipped to avoid a whole-frame to_pandas copy. Materialise a subset or "
                "pass pandas if you need it.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._target_encoding_fe import (
                    kfold_target_encode_with_recipes,
                )

                _te_cols_cfg = tuple(getattr(self, "fe_kfold_te_cols", ()) or ())
                # Explicit empty tuple -> auto-detect; explicit names -> use
                # exactly those (after intersecting with X.columns).
                _te_cols = list(_te_cols_cfg) if _te_cols_cfg else None
                if _te_cols is not None:
                    _hybrid_appended = set(self.hybrid_orth_features_ or [])
                    _mig_appended_set = set(self.mi_greedy_features_ or [])
                    _te_cols = [c for c in _te_cols if c in X.columns and c not in _hybrid_appended and c not in _mig_appended_set]
                _y_for_te = _y_np
                # TE works for both binary classification and regression as-
                # is (mean of {0,1} = P(y=1); mean of continuous = mean).
                # Cast bool / object to float to avoid type errors inside
                # the mean computation.
                _y_for_te = np.asarray(_y_for_te, dtype=np.float64).ravel()
                _X_before_te_cols = list(X.columns)
                # W6 follow-up: record this family's unified local-MI abs-MAD
                # floor kills into the FE rejection ledger (pure-record; the
                # kept set is unchanged so selection is byte-identical).
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
                _te_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _te_reject_sink(**_kw):
                    """Reject-sink callback for the k-fold target-encoding FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_te_step, **_kw)

                X_te, _te_appended, _te_recipes = kfold_target_encode_with_recipes(
                    fe_to_pandas(X), _y_for_te,
                    cat_cols=_te_cols,
                    n_folds=int(getattr(self, "fe_kfold_te_folds", 5)),
                    smoothing=float(getattr(self, "fe_kfold_te_smoothing", 10.0)),
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_te_reject_sink,
                    # Multi-stat target encoding: beyond the per-cell mean(y), also emit std / skew / kurt of y per
                    # category when requested. Helps when the category MODULATES a raw feature (heteroscedastic /
                    # varying-slope): +0.04..+0.09 OOS R^2 in those regimes (bench_multistat_cell_encoding). Default
                    # ("mean",) is byte-identical to the prior single-stat behaviour.
                    stats=tuple(getattr(self, "fe_kfold_te_stats", ("mean",)) or ("mean",)),
                )
                # Guard against silent overlap with prior stages: the
                # ``{col}__te`` suffix is dedicated to this stage so the
                # collision pre-condition would require a user-supplied
                # source column literally named ``{src}__te``. Drop any
                # accidental name collision rather than overwrite.
                _te_appended = [c for c in _te_appended if c not in _X_before_te_cols]
                if _te_appended:
                    X = fe_append_columns(X, fe_extract_columns(X_te, _te_appended))
                    self.kfold_te_features_ = list(_te_appended)
                    # Route through hybrid_orth_features_ so the end-of-fit
                    # remap routes by-name selected items into
                    # _engineered_recipes_ (Layer 23 routing path).
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_te_appended)
                    for _r in _te_recipes:
                        if _r.name in _te_appended:
                            _kfold_te_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit kfold_te: appended %d engineered " "column(s): %s",
                            len(_te_appended),
                            _te_appended[:8],
                        )
            except Exception as _te_exc:
                logger.warning(
                    "MRMR.fit kfold_te FE raised %s: %s; continuing " "without target-encoded columns.",
                    type(_te_exc).__name__,
                    _te_exc,
                )

    # GROUPED AGGREGATION OVER QUANTILE-BINNED NUMERIC CELLS. Appends leak-safe per-cell
    # mean/std/skew/kurt of numeric columns grouped by quantile-binned cells of other numerics. Runs in the
    # pre-FE region (before categorize_dataset) so the appended columns enter screening like any numeric, and
    # routes recipes through hybrid_orth_features_ so a selected binagg column lands in _engineered_recipes_.
    if _fe_family_on("fe_binned_numeric_agg_enable", False) and fe_polars_exceeds(X):
        warnings.warn(
            "MRMR: fe_binned_numeric_agg_enable=True but X is a large polars frame (> ~2 GiB); binned-agg is an OOF stat "
            "needing a full-frame decision and is skipped to avoid a whole-frame to_pandas copy.",
            UserWarning, stacklevel=3,
        )
    elif _fe_family_on("fe_binned_numeric_agg_enable", False):
        try:
            from .._binned_numeric_agg_fe import binned_numeric_agg_with_recipes
            _ba_y = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).ravel()
            _X_before_ba = list(X.columns)
            _bas_raw = getattr(self, "fe_binned_numeric_agg_stats", None)
            X_ba, _ba_appended, _ba_recipes = binned_numeric_agg_with_recipes(
                fe_to_pandas(X), _ba_y,
                stats=tuple(_bas_raw) if _bas_raw is not None else ("mean", "std", "skew", "kurt"),
                nbins_base=int(getattr(self, "fe_binned_numeric_agg_nbins", 10)),
                n_folds=int(getattr(self, "fe_kfold_te_folds", 5)),
                random_state=int(getattr(self, "random_seed", 0) or 0),
                max_pairs=int(getattr(self, "fe_binned_numeric_agg_max_pairs", 64)),
                redundancy_gate=bool(getattr(self, "fe_binned_numeric_agg_redundancy_gate", True)),
                min_cmi_gain=float(getattr(self, "fe_binned_numeric_agg_min_cmi_gain", 0.005)),
            )
            _ba_appended = [c for c in _ba_appended if c not in _X_before_ba]
            if _ba_appended:
                X = fe_append_columns(X, fe_extract_columns(X_ba, _ba_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ba_appended)
                for _r in _ba_recipes:
                    if _r.name in _ba_appended:
                        _binned_agg_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit binned_numeric_agg: appended %d engineered column(s): %s",
                        len(_ba_appended), _ba_appended[:8],
                    )
        except Exception as _ba_exc:
            logger.warning(
                "MRMR.fit binned_numeric_agg FE raised %s: %s; continuing without binned-agg columns.",
                type(_ba_exc).__name__, _ba_exc,
            )

    # 2026-05-31 Layer 34 — COUNT + FREQUENCY ENCODING + CAT x NUM
    # INTERACTION (target-mean residual). Three independent master switches;
    # each appends its own engineered columns AND emits one recipe per col.
    # Recipes route through ``hybrid_orth_features_`` so the end-of-fit
    # remap (Layer 23 pattern) routes them into ``_engineered_recipes_``.
    self.count_encoding_features_ = []
    self.frequency_encoding_features_ = []
    self.cat_num_interaction_features_ = []
    _count_enc_pre_recipes: dict = {}
    _freq_enc_pre_recipes: dict = {}
    _cat_num_pre_recipes: dict = {}
    if (
        _fe_family_on("fe_count_encoding_enable", False)
        or _fe_family_on("fe_frequency_encoding_enable", False)
        or _fe_family_on("fe_cat_num_interaction_enable", False)
    ):
        # Count / frequency / cat-num-residual encodings are OOF / full-cardinality stats (no closed-form subsample-replay),
        # so they need the full frame: gate the materialisation on size and skip a > ~2 GiB polars frame (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: Layer 34 FE (count/frequency/cat_num) enabled but X is a large polars frame (> ~2 GiB); these OOF/"
                "cardinality encodings need a full-frame decision and are skipped to avoid a whole-frame to_pandas copy.",
                UserWarning, stacklevel=3,
            )
        else:
            from .._count_freq_interaction_fe import (
                count_encode_with_recipes,
                frequency_encode_with_recipes,
                cat_num_interaction_with_recipes,
            )
            from .._target_encoding_fe import auto_detect_te_cols
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

            # W6 follow-up: shared sink for the count/freq/cat-num family's
            # unified local-MI abs-MAD floor kills (pure-record; selection
            # byte-identical).
            _l34_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _l34_reject_sink(**_kw):
                """Shared reject-sink for the count/frequency/cat-num-interaction FE family; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_l34_step, **_kw)

            _hybrid_appended_l34 = set(self.hybrid_orth_features_ or [])
            _mig_appended_l34 = set(self.mi_greedy_features_ or [])
            _te_appended_l34 = set(self.kfold_te_features_ or [])
            _engineered_seen_l34 = _hybrid_appended_l34 | _mig_appended_l34 | _te_appended_l34

            # ----- Count encoding ----------------------------------------
            if _fe_family_on("fe_count_encoding_enable", False):
                try:
                    _cnt_cfg = tuple(getattr(self, "fe_count_encoding_cols", ()) or ())
                    if _cnt_cfg:
                        _cnt_cols = [c for c in _cnt_cfg if c in X.columns and c not in _engineered_seen_l34]
                    else:
                        _cnt_cols = auto_detect_te_cols(
                            X, min_card=5, max_card=500,
                        )
                    _X_before_cnt_cols = list(X.columns)
                    _y_for_cnt = _y_np
                    X_c, _cnt_appended, _cnt_recipes = count_encode_with_recipes(
                        fe_to_pandas(X), cat_cols=_cnt_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_cnt,
                        reject_sink=_l34_reject_sink,
                    )
                    _cnt_appended = [c for c in _cnt_appended if c not in _X_before_cnt_cols]
                    if _cnt_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_c, _cnt_appended))
                        self.count_encoding_features_ = list(_cnt_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cnt_appended)
                        for _r in _cnt_recipes:
                            if _r.name in _cnt_appended:
                                _count_enc_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit count_encoding: appended %d " "engineered column(s): %s",
                                len(_cnt_appended),
                                _cnt_appended[:8],
                            )
                except Exception as _cnt_exc:
                    logger.warning(
                        "MRMR.fit count_encoding FE raised %s: %s; " "continuing without count-encoded columns.",
                        type(_cnt_exc).__name__,
                        _cnt_exc,
                    )

            # ----- Frequency encoding ------------------------------------
            if _fe_family_on("fe_frequency_encoding_enable", False):
                try:
                    _freq_cfg = tuple(getattr(self, "fe_frequency_encoding_cols", ()) or ())
                    if _freq_cfg:
                        _freq_cols = [c for c in _freq_cfg if c in X.columns and c not in _engineered_seen_l34]
                    else:
                        _freq_cols = auto_detect_te_cols(
                            X, min_card=5, max_card=500,
                        )
                    _X_before_freq_cols = list(X.columns)
                    _y_for_freq = _y_np
                    X_f, _freq_appended, _freq_recipes = frequency_encode_with_recipes(
                        fe_to_pandas(X), cat_cols=_freq_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_freq,
                        reject_sink=_l34_reject_sink,
                    )
                    _freq_appended = [c for c in _freq_appended if c not in _X_before_freq_cols]
                    if _freq_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_f, _freq_appended))
                        self.frequency_encoding_features_ = list(_freq_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_freq_appended)
                        for _r in _freq_recipes:
                            if _r.name in _freq_appended:
                                _freq_enc_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit frequency_encoding: appended %d " "engineered column(s): %s",
                                len(_freq_appended),
                                _freq_appended[:8],
                            )
                except Exception as _freq_exc:
                    logger.warning(
                        "MRMR.fit frequency_encoding FE raised %s: %s; " "continuing without frequency-encoded columns.",
                        type(_freq_exc).__name__,
                        _freq_exc,
                    )

            # ----- Cat x Num interaction (OOF residual) ------------------
            if _fe_family_on("fe_cat_num_interaction_enable", False):
                try:
                    _cn_cats = tuple(getattr(self, "fe_cat_num_interaction_cat_cols", ()) or ())
                    _cn_nums = tuple(getattr(self, "fe_cat_num_interaction_num_cols", ()) or ())
                    _cn_cats = tuple(c for c in _cn_cats if c in X.columns)
                    _cn_nums = tuple(c for c in _cn_nums if c in X.columns)
                    if _cn_cats and _cn_nums:
                        _y_for_cn = _y_np
                        _y_for_cn = np.asarray(_y_for_cn, dtype=np.float64).ravel()
                        _X_before_cn_cols = list(X.columns)
                        X_cn, _cn_appended, _cn_recipes = cat_num_interaction_with_recipes(
                            fe_to_pandas(X),
                            _y_for_cn,
                            cat_cols=_cn_cats,
                            num_cols=_cn_nums,
                            n_folds=int(getattr(self, "fe_cat_num_interaction_folds", 5)),
                            smoothing=float(getattr(self, "fe_cat_num_interaction_smoothing", 10.0)),
                            random_state=int(getattr(self, "random_seed", 0) or 0),
                            mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                            mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                            reject_sink=_l34_reject_sink,
                        )
                        _cn_appended = [c for c in _cn_appended if c not in _X_before_cn_cols]
                        if _cn_appended:
                            X = fe_append_columns(X, fe_extract_columns(X_cn, _cn_appended))
                            self.cat_num_interaction_features_ = list(_cn_appended)
                            self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cn_appended)
                            for _r in _cn_recipes:
                                if _r.name in _cn_appended:
                                    _cat_num_pre_recipes[_r.name] = _r
                            if verbose:
                                logger.info(
                                    "MRMR.fit cat_num_interaction: appended %d " "engineered column(s): %s",
                                    len(_cn_appended),
                                    _cn_appended[:8],
                                )
                except Exception as _cn_exc:
                    logger.warning(
                        "MRMR.fit cat_num_interaction FE raised %s: %s; " "continuing without cat x num residual columns.",
                        type(_cn_exc).__name__,
                        _cn_exc,
                    )

    # 2026-05-31 Layer 37 — MISSINGNESS-AWARE FE. Three independent master
    # switches (indicator / count / pattern); each appends its own engineered
    # columns AND emits one recipe per column. Recipes route through
    # ``hybrid_orth_features_`` so the end-of-fit remap (Layer 23 pattern)
    # routes them into ``_engineered_recipes_``.
    self.missingness_indicator_features_ = []
    self.missingness_count_features_ = []
    self.missingness_pattern_features_ = []
    _miss_ind_pre_recipes: dict = {}
    _miss_cnt_pre_recipes: dict = {}
    _miss_pat_pre_recipes: dict = {}
    # Unlike every other fe_*_enable family, missingness indicator/count/pattern are not a FE SEARCH step
    # consuming the fe_max_steps budget - each is a deterministic, explicit-opt-in-only static derivation
    # from the input's own NaN structure (no candidate scan, no round-trip cost the budget was meant to
    # bound). Deliberately checked directly (not via _fe_family_on, which requires fe_max_steps>0) so
    # ``fe_max_steps=0`` + an explicit fe_missingness_*_enable=True still emits the requested column(s) -
    # exactly the "disable the FE search but keep this one explicit static feature" contract callers rely on.
    if (
        bool(getattr(self, "fe_missingness_indicator_enable", False))
        or bool(getattr(self, "fe_missingness_count_enable", False))
        or bool(getattr(self, "fe_missingness_pattern_enable", False))
    ):
        # Missingness indicator/count/pattern read whole-column NaN structure (no closed-form subsample-replay), so they
        # need the full frame: gate the materialisation on size and skip a > ~2 GiB polars frame (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: Layer 37 FE (missingness indicator/count/pattern) enabled but X is a large polars frame (> ~2 GiB); "
                "the missingness encodings need a full-frame decision and are skipped to avoid a whole-frame to_pandas copy.",
                UserWarning, stacklevel=3,
            )
        else:
            from .._missingness_fe import (
                auto_detect_missing_cols,
                missing_indicator_with_recipes,
                missingness_count_with_recipes,
                missingness_pattern_with_recipes,
            )
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

            # Restore the fit-entry NaN positions on the snapshot columns before deriving missingness encodings. An earlier include_numeric /
            # binned_numeric_agg cat-FE stage GPU-categorizes and imputes X in place (when CUDA_PATH is set), which erases the very NaNs the
            # missingness-FE family encodes - is_missing__ would be all-zeros and missingness_pattern would collapse to a single pattern. The raw
            # NaNs are the user's input; MRMR's nan_strategy='separate_bin' scorer handles them downstream, so reinstating them here is correct, not a hack.
            if _fit_entry_nan_mask and isinstance(X, pd.DataFrame):
                for _mc, _mask in _fit_entry_nan_mask.items():
                    if _mc in X.columns and len(_mask) == len(X):
                        _col_now = X[_mc]
                        if not _col_now.isna().to_numpy().any():
                            _restored = _col_now.to_numpy().astype(np.float64, copy=True)
                            _restored[_mask] = np.nan
                            X[_mc] = _restored

            # W6 follow-up: missingness-indicator family's unified local-MI
            # abs-MAD floor kills (pure-record; selection byte-identical).
            _l37_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _l37_reject_sink(**_kw):
                """Shared reject-sink for the missingness-indicator FE family; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_l37_step, **_kw)

            _engineered_seen_l37 = (
                set(self.hybrid_orth_features_ or [])
                | set(self.mi_greedy_features_ or [])
                | set(getattr(self, "kfold_te_features_", []) or [])
                | set(getattr(self, "count_encoding_features_", []) or [])
                | set(getattr(self, "frequency_encoding_features_", []) or [])
                | set(getattr(self, "cat_num_interaction_features_", []) or [])
            )

            def _resolve_missing_cols(cfg):
                """Resolve the missingness-indicator family's candidate columns: explicit ``cfg`` when given, else auto-detect NaN-rate-in-[1%,99%] columns; always excludes columns already engineered by an earlier FE stage."""
                _cfg = tuple(cfg or ())
                if _cfg:
                    return [c for c in _cfg if c in X.columns and c not in _engineered_seen_l37]  # type: ignore[union-attr]
                # Auto-detect candidate cols with NaN rate in [1%, 99%].
                return [c for c in auto_detect_missing_cols(fe_to_pandas(X)) if c not in _engineered_seen_l37]

            # ----- Per-column indicator ------------------------------------
            if bool(getattr(self, "fe_missingness_indicator_enable", False)):
                try:
                    _ind_cols = _resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))
                    _X_before_ind_cols = list(X.columns)
                    _y_for_ind = _y_np
                    # Anchor the indicator's MI noise floor on the RAW input columns, not the engineered-polluted X: an earlier adaptive-Fourier stage appended high-(plug-in)-MI hijacker columns that would otherwise inflate the floor above a genuine MNAR indicator's MI and drop it (a >2%-missing source's signal lives in the NaN pattern the Fourier MI inflates).
                    _raw_floor_X = fe_to_pandas(X)[[c for c in _raw_input_cols_pre_fe if c in X.columns]] if _raw_input_cols_pre_fe else None
                    X_i, _ind_appended, _ind_recipes = missing_indicator_with_recipes(
                        fe_to_pandas(X), cols=_ind_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_ind,
                        raw_X=_raw_floor_X,
                        reject_sink=_l37_reject_sink,
                    )
                    _ind_appended = [c for c in _ind_appended if c not in _X_before_ind_cols]
                    if _ind_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_i, _ind_appended))
                        self.missingness_indicator_features_ = list(_ind_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ind_appended)
                        for _r in _ind_recipes:
                            if _r.name in _ind_appended:
                                _miss_ind_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_indicator: appended %d " "engineered column(s): %s",
                                len(_ind_appended),
                                _ind_appended[:8],
                            )
                except Exception as _ind_exc:
                    logger.warning(
                        "MRMR.fit missingness_indicator FE raised %s: %s; " "continuing without missingness indicator columns.",
                        type(_ind_exc).__name__,
                        _ind_exc,
                    )

            # ----- Per-row missingness count -------------------------------
            if bool(getattr(self, "fe_missingness_count_enable", False)):
                try:
                    _cnt_cols = _resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))
                    _X_before_mc_cols = list(X.columns)
                    X_c, _mc_appended, _mc_recipes = missingness_count_with_recipes(
                        fe_to_pandas(X), cols=_cnt_cols,
                    )
                    _mc_appended = [c for c in _mc_appended if c not in _X_before_mc_cols]
                    if _mc_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_c, _mc_appended))
                        self.missingness_count_features_ = list(_mc_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_mc_appended)
                        for _r in _mc_recipes:
                            if _r.name in _mc_appended:
                                _miss_cnt_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_count: appended %d " "engineered column(s): %s",
                                len(_mc_appended),
                                _mc_appended[:8],
                            )
                except Exception as _mc_exc:
                    logger.warning(
                        "MRMR.fit missingness_count FE raised %s: %s; " "continuing without missingness count column.",
                        type(_mc_exc).__name__,
                        _mc_exc,
                    )

            # ----- Per-row top-K pattern -----------------------------------
            if bool(getattr(self, "fe_missingness_pattern_enable", False)):
                try:
                    _pat_cols = _resolve_missing_cols(getattr(self, "fe_missingness_indicator_cols", ()))
                    _top_k = int(getattr(self, "fe_missingness_pattern_top_k", 5))
                    _X_before_pat_cols = list(X.columns)
                    X_p, _pat_appended, _pat_recipes = missingness_pattern_with_recipes(
                        fe_to_pandas(X), cols=_pat_cols, top_k=_top_k,
                    )
                    _pat_appended = [c for c in _pat_appended if c not in _X_before_pat_cols]
                    if _pat_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_p, _pat_appended))
                        self.missingness_pattern_features_ = list(_pat_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_pat_appended)
                        for _r in _pat_recipes:
                            if _r.name in _pat_appended:
                                _miss_pat_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_pattern: appended %d " "engineered column(s): %s",
                                len(_pat_appended),
                                _pat_appended[:8],
                            )
                except Exception as _pat_exc:
                    logger.warning(
                        "MRMR.fit missingness_pattern FE raised %s: %s; " "continuing without missingness pattern column.",
                        type(_pat_exc).__name__,
                        _pat_exc,
                    )

    # 2026-05-31 Layer 38 — CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF.
    # Four independent master switches (ratio / log_ratio / grouped_delta /
    # lagged_diff); each appends its engineered columns AND emits one recipe
    # per column. Routing piggybacks on hybrid_orth_features_ (same Layer 23
    # remap pattern used by Layers 33/34/37).
    self.pairwise_ratio_features_ = []
    self.pairwise_log_ratio_features_ = []
    self.grouped_delta_features_ = []
    self.lagged_diff_features_ = []
    self.grouped_agg_features_ = []
    self.composite_group_agg_features_ = []
    self.grouped_quantile_features_ = []
    self.cat_pair_features_ = []
    self.cat_triple_features_ = []
    self.numeric_decompose_features_ = []
    self.temporal_agg_features_ = []
    self.modular_features_ = []
    self.pairwise_modular_features_ = []
    self.integer_lattice_features_ = []
    self.row_argmax_features_ = []
    self.conditional_gate_features_ = []
    # RAW SOURCE OPERANDS of the selected gate_mask / row_argmax features (their recipe src_names).
    # The FE pair step re-classifies these from synergy-bootstrap to REGULARLY-selected operands so
    # the elementary pair over a gate's raw sources competes on the LENIENT prevalence bar instead of
    # being demoted to the stricter synergy bar (a high-MI gate built FROM a raw col evicts that col
    # from selected_vars, so its clean elementary pair would otherwise be suppressed). 2026-06-13.
    self._gate_raw_operands_ = set()
    # Per-gate-column -> set of its RAW source variables (recipe ``src_names``). The FE step uses this to
    # resolve the raw-variable coverage of a gate-operand COMPOSITE (whose gate operand buries its raw
    # vars inside the column name) so it can drop a composite whose entire raw coverage is already provided
    # by clean non-gate engineered survivors (CASE1) while keeping one that adds genuinely new (c,d)
    # coverage no clean form expresses (CASE2). Empty when no gate fired. 2026-06-13.
    self._gate_col_src_vars_ = {}
    self.group_distance_features_ = []
    _cat_pair_pre_recipes: dict = {}
    _cat_triple_pre_recipes: dict = {}
    _numeric_decompose_pre_recipes: dict = {}
    _temporal_agg_pre_recipes: dict = {}
    _modular_pre_recipes: dict = {}
    _pairwise_modular_pre_recipes: dict = {}
    _integer_lattice_pre_recipes: dict = {}
    _row_argmax_pre_recipes: dict = {}
    _conditional_gate_pre_recipes: dict = {}
    _group_distance_pre_recipes: dict = {}
    _rare_category_pre_recipes: dict = {}
    _conditional_residual_pre_recipes: dict = {}
    _conditional_dispersion_pre_recipes: dict = {}
    _conditional_quantile_rank_pre_recipes: dict = {}
    _ordinal_pattern_pre_recipes: dict = {}
    _random_fourier_pre_recipes: dict = {}
    _sir_direction_pre_recipes: dict = {}
    _lof_pre_recipes: dict = {}
    _mahalanobis_density_pre_recipes: dict = {}
    _wavelet_pre_recipes: dict = {}
    _rankgauss_pre_recipes: dict = {}
    _ratio_pre_recipes: dict = {}
    _log_ratio_pre_recipes: dict = {}
    _grouped_delta_pre_recipes: dict = {}
    _lagged_diff_pre_recipes: dict = {}
    _grouped_agg_pre_recipes: dict = {}
    _composite_group_agg_pre_recipes: dict = {}
    _grouped_quantile_pre_recipes: dict = {}
    if (
        _fe_family_on("fe_pairwise_ratio_enable", False)
        or _fe_family_on("fe_pairwise_log_ratio_enable", False)
        or _fe_family_on("fe_grouped_delta_enable", False)
        or _fe_family_on("fe_lagged_diff_enable", False)
    ):
        # grouped_delta / lagged_diff are cross-row (group / time ordered) and ratio / log-ratio rank their mi_gate on the
        # full frame, none wired for closed-form subsample-replay - so this block needs the full frame: gate the materialisation
        # on size and skip a > ~2 GiB polars frame (CLAUDE.md eager rule).
        if fe_polars_exceeds(X):
            warnings.warn(
                "MRMR: Layer 38 FE (ratio/log-ratio/grouped-delta/lagged-diff) enabled but X is a large polars frame "
                "(> ~2 GiB); these families need a full-frame decision and are skipped to avoid a whole-frame to_pandas copy.",
                UserWarning, stacklevel=3,
            )
        else:
            from .._ratio_delta_fe import (
                pairwise_ratio_with_recipes,
                pairwise_log_ratio_with_recipes,
                grouped_delta_with_recipes,
                lagged_diff_with_recipes,
            )

            _l38_mi_gate = bool(getattr(self, "fe_local_mi_gate", False))
            _l38_mi_gate_top_k = int(getattr(self, "fe_local_mi_gate_top_k", 20))
            _y_for_l38 = _y_np
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

            # W6 follow-up: shared sink for the ratio/log-ratio/grouped-delta/
            # lagged-diff family's unified local-MI abs-MAD floor kills
            # (pure-record; selection byte-identical).
            _l38_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _l38_reject_sink(**_kw):
                """Shared reject-sink for the ratio/log-ratio/grouped-delta/lagged-diff FE family; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                _record_fe_rejection(self, step=_l38_step, **_kw)

            # ----- Pairwise ratio --------------------------------------------
            if _fe_family_on("fe_pairwise_ratio_enable", False):
                try:
                    _ratio_cols = tuple(getattr(self, "fe_pairwise_ratio_cols", ()) or ())
                    _ratio_cols = tuple(c for c in _ratio_cols if c in X.columns)
                    _eps = float(getattr(self, "fe_pairwise_ratio_eps", 1e-9))
                    _X_before_r_cols = list(X.columns)
                    X_r, _r_appended, _r_recipes = pairwise_ratio_with_recipes(
                        fe_to_pandas(X), cols=_ratio_cols, eps=_eps,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _r_appended = [c for c in _r_appended if c not in _X_before_r_cols]
                    if _r_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_r, _r_appended))
                        self.pairwise_ratio_features_ = list(_r_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_r_appended)
                        for _r in _r_recipes:
                            if _r.name in _r_appended:
                                _ratio_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit pairwise_ratio: appended %d " "engineered column(s): %s",
                                len(_r_appended),
                                _r_appended[:8],
                            )
                except Exception as _r_exc:
                    logger.warning(
                        "MRMR.fit pairwise_ratio FE raised %s: %s; " "continuing without ratio columns.",
                        type(_r_exc).__name__,
                        _r_exc,
                    )

            # ----- Pairwise log-ratio ----------------------------------------
            if _fe_family_on("fe_pairwise_log_ratio_enable", False):
                try:
                    _lr_cols = tuple(getattr(self, "fe_pairwise_log_ratio_cols", ()) or ())
                    _lr_cols = tuple(c for c in _lr_cols if c in X.columns)
                    _eps_lr = float(getattr(self, "fe_pairwise_ratio_eps", 1e-9))
                    _X_before_lr_cols = list(X.columns)
                    X_lr, _lr_appended, _lr_recipes = pairwise_log_ratio_with_recipes(
                        fe_to_pandas(X), cols=_lr_cols, eps=_eps_lr,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _lr_appended = [c for c in _lr_appended if c not in _X_before_lr_cols]
                    if _lr_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_lr, _lr_appended))
                        self.pairwise_log_ratio_features_ = list(_lr_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_lr_appended)
                        for _r in _lr_recipes:
                            if _r.name in _lr_appended:
                                _log_ratio_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit pairwise_log_ratio: appended %d " "engineered column(s): %s",
                                len(_lr_appended),
                                _lr_appended[:8],
                            )
                except Exception as _lr_exc:
                    logger.warning(
                        "MRMR.fit pairwise_log_ratio FE raised %s: %s; " "continuing without log-ratio columns.",
                        type(_lr_exc).__name__,
                        _lr_exc,
                    )

            # ----- Grouped delta ---------------------------------------------
            if _fe_family_on("fe_grouped_delta_enable", False):
                try:
                    _gd_group = getattr(self, "fe_grouped_delta_group_col", None)
                    _gd_nums = tuple(getattr(self, "fe_grouped_delta_num_cols", ()) or ())
                    _gd_nums = tuple(c for c in _gd_nums if c in X.columns)
                    _X_before_gd_cols = list(X.columns)
                    X_gd, _gd_appended, _gd_recipes = grouped_delta_with_recipes(
                        fe_to_pandas(X), group_col=_gd_group, num_cols=_gd_nums,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _gd_appended = [c for c in _gd_appended if c not in _X_before_gd_cols]
                    if _gd_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_gd, _gd_appended))
                        self.grouped_delta_features_ = list(_gd_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_gd_appended)
                        for _r in _gd_recipes:
                            if _r.name in _gd_appended:
                                _grouped_delta_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit grouped_delta: appended %d " "engineered column(s): %s",
                                len(_gd_appended),
                                _gd_appended[:8],
                            )
                except Exception as _gd_exc:
                    logger.warning(
                        "MRMR.fit grouped_delta FE raised %s: %s; " "continuing without grouped-delta columns.",
                        type(_gd_exc).__name__,
                        _gd_exc,
                    )

            # ----- Lagged diff -----------------------------------------------
            if _fe_family_on("fe_lagged_diff_enable", False):
                try:
                    _ld_time = getattr(self, "fe_lagged_diff_time_col", None)
                    _ld_vals = tuple(getattr(self, "fe_lagged_diff_value_cols", ()) or ())
                    _ld_vals = tuple(c for c in _ld_vals if c in X.columns)
                    _ld_periods = tuple(getattr(self, "fe_lagged_diff_periods", (1, 2)) or (1, 2))
                    _X_before_ld_cols = list(X.columns)
                    X_ld, _ld_appended, _ld_recipes = lagged_diff_with_recipes(
                        fe_to_pandas(X), time_col=_ld_time, value_cols=_ld_vals,
                        periods=_ld_periods,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38, reject_sink=_l38_reject_sink,
                    )
                    _ld_appended = [c for c in _ld_appended if c not in _X_before_ld_cols]
                    if _ld_appended:
                        X = fe_append_columns(X, fe_extract_columns(X_ld, _ld_appended))
                        self.lagged_diff_features_ = list(_ld_appended)
                        self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ld_appended)
                        for _r in _ld_recipes:
                            if _r.name in _ld_appended:
                                _lagged_diff_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit lagged_diff: appended %d " "engineered column(s): %s",
                                len(_ld_appended),
                                _ld_appended[:8],
                            )
                except Exception as _ld_exc:
                    logger.warning(
                        "MRMR.fit lagged_diff FE raised %s: %s; " "continuing without lagged-diff columns.",
                        type(_ld_exc).__name__,
                        _ld_exc,
                    )

    # Layer 87: grouped multi-stat aggregator with CMI gate.
    # NVIDIA cuDF Kaggle-Grandmaster technique #1. Per-group statistics of a
    # continuous column broadcast to rows + z-within / ratio residuals, each
    # CMI-gated against the raw support and uplift-gated against the source
    # num_col marginal MI. Routing piggybacks on hybrid_orth_features_ (same
    # Layer 23 remap as Layers 33/34/37/38).
    if _fe_family_on("fe_grouped_agg_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 87 grouped_agg FE enabled but X is not a pandas "
                "DataFrame; the aggregates are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._grouped_agg_fe import hybrid_grouped_agg_fe

                # CMI gate needs a class-typed target; bin continuous y the
                # same way the Layer 60 CMI-greedy stage does.
                _y_for_ga = _y_np
                if _y_for_ga.dtype.kind in "fc":
                    _n_unique_ga = int(np.unique(_y_for_ga).size)
                    if _n_unique_ga <= 32:
                        _y_for_ga = _y_for_ga.astype(np.int64)
                    else:
                        try:
                            _y_for_ga = pd.qcut(
                                _y_for_ga, q=10, labels=False, duplicates="drop",
                            ).astype(np.int64)
                        except Exception as exc:
                            logger.debug("mrmr: y densification failed for the grouped-aggregation FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                            _y_for_ga = _y_for_ga.astype(np.int64)

                _ga_groups = tuple(getattr(self, "fe_grouped_agg_group_cols", ()) or ())
                _ga_groups = [c for c in _ga_groups if c in X.columns] or None  # type: ignore[assignment]
                _ga_nums = tuple(getattr(self, "fe_grouped_agg_num_cols", ()) or ())
                _ga_nums = [c for c in _ga_nums if c in X.columns] or None  # type: ignore[assignment]
                _ga_stats_raw = getattr(self, "fe_grouped_agg_stats", None)
                _ga_stats = tuple(_ga_stats_raw) if _ga_stats_raw is not None else ("mean", "std", "min", "max", "nunique", "skew", "median")
                _ga_top_k = int(getattr(self, "fe_grouped_agg_top_k", 10))
                _X_before_ga_cols = list(X.columns)
                X_ga, _ga_appended, _ga_recipes, _ga_scores = hybrid_grouped_agg_fe(
                    X, _y_for_ga,
                    group_cols=_ga_groups, num_cols=_ga_nums,
                    stats=_ga_stats, top_k=_ga_top_k,
                )
                _ga_appended = [c for c in _ga_appended if c not in _X_before_ga_cols]
                if _ga_appended:
                    X = X_ga
                    self.grouped_agg_features_ = list(_ga_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ga_appended)
                    for _r in _ga_recipes:
                        if _r.name in _ga_appended:
                            _grouped_agg_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit grouped_agg: appended %d engineered " "column(s): %s",
                            len(_ga_appended),
                            _ga_appended[:8],
                        )
            except Exception as _ga_exc:
                logger.warning(
                    "MRMR.fit grouped_agg FE raised %s: %s; continuing " "without grouped-aggregate columns.",
                    type(_ga_exc).__name__,
                    _ga_exc,
                )

    # Layer 93: COMPOSITE (multi-column) group-key aggregates.
    # Multi-col extension of Layer 87: each composite key is factorized into
    # one integer-coded group and run through the same per-group stat / z /
    # ratio machinery; survivors are CMI-gated against the raw support and
    # uplift-gated against the source num_col marginal MI. Composite keys whose
    # distinct-cell count exceeds 0.5*n are refused (Layer 29 guard). Routing
    # piggybacks on hybrid_orth_features_ (same Layer 23 remap as 33/.../87).
    if _fe_family_on("fe_composite_group_agg_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 93 composite_group_agg FE enabled but X is not a "
                "pandas DataFrame; the aggregates are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._composite_group_agg_fe import hybrid_composite_group_agg_fe

                _y_for_cga = _y_np
                if _y_for_cga.dtype.kind in "fc":
                    _n_unique_cga = int(np.unique(_y_for_cga).size)
                    if _n_unique_cga <= 32:
                        _y_for_cga = _y_for_cga.astype(np.int64)
                    else:
                        try:
                            _y_for_cga = pd.qcut(
                                _y_for_cga, q=10, labels=False, duplicates="drop",
                            ).astype(np.int64)
                        except Exception as exc:
                            logger.debug("mrmr: y densification failed for the composite-group-aggregation FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                            _y_for_cga = _y_for_cga.astype(np.int64)

                # key_sets: each entry is a tuple of >= 2 group cols. Empty =>
                # auto-detect r-combinations of detected group columns.
                _cga_key_sets_raw = tuple(getattr(self, "fe_composite_group_agg_key_sets", ()) or ())
                _cga_key_sets = [tuple(c for c in gset if c in X.columns) for gset in _cga_key_sets_raw]
                _cga_key_sets = [g for g in _cga_key_sets if len(g) >= 2] or None  # type: ignore[assignment]
                _cga_nums = tuple(getattr(self, "fe_composite_group_agg_num_cols", ()) or ())
                _cga_nums = [c for c in _cga_nums if c in X.columns] or None  # type: ignore[assignment]
                _cga_stats_raw = getattr(self, "fe_composite_group_agg_stats", None)
                _cga_stats = tuple(_cga_stats_raw) if _cga_stats_raw is not None else ("mean", "std", "count")
                _cga_max_arity = int(getattr(self, "fe_composite_group_agg_max_arity", 2))
                _cga_top_k = int(getattr(self, "fe_composite_group_agg_top_k", 10))
                _X_before_cga_cols = list(X.columns)
                X_cga, _cga_appended, _cga_recipes, _cga_scores = (
                    hybrid_composite_group_agg_fe(
                        X, _y_for_cga,
                        group_col_sets=_cga_key_sets, num_cols=_cga_nums,
                        stats=_cga_stats, max_arity=_cga_max_arity,
                        top_k=_cga_top_k,
                    )
                )
                _cga_appended = [c for c in _cga_appended if c not in _X_before_cga_cols]
                if _cga_appended:
                    X = X_cga
                    self.composite_group_agg_features_ = list(_cga_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cga_appended)
                    for _r in _cga_recipes:
                        if _r.name in _cga_appended:
                            _composite_group_agg_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit composite_group_agg: appended %d " "engineered column(s): %s",
                            len(_cga_appended),
                            _cga_appended[:8],
                        )
            except Exception as _cga_exc:
                logger.warning(
                    "MRMR.fit composite_group_agg FE raised %s: %s; continuing " "without composite-aggregate columns.",
                    type(_cga_exc).__name__,
                    _cga_exc,
                )

    # Layer 88: per-group histogram + quantile FE with
    # target-aware edges. NVIDIA cuDF Kaggle-Grandmaster technique #2.
    # Percentile-rank-within-group + per-group IQR / p90-p10 spread, optionally
    # the OOF-fit target-aware supervised bin index; each survivor MI-gated
    # against the source num_col marginal MI. Routing piggybacks on
    # hybrid_orth_features_ (same Layer 23 remap as Layers 33/34/37/38/87).
    if _fe_family_on("fe_grouped_quantile_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 88 grouped_quantile FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._grouped_quantile_fe import hybrid_grouped_quantile_fe

                _y_for_gq = _y_np
                # Scope auto-detection to the RAW pre-FE columns: by this point X
                # is already augmented with engineered intermediates from prior FE
                # stages, and a grouped_quantile recipe built on an engineered group
                # / num source cannot be replayed at transform() (the engineered
                # parent is regenerated independently, not present in the apply X)
                # -> KeyError. Mirrors the cat_pair / cat_triple guard.
                _gq_groups = tuple(getattr(self, "fe_grouped_quantile_group_cols", ()) or ())
                _gq_groups = [c for c in _gq_groups if c in X.columns] or None  # type: ignore[assignment]
                _gq_nums = tuple(getattr(self, "fe_grouped_quantile_num_cols", ()) or ())
                _gq_nums = [c for c in _gq_nums if c in X.columns] or None  # type: ignore[assignment]
                _gq_raw = set(_raw_input_cols_pre_fe)
                if _gq_groups is None or _gq_nums is None:
                    from .._grouped_quantile_fe import (
                        _auto_detect_group_cols as _gq_detect_groups,
                        _auto_detect_num_cols as _gq_detect_nums,
                    )
                    _gq_raw_view = X[[c for c in X.columns if c in _gq_raw]]
                    if _gq_groups is None:
                        _gq_groups = _gq_detect_groups(_gq_raw_view) or None
                    if _gq_nums is None:
                        _gq_det_groups = _gq_groups or []
                        _gq_nums = _gq_detect_nums(_gq_raw_view, _gq_det_groups) or None
                _gq_quantiles_raw = getattr(self, "fe_grouped_quantile_quantiles", None)
                _gq_quantiles = tuple(_gq_quantiles_raw) if _gq_quantiles_raw is not None else (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
                _gq_target_aware = bool(getattr(self, "fe_grouped_quantile_target_aware", False))
                _gq_n_bins = int(getattr(self, "fe_grouped_quantile_n_bins", 5))
                _gq_top_k = int(getattr(self, "fe_grouped_quantile_top_k", 8))
                _X_before_gq_cols = list(X.columns)
                X_gq, _gq_appended, _gq_recipes, _gq_scores = hybrid_grouped_quantile_fe(
                    X, _y_for_gq,
                    group_cols=_gq_groups, num_cols=_gq_nums,
                    quantiles=_gq_quantiles, target_aware=_gq_target_aware,
                    n_bins=_gq_n_bins, top_k=_gq_top_k,
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                )
                _gq_appended = [c for c in _gq_appended if c not in _X_before_gq_cols]
                if _gq_appended:
                    X = X_gq
                    self.grouped_quantile_features_ = list(_gq_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_gq_appended)
                    for _r in _gq_recipes:
                        if _r.name in _gq_appended:
                            _grouped_quantile_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit grouped_quantile: appended %d engineered " "column(s): %s",
                            len(_gq_appended),
                            _gq_appended[:8],
                        )
            except Exception as _gq_exc:
                logger.warning(
                    "MRMR.fit grouped_quantile FE raised %s: %s; continuing " "without grouped-quantile columns.",
                    type(_gq_exc).__name__,
                    _gq_exc,
                )

    # Layer 89: cat x cat synergy cross with II pre-filter.
    if _fe_family_on("fe_cat_pair_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 89 cat_pair FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._cat_pair_fe import hybrid_cat_pair_fe

                _y_for_cp = _y_np
                _cp_cols = tuple(getattr(self, "fe_cat_pair_cat_cols", ()) or ())
                _cp_cols = [c for c in _cp_cols if c in X.columns] or None  # type: ignore[assignment]
                # When auto-detecting cat-pair members, restrict candidates to
                # the RAW input columns. By this point X carries engineered
                # intermediates (count/frequency-encoded integer columns from
                # the L34 stage) whose low cardinality would otherwise let
                # auto_detect_cat_pair_cols promote them as pair members. A
                # cross built on an engineered column cannot be replayed at
                # transform time (the recipe looks the column up directly in
                # X_test, where only raw inputs are guaranteed present) and
                # raises KeyError. Crossing raw categoricals only keeps the
                # recipe a pure function of X.
                if _cp_cols is None:
                    _cp_cols = [c for c in _raw_input_cols_pre_fe if c in X.columns] or None
                _cp_min_ii = float(getattr(self, "fe_cat_pair_min_interaction_info", 0.001))
                _cp_top_k = int(getattr(self, "fe_cat_pair_top_k", 5))
                _X_before_cp_cols = list(X.columns)
                X_cp, _cp_appended, _cp_recipes, _cp_scores = hybrid_cat_pair_fe(
                    X, _y_for_cp,
                    cat_cols=_cp_cols,
                    min_interaction_info=_cp_min_ii,
                    top_k=_cp_top_k,
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                )
                _cp_appended = [c for c in _cp_appended if c not in _X_before_cp_cols]
                if _cp_appended:
                    X = X_cp
                    self.cat_pair_features_ = list(_cp_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cp_appended)
                    for _r in _cp_recipes:
                        if _r.name in _cp_appended:
                            _cat_pair_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit cat_pair: appended %d engineered " "column(s): %s",
                            len(_cp_appended),
                            _cp_appended[:8],
                        )
            except Exception as _cp_exc:
                logger.warning(
                    "MRMR.fit cat_pair FE raised %s: %s; continuing without " "cat-pair-cross columns.",
                    type(_cp_exc).__name__,
                    _cp_exc,
                )

    # Layer 94: cat x cat x cat TRIPLE synergy cross via beam
    # search over three-way interaction information (co-information).
    if _fe_family_on("fe_cat_triple_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 94 cat_triple FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._cat_triple_fe import hybrid_cat_triple_fe

                _y_for_ct = _y_np
                _ct_cols = tuple(getattr(self, "fe_cat_triple_cat_cols", ()) or ())
                _ct_cols = [c for c in _ct_cols if c in X.columns] or None  # type: ignore[assignment]
                # Same raw-column restriction as the cat_pair stage: auto-
                # detected triple members must be raw inputs so the cross
                # recipe replays as a pure function of X (an engineered
                # intermediate would raise KeyError at transform time).
                if _ct_cols is None:
                    _ct_cols = [c for c in _raw_input_cols_pre_fe if c in X.columns] or None
                _ct_min_ii = float(getattr(self, "fe_cat_triple_min_interaction_info", 0.001))
                _ct_beam = int(getattr(self, "fe_cat_triple_beam_width", 3))
                _ct_top_k = int(getattr(self, "fe_cat_triple_top_k", 3))
                _X_before_ct_cols = list(X.columns)
                X_ct, _ct_appended, _ct_recipes, _ct_scores = hybrid_cat_triple_fe(
                    X, _y_for_ct,
                    cat_cols=_ct_cols,
                    min_interaction_info=_ct_min_ii,
                    top_k=_ct_top_k,
                    beam_width=_ct_beam,
                    top_k_pairs=_ct_beam,
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                )
                _ct_appended = [c for c in _ct_appended if c not in _X_before_ct_cols]
                if _ct_appended:
                    X = X_ct
                    self.cat_triple_features_ = list(_ct_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ct_appended)
                    for _r in _ct_recipes:
                        if _r.name in _ct_appended:
                            _cat_triple_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit cat_triple: appended %d engineered " "column(s): %s",
                            len(_ct_appended),
                            _ct_appended[:8],
                        )
            except Exception as _ct_exc:
                logger.warning(
                    "MRMR.fit cat_triple FE raised %s: %s; continuing without " "cat-triple-cross columns.",
                    type(_ct_exc).__name__,
                    _ct_exc,
                )

    # Layer 90: numeric decomposition (multi-precision rounding +
    # decimal-digit extraction) with a bootstrap-stable MI gate.
    if _fe_family_on("fe_numeric_decompose_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 90 numeric_decompose FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._numeric_decompose_fe import (
                    hybrid_numeric_decompose_fe_with_recipes,
                )

                _y_for_nd = _y_np
                _nd_precisions = tuple(getattr(self, "fe_numeric_decompose_precisions", (1, 0.1, 0.01, 0.001)))
                _nd_digits = tuple(getattr(self, "fe_numeric_decompose_digits", (0, 1, 2)))
                _nd_n_boot = int(getattr(self, "fe_numeric_decompose_n_boot", 10))
                _nd_top_k = int(getattr(self, "fe_numeric_decompose_top_k", 5))
                _X_before_nd_cols = list(X.columns)
                X_nd, _nd_appended, _nd_recipes, _nd_scores = hybrid_numeric_decompose_fe_with_recipes(
                    X,
                    _y_for_nd,
                    cols=None,
                    precisions=_nd_precisions,
                    digit_positions=_nd_digits,
                    top_k=_nd_top_k,
                    n_boot=_nd_n_boot,
                    seed=int(getattr(self, "random_seed", 0) or 0),
                )
                _nd_appended = [c for c in _nd_appended if c not in _X_before_nd_cols]
                if _nd_appended:
                    X = X_nd
                    self.numeric_decompose_features_ = list(_nd_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_nd_appended)
                    for _r in _nd_recipes:
                        if _r.name in _nd_appended:
                            _numeric_decompose_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit numeric_decompose: appended %d engineered " "column(s): %s",
                            len(_nd_appended),
                            _nd_appended[:8],
                        )
            except Exception as _nd_exc:
                logger.warning(
                    "MRMR.fit numeric_decompose FE raised %s: %s; continuing " "without numeric-decomposition columns.",
                    type(_nd_exc).__name__,
                    _nd_exc,
                )

    # Layer 95 PART A: periodic / modular decomposition. For each
    # (col, period) emit x mod period plus its sin/cos phase encoding; each
    # candidate gated by Layer 62 bootstrap-stable MI (the gate doubles as
    # auto-period detection). Routing piggybacks on hybrid_orth_features_.
    if _fe_family_on("fe_modular_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 95 modular FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._periodic_fe import hybrid_modular_fe_with_recipes

                _y_for_md = _y_np
                _md_periods = tuple(getattr(self, "fe_modular_periods", (7, 12, 24, 30, 365)) or (7, 12, 24, 30, 365))
                _md_top_k = int(getattr(self, "fe_modular_top_k", 6))
                _X_before_md_cols = list(X.columns)
                X_md, _md_appended, _md_recipes, _md_scores = hybrid_modular_fe_with_recipes(
                    X,
                    _y_for_md,
                    cols=None,
                    periods=_md_periods,
                    top_k=_md_top_k,
                    seed=int(getattr(self, "random_seed", 0) or 0),
                )
                _md_appended = [c for c in _md_appended if c not in _X_before_md_cols]
                if _md_appended:
                    X = X_md
                    self.modular_features_ = list(_md_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_md_appended)
                    for _r in _md_recipes:
                        if _r.name in _md_appended:
                            _modular_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit modular: appended %d engineered " "column(s): %s",
                            len(_md_appended),
                            _md_appended[:8],
                        )
            except Exception as _md_exc:
                logger.warning(
                    "MRMR.fit modular FE raised %s: %s; continuing without " "modular columns.",
                    type(_md_exc).__name__,
                    _md_exc,
                )

    # Pairwise / n-way modular FE: detect a target that is an integer modulus of a
    # combination of integer columns - (a+b) mod m, (a*b) mod m, n-way parity, or a
    # single column's hidden non-calendar period - which smooth bases cannot fit.
    # Cheap-first / escalate + permutation-null gate; budget-guarded on wide frames.
    # The four discrete-structural families (pairwise-modular / row-argmax / conditional-gate / binned-agg)
    # used to fire INDEPENDENTLY of fe_max_steps, carrying a small-n reliability floor to keep their
    # high-cardinality composites from admitting noise at fe_max_steps=0. They now honour the same
    # unconditional budget rule as every other family (see _fe_family_on), which subsumes that floor: with
    # FE enabled the normal pipeline competes the composites down, and with FE off they simply never build.
    _discrete_fe_master = _fe_family_on("fe_discrete_structural_operators_enable", True) and _fe_budget_ok()
    # OPERATOR SKIP-GATE (2026-06-18, perf). The four discrete-structural operators (pairwise-modular /
    # row-argmax / conditional-gate / binned-agg) hunt for NONLINEAR/regime structure via MI-kernel scans
    # over many candidate combos - ~58% of an additive-regression fit (cProfile: cheap_conditional_gate_scan
    # 7.2s + binned_numeric_agg 4s of a 19s fit). On an additive-LINEAR regression target there is no such
    # structure to find, so a single cheap linear fit on the raws is a necessary-condition gate: if the raws
    # already explain y (R^2>=0.92), skip the scans. Classification keeps them (R^2 N/A there -> the gate
    # returns False), and any genuine regime/modular/interaction target leaves a large linear residual
    # (low R^2) -> the operators still fire. One ~0.1s linear fit vs ~11s of scans.
    #
    # SCOPE: AUTOMATIC PATH ONLY (fe_max_steps>0) -- moot at fe_max_steps==0 anyway, since
    # ``_discrete_fe_master`` is already False by then via ``_fe_family_on``'s unconditional budget gate (see
    # above; the discrete-structural fe_max_steps=0 carve-out this skip-gate originally scoped itself against
    # was retired). This skip-gate is a perf optimisation for the automatic FE pipeline only: when
    # the operators run alongside the basis/escalation passes, skip their scans if a cheap linear fit already
    # explains y (R^2>=0.92) -- but that in-sample score is not proof of NO operator structure (e.g.
    # y=1[argmax(a,b,c)==0]: raw-only in-sample logistic AUC ~0.98 yet argmax__a__b__c is a clean, selectable
    # composite), so the gate only fires within the automatic budget, never as a blanket "skip if explainable".
    if _discrete_fe_master and fe_max_steps > 0:
        try:
            from .._fe_linear_explainability import raws_linearly_explain_y

            if raws_linearly_explain_y(X, y, seed=int(getattr(self, "random_seed", 0) or 0)):
                _discrete_fe_master = False
        except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
            logger.debug("raws_linearly_explain_y gate failed (%s: %s) -- keeping the operators (the safe/correct path)", type(e).__name__, e)

    # Shared class-MI target binning for the four discrete-structural FE operators (pairwise-modular / integer-lattice / row-argmax / conditional-gate).
    # All four gate candidates on the SAME 1D y binned with the SAME quantization_nbins via bin_y_for_class_mi; compute the applicability flag + binned
    # labels ONCE here and reuse, rather than re-quantile-binning the identical target inside each block. _y_np is fixed for the whole fit (never rebound).
    _y_class_mi_applicable = False
    _y_class_mi_binned = None
    if (
        _discrete_fe_master
        and isinstance(X, pd.DataFrame)
        and (
            _fe_family_on("fe_pairwise_modular_enable", False)
            or _fe_family_on("fe_integer_lattice_enable", False)
            or _fe_family_on("fe_row_argmax_enable", False)
            or _fe_family_on("fe_conditional_gate_enable", False)
        )
    ):
        from .._fe_accuracy_gate import bin_y_for_class_mi as _bin_y_class_mi, class_mi_fe_applicable as _class_mi_applicable

        _y_class_mi_applicable = _class_mi_applicable(_y_np)
        if _y_class_mi_applicable:
            _y_class_mi_binned = _bin_y_class_mi(_y_np, nbins=int(getattr(self, "quantization_nbins", 10)))

    if _discrete_fe_master and _fe_family_on("fe_pairwise_modular_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: pairwise-modular FE enabled but X is not a pandas DataFrame; " "the features are skipped. Convert via X.to_pandas() before fit().",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from .._pairwise_modular_fe import (
                    apply_pairwise_modular,
                    hybrid_pairwise_modular_fe_with_recipes,
                )

                # The detector's relevance floor is class-MI. 1D classification y feeds directly; a CONTINUOUS 1D y is quantile-binned once
                # (bin_y_for_class_mi, nbins=quantization_nbins) so the kernel sees a discrete target - the prior int64 cast collapsed continuous y
                # to ~n bogus classes. Only a 2D (multilabel/multi-target) y stays skipped (binning a label matrix is out of scope). Reuses the
                # shared _y_class_mi_* computed once above (identical y + nbins across all four discrete-structural operators).
                _pm_appended, _pm_recipes = ([], [])
                if _y_class_mi_applicable:
                    _y_pm_binned = _y_class_mi_binned
                    # Restrict operands to raw input columns: combining on already-engineered columns yields nested recipes
                    # whose engineered source is not resolvable at replay time (transform() emits NaN and drops the feature).
                    _pm_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]
                    _pm_appended, _pm_recipes = hybrid_pairwise_modular_fe_with_recipes(
                        X, _y_pm_binned,  # type: ignore[arg-type]
                        cols=_pm_raw_cols,
                        top_k=int(getattr(self, "fe_pairwise_modular_top_k", 4)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                        max_int_cols=int(getattr(self, "fe_pairwise_modular_max_int_cols", 30)),
                        max_triple_cols=int(getattr(self, "fe_pairwise_modular_max_triple_cols", 20)),
                    )
                _pm_appended = [c for c in _pm_appended if c not in X.columns]
                if _pm_appended:
                    _pm_new = {
                        _r.name: apply_pairwise_modular(
                            X, _r.extra["op"], _r.src_names, _r.extra["modulus"],
                        )
                        for _r in _pm_recipes if _r.name in _pm_appended
                    }
                    X = pd.concat(
                        [X, pd.DataFrame(_pm_new, index=X.index)], axis=1,
                    )
                    self.pairwise_modular_features_ = list(_pm_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_pm_appended)
                    for _r in _pm_recipes:
                        if _r.name in _pm_appended:
                            _pairwise_modular_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit pairwise_modular: appended %d engineered " "column(s): %s",
                            len(_pm_appended),
                            _pm_appended[:8],
                        )
            except Exception as _pm_exc:
                logger.warning(
                    "MRMR.fit pairwise-modular FE raised %s: %s; continuing without " "pairwise-modular columns.",
                    type(_pm_exc).__name__,
                    _pm_exc,
                )

    # Pairwise integer-lattice FE (sibling of pairwise-modular): detect a target that is a function of a hidden common
    # divisor (gcd), its dual lcm, or a bit-level co-occurrence (a & b) of integer columns - structure smooth/arithmetic/
    # modular ops cannot express. Cheap-first pairs-only scan + dual margin/permutation-null gate; budget-guarded.
    if _discrete_fe_master and _fe_family_on("fe_integer_lattice_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: integer-lattice FE enabled but X is not a pandas DataFrame; " "the features are skipped. Convert via X.to_pandas() before fit().",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from .._integer_lattice_fe import (
                    apply_integer_lattice,
                    hybrid_integer_lattice_fe_with_recipes,
                )

                # Class-MI floor: 1D classification feeds directly, continuous 1D is quantile-binned once, 2D stays skipped (see modular note).
                # Reuses the shared _y_class_mi_* binned above.
                _il_appended, _il_recipes = ([], [])
                if _y_class_mi_applicable:
                    _y_il_binned = _y_class_mi_binned
                    # Raw-column operands only (excludes pmod_/orth engineered columns added upstream); see the modular note.
                    _il_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]
                    _il_appended, _il_recipes = hybrid_integer_lattice_fe_with_recipes(
                        X, _y_il_binned,  # type: ignore[arg-type]
                        cols=_il_raw_cols,
                        top_k=int(getattr(self, "fe_integer_lattice_top_k", 4)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                        max_int_cols=int(getattr(self, "fe_integer_lattice_max_int_cols", 30)),
                    )
                _il_appended = [c for c in _il_appended if c not in X.columns]
                if _il_appended:
                    _il_new = {
                        _r.name: apply_integer_lattice(
                            X, _r.extra["op"], _r.src_names,
                        )
                        for _r in _il_recipes if _r.name in _il_appended
                    }
                    X = pd.concat(
                        [X, pd.DataFrame(_il_new, index=X.index)], axis=1,
                    )
                    self.integer_lattice_features_ = list(_il_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_il_appended)
                    for _r in _il_recipes:
                        if _r.name in _il_appended:
                            _integer_lattice_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit integer_lattice: appended %d engineered " "column(s): %s",
                            len(_il_appended),
                            _il_appended[:8],
                        )
            except Exception as _il_exc:
                logger.warning(
                    "MRMR.fit integer-lattice FE raised %s: %s; continuing without " "integer-lattice columns.",
                    type(_il_exc).__name__,
                    _il_exc,
                )

    # Row-argmax FE (frontier pass 2): for a column triple (a, b, c) emit the integer index 0/1/2 of the row-maximum - an
    # ordinal/comparison pattern the MI/linear path cannot read off marginals or pairwise diffs. ZERO free params, detector-clean;
    # leak-free deterministic replay (np.argmax over the stacked source columns). Budget-guarded on wide frames.
    if _discrete_fe_master and _fe_family_on("fe_row_argmax_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: row-argmax FE enabled but X is not a pandas DataFrame; " "the features are skipped. Convert via X.to_pandas() before fit().",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from .._conditional_gate_fe import (
                    apply_row_argmax,
                    hybrid_row_argmax_fe_with_recipes,
                )

                # Class-MI floor: 1D classification feeds directly, continuous 1D is quantile-binned once, 2D stays skipped (see modular note).
                # Reuses the shared _y_class_mi_* binned above.
                _am_appended, _am_recipes = ([], [])
                if _y_class_mi_applicable:
                    _y_am_binned = _y_class_mi_binned
                    # Raw-column operands only (excludes pmod_/il_/orth engineered columns added upstream); combining on already-
                    # engineered columns yields nested recipes whose engineered source is not resolvable at replay -> NaN drop.
                    _am_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]
                    _am_appended, _am_recipes = hybrid_row_argmax_fe_with_recipes(
                        X, _y_am_binned,  # type: ignore[arg-type]
                        cols=_am_raw_cols,
                        top_k=int(getattr(self, "fe_row_argmax_top_k", 4)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                        max_cols=int(getattr(self, "fe_row_argmax_max_cols", 30)),
                    )
                _am_appended = [c for c in _am_appended if c not in X.columns]
                if _am_appended:
                    _am_new = {_r.name: apply_row_argmax(X, _r.src_names) for _r in _am_recipes if _r.name in _am_appended}
                    X = pd.concat(
                        [X, pd.DataFrame(_am_new, index=X.index)], axis=1,
                    )
                    self.row_argmax_features_ = list(_am_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_am_appended)
                    for _r in _am_recipes:
                        if _r.name in _am_appended:
                            _row_argmax_pre_recipes[_r.name] = _r
                            # Record the raw source operands so the FE step keeps them as
                            # regularly-selected pair operands (see _gate_raw_operands_ init).
                            self._gate_raw_operands_.update(str(s) for s in _r.src_names)
                            self._gate_col_src_vars_[str(_r.name)] = {str(s) for s in _r.src_names}
                    if verbose:
                        logger.info(
                            "MRMR.fit row_argmax: appended %d engineered " "column(s): %s",
                            len(_am_appended),
                            _am_appended[:8],
                        )
            except Exception as _am_exc:
                logger.warning(
                    "MRMR.fit row-argmax FE raised %s: %s; continuing without " "row-argmax columns.",
                    type(_am_exc).__name__,
                    _am_exc,
                )

    # Conditional-gate FE (frontier pass 2): detect a regime switch c>tau ? a : b (select) or a masked interaction 1[c>tau]*a
    # (mask) routed by a third column's data-dependent threshold tau (frozen in the recipe). HARDENED detector gates vs the
    # best-existing-op MI (not the raw single-operand floor) so smooth/ordinary_mul controls stay silent. Budget-guarded.
    if _discrete_fe_master and _fe_family_on("fe_conditional_gate_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: conditional-gate FE enabled but X is not a pandas DataFrame; " "the features are skipped. Convert via X.to_pandas() before fit().",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from .._conditional_gate_fe import (
                    apply_conditional_gate,
                    hybrid_conditional_gate_fe_with_recipes,
                )

                # The gate detector's MI floor is class-MI (_mi_classif_batch). A CONTINUOUS regression target is quantile-binned once
                # (bin_y_for_class_mi) before the tau-grid + conditional-divergence sweep - the prior int64 cast turned continuous y into ~n
                # distinct classes (the tau-sweep MI exploded / never completed). A 2D y stays skipped (the kernel reads a dead signal).
                # Reuses the shared _y_class_mi_* binned above.
                _cg_appended, _cg_recipes = ([], [])
                if _y_class_mi_applicable:
                    _y_cg_binned = _y_class_mi_binned
                    # Raw-column operands only (see the row-argmax / modular note); engineered operands would orphan at replay.
                    _cg_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]
                    _cg_appended, _cg_recipes = hybrid_conditional_gate_fe_with_recipes(
                        X, _y_cg_binned,  # type: ignore[arg-type]
                        cols=_cg_raw_cols,
                        top_k=int(getattr(self, "fe_conditional_gate_top_k", 4)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                        max_cols=int(getattr(self, "fe_conditional_gate_max_cols", 200)),
                        k_gate=int(getattr(self, "fe_conditional_gate_k_gate", 8)),
                        k_operand=int(getattr(self, "fe_conditional_gate_k_operand", 10)),
                        # SCREEN SUBSAMPLE: subsample the gate-DETECTION scan (tau + MI
                        # ranking are rank-stable; the recipe replays the gate at FULL n). Reuse the
                        # resolved screen-n (fe_check_pairs_subsample_n) UNCONDITIONALLY - the default-
                        # screen profile shrinks it for large n on every fit, so the gate-detection
                        # (n, K) float64 buffer is built on the small sample and no longer OOMs + gets
                        # silently skipped. >=n / 0 keeps the legacy full-n scan (small-n unchanged).
                        subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                    )
                _cg_appended = [c for c in _cg_appended if c not in X.columns]
                if _cg_appended:
                    _cg_new = {
                        _r.name: apply_conditional_gate(
                            X, _r.extra["mode"], _r.src_names, _r.extra["tau"],
                        )
                        for _r in _cg_recipes if _r.name in _cg_appended
                    }
                    X = pd.concat(
                        [X, pd.DataFrame(_cg_new, index=X.index)], axis=1,
                    )
                    self.conditional_gate_features_ = list(_cg_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cg_appended)
                    for _r in _cg_recipes:
                        if _r.name in _cg_appended:
                            _conditional_gate_pre_recipes[_r.name] = _r
                            # Record the raw source operands so the FE step keeps them as
                            # regularly-selected pair operands (see _gate_raw_operands_ init).
                            self._gate_raw_operands_.update(str(s) for s in _r.src_names)
                            self._gate_col_src_vars_[str(_r.name)] = {str(s) for s in _r.src_names}
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_gate: appended %d engineered " "column(s): %s",
                            len(_cg_appended),
                            _cg_appended[:8],
                        )
            except Exception as _cg_exc:
                logger.warning(
                    "MRMR.fit conditional-gate FE raised %s: %s; continuing without " "conditional-gate columns.",
                    type(_cg_exc).__name__,
                    _cg_exc,
                )

    # Layer 95 PART B: per-group distribution-distance. For each
    # (group, num) emit the group-level z / KL / Wasserstein-1 distance from the
    # global distribution, broadcast to rows; each survivor MI-gated against the
    # source num_col marginal MI. Routing piggybacks on hybrid_orth_features_.
    if _fe_family_on("fe_group_distance_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 95 group_distance FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._group_distance_fe import hybrid_group_distance_fe

                _y_for_gd = _y_np
                _gd_groups = tuple(getattr(self, "fe_group_distance_group_cols", ()) or ())
                _gd_groups = [c for c in _gd_groups if c in X.columns] or None  # type: ignore[assignment]
                _gd_nums = tuple(getattr(self, "fe_group_distance_num_cols", ()) or ())
                _gd_nums = [c for c in _gd_nums if c in X.columns] or None  # type: ignore[assignment]
                _gd_top_k = int(getattr(self, "fe_group_distance_top_k", 6))
                _X_before_gd_cols = list(X.columns)
                X_gd, _gd_appended, _gd_recipes, _gd_scores = hybrid_group_distance_fe(
                    X,
                    _y_for_gd,
                    group_cols=_gd_groups,
                    num_cols=_gd_nums,
                    top_k=_gd_top_k,
                )
                _gd_appended = [c for c in _gd_appended if c not in _X_before_gd_cols]
                if _gd_appended:
                    X = X_gd
                    self.group_distance_features_ = list(_gd_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_gd_appended)
                    for _r in _gd_recipes:
                        if _r.name in _gd_appended:
                            _group_distance_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit group_distance: appended %d engineered " "column(s): %s",
                            len(_gd_appended),
                            _gd_appended[:8],
                        )
            except Exception as _gd_exc:
                logger.warning(
                    "MRMR.fit group_distance FE raised %s: %s; continuing " "without group-distance columns.",
                    type(_gd_exc).__name__,
                    _gd_exc,
                )

    # Layer 104: THREE new recipe-based FE families.
    # Family D: conditional dispersion / 2nd-moment.
    self.rare_category_features_ = []
    self.conditional_residual_features_ = []
    self.conditional_dispersion_features_ = []
    self.conditional_quantile_rank_features_ = []
    self.ordinal_pattern_features_ = []
    self.random_fourier_features_ = []
    self.sir_direction_features_ = []
    self.lof_features_ = []
    self.mahalanobis_density_features_ = []
    self.wavelet_features_ = []
    self.rankgauss_features_ = []

    # FAMILY A - rare-category indicator + frequency-band encoding. A category
    # being RARE is itself predictive; emit is_rare_{col} + freq_band_{col}.
    # MI-gated against the raw-baseline floor. Routing piggybacks on
    # hybrid_orth_features_.
    if _fe_family_on("fe_rare_category_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 104 rare_category FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_rare_category_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                # W6 follow-up: rare-category family's unified local-MI abs-MAD
                # floor kills (pure-record; selection byte-identical).
                _rc_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _rc_reject_sink(**_kw):
                    """Reject-sink callback for the rare-category FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_rc_step, **_kw)

                _y_for_rc = _y_np
                _rc_cols = tuple(getattr(self, "fe_rare_category_cols", ()) or ())
                _rc_cols = [c for c in _rc_cols if c in X.columns] or None  # type: ignore[assignment]
                _X_before_rc_cols = list(X.columns)
                _rc_raw_floor = X[[c for c in _raw_input_cols_pre_fe if c in X.columns]] if _raw_input_cols_pre_fe else None
                X_rc, _rc_appended, _rc_recipes, _ = hybrid_rare_category_fe(
                    X, _y_for_rc,
                    cat_cols=_rc_cols,
                    rare_threshold=float(getattr(self, "fe_rare_category_threshold", 0.01)),
                    top_k=int(getattr(self, "fe_rare_category_top_k", 10)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_rc_reject_sink,
                    raw_floor_X=_rc_raw_floor,
                )
                _rc_appended = [c for c in _rc_appended if c not in _X_before_rc_cols]
                if _rc_appended:
                    X = X_rc
                    self.rare_category_features_ = list(_rc_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rc_appended)
                    for _r in _rc_recipes:
                        if _r.name in _rc_appended:
                            _rare_category_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit rare_category: appended %d engineered " "column(s): %s",
                            len(_rc_appended),
                            _rc_appended[:8],
                        )
            except Exception as _rc_exc:
                logger.warning(
                    "MRMR.fit rare_category FE raised %s: %s; continuing " "without rare-category columns.",
                    type(_rc_exc).__name__,
                    _rc_exc,
                )

    # FAMILY B - NUM x NUM conditional residual x_i - E[x_i | bin(x_j)].
    # Cardinality-bounded by top raw-MI columns; MI-gated. Routing piggybacks on
    # hybrid_orth_features_.
    if _fe_family_on("fe_conditional_residual_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 104 conditional_residual FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_conditional_residual_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                # W6 follow-up: conditional-residual family's unified local-MI
                # abs-MAD floor kills (pure-record; selection byte-identical).
                _cr_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _cr_reject_sink(**_kw):
                    """Reject-sink callback for the num-x-num conditional-residual FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_cr_step, **_kw)

                _y_for_cr = _y_np
                _cr_cols = tuple(getattr(self, "fe_conditional_residual_cols", ()) or ())
                _cr_cols = [c for c in _cr_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (mirrors conditional_dispersion / wavelet): X is
                # already augmented with engineered intermediates here, and a
                # conditional-residual recipe built on an engineered x_i / x_j source
                # cannot be replayed at transform() (the engineered parent is not
                # present in the apply X) -> KeyError. Scope auto-detect to raw cols.
                if _cr_cols is None:
                    _cr_raw = set(_raw_input_cols_pre_fe)
                    _cr_cols = [c for c in X.columns if c in _cr_raw] or None
                _X_before_cr_cols = list(X.columns)
                _cr_raw_floor = X[[c for c in _raw_input_cols_pre_fe if c in X.columns]] if _raw_input_cols_pre_fe else None
                X_cr, _cr_appended, _cr_recipes, _ = hybrid_conditional_residual_fe(
                    X, _y_for_cr,
                    num_cols=_cr_cols,
                    n_bins=int(getattr(self, "fe_conditional_residual_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_residual_top_k", 10)),
                    max_pair_cols=int(getattr(self, "fe_conditional_residual_max_pair_cols", 6)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_cr_reject_sink,
                    raw_floor_X=_cr_raw_floor,
                )
                _cr_appended = [c for c in _cr_appended if c not in _X_before_cr_cols]
                if _cr_appended:
                    X = X_cr
                    self.conditional_residual_features_ = list(_cr_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cr_appended)
                    for _r in _cr_recipes:
                        if _r.name in _cr_appended:
                            _conditional_residual_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_residual: appended %d " "engineered column(s): %s",
                            len(_cr_appended),
                            _cr_appended[:8],
                        )
            except Exception as _cr_exc:
                logger.warning(
                    "MRMR.fit conditional_residual FE raised %s: %s; continuing " "without conditional-residual columns.",
                    type(_cr_exc).__name__,
                    _cr_exc,
                )

    # FAMILY D - NUM x NUM conditional DISPERSION / 2nd-moment.
    # Bin x_j; per bin store conditional STD of x_i; emit |z| / z^2 (conditional
    # dispersion anomaly). DEFAULT-ON: MI-gateable (|z| is a non-monotone fold ->
    # genuine MI on heteroscedastic targets) + SELF-LIMITING (a dual-uplift gate
    # admits a column only when its MI beats BOTH raw x_i AND the |mean-residual|
    # Family-B sibling, so homoscedastic / canonical fixtures admit 0 and the
    # operator does not perturb pair-FE recovery). Routing piggybacks on
    # hybrid_orth_features_; recipes carry no y -> leak-safe replay.
    if _fe_family_on("fe_conditional_dispersion_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Family D conditional_dispersion FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_conditional_dispersion_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                # W6 follow-up: conditional-dispersion family's unified local-MI
                # abs-MAD floor kills (pure-record; selection byte-identical).
                _cd_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _cd_reject_sink(**_kw):
                    """Reject-sink callback for the num-x-num conditional-dispersion FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_cd_step, **_kw)

                _y_for_cd = _y_np
                _cd_cols = tuple(getattr(self, "fe_conditional_dispersion_cols", ()) or ())
                _cd_cols = [c for c in _cd_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (2026-06-10 fix, same class as the wavelet stage
                # below): the all-numeric default scope over the already-augmented X
                # builds dispersion features OF engineered columns -> nested recipes
                # the 1-deep replay cannot order at transform() time (KeyError on the
                # engineered parent when it is not selected). Raw scope keeps every
                # conditional-dispersion recipe replayable.
                # ``feature_names_in_`` is not yet assigned here; scope to the raw
                # pre-FE column snapshot (the cat_pair / cat_triple guard's ledger),
                # which is strictly safer than the ``hybrid_orth_features_`` exclusion
                # - that ledger only tracks orth / hinge / wavelet columns and misses
                # ratio / grouped-agg / numeric-decompose engineered intermediates a
                # dispersion recipe would otherwise build on and fail to replay.
                if _cd_cols is None:
                    _cd_raw = set(_raw_input_cols_pre_fe)
                    _cd_cols = [c for c in X.columns if c in _cd_raw] or None
                _X_before_cd_cols = list(X.columns)
                X_cd, _cd_appended, _cd_recipes, _ = hybrid_conditional_dispersion_fe(
                    X, _y_for_cd,
                    num_cols=_cd_cols,
                    n_bins=int(getattr(self, "fe_conditional_dispersion_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_dispersion_top_k", 10)),
                    max_pair_cols=int(getattr(self, "fe_conditional_dispersion_max_pair_cols", 6)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_cd_reject_sink,
                )
                _cd_appended = [c for c in _cd_appended if c not in _X_before_cd_cols]
                if _cd_appended:
                    X = X_cd
                    self.conditional_dispersion_features_ = list(_cd_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cd_appended)
                    for _r in _cd_recipes:
                        if _r.name in _cd_appended:
                            _conditional_dispersion_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_dispersion: appended %d " "engineered column(s): %s",
                            len(_cd_appended),
                            _cd_appended[:8],
                        )
            except Exception as _cd_exc:
                logger.warning(
                    "MRMR.fit conditional_dispersion FE raised %s: %s; continuing " "without conditional-dispersion columns.",
                    type(_cd_exc).__name__,
                    _cd_exc,
                )

    # CONDITIONAL QUANTILE-RANK: 4th member of the
    # conditional-dispersion family. Bin x_j; emit q(row) = empirical_rank(x_i within bin(x_j)) -
    # the row's TRUE within-bin percentile, not a z-score. MI-gated + self-limiting (a near-
    # monotone reparametrization on homoscedastic/non-skewed data clears no uplift over raw x_i, so
    # it does not perturb genuine-feature recovery on canonical fixtures). Routing piggybacks on
    # hybrid_orth_features_; recipes carry no y -> leak-safe replay.
    if _fe_family_on("fe_conditional_quantile_rank_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: conditional_quantile_rank FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._conditional_quantile_rank_fe import hybrid_conditional_quantile_rank_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _cqr_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _cqr_reject_sink(**_kw):
                    """Reject-sink callback for the num-x-num conditional-quantile-rank FE stage; records
                    MI-floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_cqr_step, **_kw)

                _y_for_cqr = _y_np
                _cqr_cols = tuple(getattr(self, "fe_conditional_quantile_rank_cols", ()) or ())
                _cqr_cols = [c for c in _cqr_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion / wavelet above): the
                # all-numeric default scope over the already-augmented X builds quantile-rank
                # features OF engineered columns -> nested recipes the 1-deep replay cannot order
                # at transform() time. Raw scope keeps every recipe replayable.
                if _cqr_cols is None:
                    _cqr_raw = set(_raw_input_cols_pre_fe)
                    _cqr_cols = [c for c in X.columns if c in _cqr_raw] or None
                _X_before_cqr_cols = list(X.columns)
                X_cqr, _cqr_appended, _cqr_recipes, _ = hybrid_conditional_quantile_rank_fe(
                    X, _y_for_cqr,
                    num_cols=_cqr_cols,
                    n_bins=int(getattr(self, "fe_conditional_quantile_rank_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_quantile_rank_top_k", 10)),
                    max_pair_cols=int(getattr(self, "fe_conditional_quantile_rank_max_pair_cols", 6)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_cqr_reject_sink,
                )
                _cqr_appended = [c for c in _cqr_appended if c not in _X_before_cqr_cols]
                if _cqr_appended:
                    X = X_cqr
                    self.conditional_quantile_rank_features_ = list(_cqr_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cqr_appended)
                    for _r in _cqr_recipes:
                        if _r.name in _cqr_appended:
                            _conditional_quantile_rank_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_quantile_rank: appended %d " "engineered column(s): %s",
                            len(_cqr_appended),
                            _cqr_appended[:8],
                        )
            except Exception as _cqr_exc:
                logger.warning(
                    "MRMR.fit conditional_quantile_rank FE raised %s: %s; continuing " "without conditional-quantile-rank columns.",
                    type(_cqr_exc).__name__,
                    _cqr_exc,
                )

    # ORDINAL PATTERN (Bandt-Pompe) K-fold TARGET ENCODING.
    # For each K-tuple of raw numeric columns, compute the row's rank-permutation id (0..K!-1) and
    # K-fold-TE encode it - a fused single-hop recipe: the intermediate perm_id categorical is
    # never exposed as its own column, avoiding a 2-deep nested-recipe replay the 1-deep convention
    # here cannot order. Routing piggybacks on hybrid_orth_features_; recipe carries a frozen
    # (fit-time) TE lookup, not y -> leak-safe replay.
    if _fe_family_on("fe_ordinal_pattern_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: ordinal_pattern FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._ordinal_pattern_fe import hybrid_ordinal_pattern_te_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _opat_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _opat_reject_sink(**_kw):
                    """Reject-sink callback for the ordinal-pattern-TE FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_opat_step, **_kw)

                _y_for_opat = _y_np
                _opat_cols = tuple(getattr(self, "fe_ordinal_pattern_cols", ()) or ())
                _opat_cols = [c for c in _opat_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank above): the
                # all-numeric default scope over the already-augmented X builds ordinal patterns OF
                # engineered columns -> nested recipes the 1-deep replay cannot order at
                # transform() time. Raw scope keeps every recipe replayable.
                if _opat_cols is None:
                    _opat_raw = set(_raw_input_cols_pre_fe)
                    _opat_cols = [c for c in X.columns if c in _opat_raw] or None
                _X_before_opat_cols = list(X.columns)
                X_opat, _opat_appended, _opat_recipes, _ = hybrid_ordinal_pattern_te_fe(
                    X, _y_for_opat,
                    num_cols=_opat_cols,
                    k=int(getattr(self, "fe_ordinal_pattern_k", 3)),
                    max_cols_for_tuples=int(getattr(self, "fe_ordinal_pattern_max_cols_for_tuples", 5)),
                    n_folds=int(getattr(self, "fe_ordinal_pattern_n_folds", 5)),
                    smoothing=float(getattr(self, "fe_ordinal_pattern_smoothing", 10.0)),
                    top_k=int(getattr(self, "fe_ordinal_pattern_top_k", 5)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_opat_reject_sink,
                )
                _opat_appended = [c for c in _opat_appended if c not in _X_before_opat_cols]
                if _opat_appended:
                    X = X_opat
                    self.ordinal_pattern_features_ = list(_opat_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_opat_appended)
                    for _r in _opat_recipes:
                        if _r.name in _opat_appended:
                            _ordinal_pattern_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit ordinal_pattern: appended %d " "engineered column(s): %s",
                            len(_opat_appended),
                            _opat_appended[:8],
                        )
            except Exception as _opat_exc:
                logger.warning(
                    "MRMR.fit ordinal_pattern FE raised %s: %s; continuing " "without ordinal-pattern columns.",
                    type(_opat_exc).__name__,
                    _opat_exc,
                )

    # RANDOM FOURIER FEATURES (random kitchen sinks) joint kernel-approximation block
    # . Unlike every pair/triplet/quadruplet cross-basis
    # family, this draws m random features that are jointly a smooth function of MANY (5+) raw
    # columns simultaneously without combinatorial blow-up, approximating an RBF kernel over the
    # bounded column pool. Routing piggybacks on hybrid_orth_features_; recipe carries the frozen
    # W-column/phase/bandwidth, never y -> leak-safe replay.
    if _fe_family_on("fe_random_fourier_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: random_fourier FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._random_fourier_features_fe import hybrid_random_fourier_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _rff_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _rff_reject_sink(**_kw):
                    """Reject-sink callback for the random-fourier FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_rff_step, **_kw)

                _y_for_rff = _y_np
                _rff_cols = tuple(getattr(self, "fe_random_fourier_cols", ()) or ())
                _rff_cols = [c for c in _rff_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern
                # above): the all-numeric default scope over the already-augmented X builds RFF
                # features OF engineered columns -> nested recipes the 1-deep replay cannot order at
                # transform() time. Raw scope keeps every recipe replayable.
                if _rff_cols is None:
                    _rff_raw = set(_raw_input_cols_pre_fe)
                    _rff_cols = [c for c in X.columns if c in _rff_raw] or None
                _X_before_rff_cols = list(X.columns)
                X_rff, _rff_appended, _rff_recipes, _ = hybrid_random_fourier_fe(
                    X, _y_for_rff,
                    num_cols=_rff_cols,
                    m=int(getattr(self, "fe_random_fourier_m", 64)),
                    bandwidth=getattr(self, "fe_random_fourier_bandwidth", None),
                    max_cols_for_block=int(getattr(self, "fe_random_fourier_max_cols_for_block", 8)),
                    top_k=int(getattr(self, "fe_random_fourier_top_k", 8)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_rff_reject_sink,
                )
                _rff_appended = [c for c in _rff_appended if c not in _X_before_rff_cols]
                if _rff_appended:
                    X = X_rff
                    self.random_fourier_features_ = list(_rff_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rff_appended)
                    for _r in _rff_recipes:
                        if _r.name in _rff_appended:
                            _random_fourier_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit random_fourier: appended %d " "engineered column(s): %s",
                            len(_rff_appended),
                            _rff_appended[:8],
                        )
            except Exception as _rff_exc:
                logger.warning(
                    "MRMR.fit random_fourier FE raised %s: %s; continuing " "without random-fourier columns.",
                    type(_rff_exc).__name__,
                    _rff_exc,
                )

    # SLICED INVERSE REGRESSION (SIR) oblique-direction projection (
    # fe_expansion.md). Recovers a genuinely OBLIQUE (rotated) linear combination spread thinly
    # across several correlated columns - where every individual weight is too small for that
    # column's own marginal MI to clear the screening floor, and no pairwise/triplet/quadruplet
    # product reconstructs the rotated hyperplane. Routing piggybacks on hybrid_orth_features_;
    # recipe carries the frozen centering/direction, not y -> leak-safe replay.
    if _fe_family_on("fe_sir_direction_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: sir_direction FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._sliced_inverse_regression_fe import hybrid_sir_direction_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _sir_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _sir_reject_sink(**_kw):
                    """Reject-sink callback for the SIR-direction FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_sir_step, **_kw)

                _y_for_sir = _y_np
                _sir_cols = tuple(getattr(self, "fe_sir_direction_cols", ()) or ())
                _sir_cols = [c for c in _sir_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern/
                # random_fourier above): the all-numeric default scope over the already-augmented X
                # builds SIR directions OF engineered columns -> nested recipes the 1-deep replay
                # cannot order at transform() time. Raw scope keeps every recipe replayable.
                if _sir_cols is None:
                    _sir_raw = set(_raw_input_cols_pre_fe)
                    _sir_cols = [c for c in X.columns if c in _sir_raw] or None
                _X_before_sir_cols = list(X.columns)
                X_sir, _sir_appended, _sir_recipes, _ = hybrid_sir_direction_fe(
                    X, _y_for_sir,
                    num_cols=_sir_cols,
                    n_slices=int(getattr(self, "fe_sir_direction_n_slices", 10)),
                    n_directions=int(getattr(self, "fe_sir_direction_n_directions", 2)),
                    max_cols_for_block=int(getattr(self, "fe_sir_direction_max_cols_for_block", 8)),
                    top_k=int(getattr(self, "fe_sir_direction_top_k", 2)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_sir_reject_sink,
                )
                _sir_appended = [c for c in _sir_appended if c not in _X_before_sir_cols]
                if _sir_appended:
                    X = X_sir
                    self.sir_direction_features_ = list(_sir_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_sir_appended)
                    for _r in _sir_recipes:
                        if _r.name in _sir_appended:
                            _sir_direction_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit sir_direction: appended %d " "engineered column(s): %s",
                            len(_sir_appended),
                            _sir_appended[:8],
                        )
            except Exception as _sir_exc:
                logger.warning(
                    "MRMR.fit sir_direction FE raised %s: %s; continuing " "without sir-direction columns.",
                    type(_sir_exc).__name__,
                    _sir_exc,
                )

    # LOCAL OUTLIER FACTOR / k-NN local density-ratio.
    # LOCAL and non-parametric (unlike a global Mahalanobis ellipsoid), catching a row anomalous
    # for sitting in a locally-sparse gap between well-separated clusters even when its GLOBAL
    # distance to the overall mean is unremarkable. Routing piggybacks on hybrid_orth_features_;
    # recipe carries a bounded frozen reference sample (RAM discipline), never y or the whole fit
    # frame -> leak-safe replay.
    if _fe_family_on("fe_lof_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: lof FE enabled but X is not a " "pandas DataFrame; the features are skipped. Convert via " "X.to_pandas() before fit() to apply them.",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from .._lof_fe import hybrid_lof_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _lof_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _lof_reject_sink(**_kw):
                    """Reject-sink callback for the LOF FE stage; records MI-floor kills into the
                    FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_lof_step, **_kw)

                _y_for_lof = _y_np
                _lof_cols = tuple(getattr(self, "fe_lof_cols", ()) or ())
                _lof_cols = [c for c in _lof_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern/
                # random_fourier/sir_direction above): the all-numeric default scope over the
                # already-augmented X builds LOF scores OF engineered columns -> nested recipes the
                # 1-deep replay cannot order at transform() time. Raw scope keeps every recipe replayable.
                if _lof_cols is None:
                    _lof_raw = set(_raw_input_cols_pre_fe)
                    _lof_cols = [c for c in X.columns if c in _lof_raw] or None
                _X_before_lof_cols = list(X.columns)
                X_lof, _lof_appended, _lof_recipes, _ = hybrid_lof_fe(
                    X, _y_for_lof,
                    num_cols=_lof_cols,
                    k=int(getattr(self, "fe_lof_k", 20)),
                    max_ref=int(getattr(self, "fe_lof_max_ref", 2000)),
                    max_cols_for_block=int(getattr(self, "fe_lof_max_cols_for_block", 8)),
                    top_k=int(getattr(self, "fe_lof_top_k", 1)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_lof_reject_sink,
                )
                _lof_appended = [c for c in _lof_appended if c not in _X_before_lof_cols]
                if _lof_appended:
                    X = X_lof
                    self.lof_features_ = list(_lof_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_lof_appended)
                    for _r in _lof_recipes:
                        if _r.name in _lof_appended:
                            _lof_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit lof: appended %d " "engineered column(s): %s",
                            len(_lof_appended),
                            _lof_appended[:8],
                        )
            except Exception as _lof_exc:
                logger.warning(
                    "MRMR.fit lof FE raised %s: %s; continuing " "without lof columns.",
                    type(_lof_exc).__name__,
                    _lof_exc,
                )

    # MULTIVARIATE MAHALANOBIS / GAUSSIAN-COPULA JOINT DENSITY anomaly score (
    # fe_expansion.md). Catches y depending on whether a row sits inside/outside an ELLIPSOIDAL
    # level-set of a p=15-30-way joint distribution where no single column, pair, triplet, or even
    # quadruplet cross-basis is individually extreme - the p-way generalization of the existing
    # group_distance / conditional-dispersion families' one-column-conditioned-on-one-other-column
    # scope. Routing piggybacks on hybrid_orth_features_; recipe carries the frozen Ledoit-Wolf
    # mu/Sigma_inv, never y -> leak-safe replay.
    if _fe_family_on("fe_mahalanobis_density_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: mahalanobis_density FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._mahalanobis_density_fe import hybrid_mahalanobis_density_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _mahal_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _mahal_reject_sink(**_kw):
                    """Reject-sink callback for the Mahalanobis-density FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_mahal_step, **_kw)

                _y_for_mahal = _y_np
                _mahal_cols = tuple(getattr(self, "fe_mahalanobis_density_cols", ()) or ())
                _mahal_cols = [c for c in _mahal_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern/
                # random_fourier/sir_direction/lof above): the all-numeric default scope over the
                # already-augmented X builds Mahalanobis density OF engineered columns -> nested
                # recipes the 1-deep replay cannot order at transform() time. Raw scope keeps every
                # recipe replayable.
                if _mahal_cols is None:
                    _mahal_raw = set(_raw_input_cols_pre_fe)
                    _mahal_cols = [c for c in X.columns if c in _mahal_raw] or None
                _X_before_mahal_cols = list(X.columns)
                X_mahal, _mahal_appended, _mahal_recipes, _ = hybrid_mahalanobis_density_fe(
                    X, _y_for_mahal,
                    num_cols=_mahal_cols,
                    max_cols_for_block=int(getattr(self, "fe_mahalanobis_density_max_cols_for_block", 20)),
                    top_k=int(getattr(self, "fe_mahalanobis_density_top_k", 1)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_mahal_reject_sink,
                )
                _mahal_appended = [c for c in _mahal_appended if c not in _X_before_mahal_cols]
                if _mahal_appended:
                    X = X_mahal
                    self.mahalanobis_density_features_ = list(_mahal_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_mahal_appended)
                    for _r in _mahal_recipes:
                        if _r.name in _mahal_appended:
                            _mahalanobis_density_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit mahalanobis_density: appended %d " "engineered column(s): %s",
                            len(_mahal_appended),
                            _mahal_appended[:8],
                        )
            except Exception as _mahal_exc:
                logger.warning(
                    "MRMR.fit mahalanobis_density FE raised %s: %s; continuing " "without mahalanobis-density columns.",
                    type(_mahal_exc).__name__,
                    _mahal_exc,
                )

    # HAAR WAVELET / localized multiresolution basis.
    # A NEW operator for LOCALIZED bump / multiscale piecewise structure: y jumps
    # only inside a narrow sub-window of x (Fourier Gibbs-rings it, spline's fixed
    # quantile knots smooth it away). Emits a small held-out-scale-selected dyadic
    # set of Haar indicators psi_{j,k} (+1 left / -1 right half of a dyadic
    # interval). DEFAULT-ON + SELF-LIMITING: the noise-aware held-out MAD floor +
    # max-legs cap bound the candidate explosion, and each leg is admitted on its
    # held-out INCREMENTAL MI over raw x AND a complementarity guard (must beat a
    # SMOOTH location-refinement of x) - so a localized step/bump admits legs, a
    # SMOOTH (sin / monotone) column admits 0 (Fourier owns it, complementary),
    # pure noise admits 0. The leg is NON-monotone -> MI-VISIBLE, so it routes
    # through the MI-based gate (no deferred-materialise / re-add dance the
    # MI-invariant hinge needs). Recipes (``orth_wavelet``) store (lo, span) +
    # dyadic (j, k); replay is the closed-form indicator - no y, leak-safe.
    # Routing piggybacks on hybrid_orth_features_ (like Family D dispersion).
    if _fe_family_on("fe_wavelet_enable", False) and _fe_budget_ok():
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Haar wavelet FE enabled but X is not a pandas DataFrame; "
                "the features are skipped. Convert via X.to_pandas() before fit() "
                "to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._wavelet_basis_fe_recipes import hybrid_wavelet_fe_with_recipes

                _y_for_wv = _y_np
                _wv_cols = tuple(getattr(self, "fe_wavelet_cols", ()) or ())
                _wv_cols = [c for c in _wv_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (2026-06-10 fix, mirrors the extra-basis stage's
                # guard at the hybrid_orth call above): by this point X is ALREADY
                # augmented with poly/fourier/spline/hinge engineered columns, so the
                # all-numeric default scope emitted NESTED recipes (e.g.
                # ``x0__p2sin1__haar_j3k5`` - a Haar leg of an engineered Fourier
                # column) whose 1-deep replay cannot order the parent materialisation
                # and raised KeyError('x0__p2sin1') at transform() time whenever the
                # parent was not itself selected. Scoping to ``feature_names_in_``
                # keeps every wavelet recipe 1-deep and replayable.
                # NOTE: ``self.feature_names_in_`` is not assigned until the
                # target-injection block far below, so the exclusion source is the
                # ``hybrid_orth_features_`` ledger every prior univariate stage
                # appends to (the hinge stage's exact pattern).
                if _wv_cols is None:
                    _wv_already = set(getattr(self, "hybrid_orth_features_", None) or [])
                    _wv_cols = [c for c in X.columns if c not in _wv_already] or None
                _X_before_wv_cols = list(X.columns)
                X_wv, _wv_appended, _wv_recipes, _ = hybrid_wavelet_fe_with_recipes(
                    X, _y_for_wv,
                    cols=_wv_cols,
                    max_scale=int(getattr(self, "fe_wavelet_max_scale", 3)),
                    max_legs=int(getattr(self, "fe_wavelet_max_legs", 6)),
                    top_k=int(getattr(self, "fe_wavelet_top_k", 8)),
                    feature_dtype=getattr(self, "usability_feature_dtype", np.float32),
                    max_cols=getattr(self, "fe_wavelet_max_cols", None),
                )
                _wv_appended = [c for c in _wv_appended if c not in _X_before_wv_cols]
                if _wv_appended:
                    X = X_wv
                    self.wavelet_features_ = list(_wv_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_wv_appended)
                    for _r in _wv_recipes:
                        if _r.name in _wv_appended:
                            _wavelet_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit wavelet: appended %d engineered column(s): %s",
                            len(_wv_appended), _wv_appended[:8],
                        )
            except Exception as _wv_exc:
                logger.warning(
                    "MRMR.fit Haar wavelet FE raised %s: %s; continuing without " "wavelet columns.",
                    type(_wv_exc).__name__,
                    _wv_exc,
                )

    # FAMILY C - RankGauss (rank-Gaussianisation). NOT MI-gated: monotone ->
    # MI-invariant by the data-processing inequality; the pool is bounded by raw
    # marginal MI and the value is downstream (linear / NN). Routing piggybacks
    # on hybrid_orth_features_.
    if _fe_family_on("fe_rankgauss_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 104 rankgauss FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_rankgauss_fe

                _y_for_rg = _y_np
                _rg_cols = tuple(getattr(self, "fe_rankgauss_cols", ()) or ())
                _rg_cols = [c for c in _rg_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (2026-06-10 fix, same class as the wavelet /
                # conditional-dispersion stages): keep rankgauss recipes 1-deep and
                # replayable - never rank-Gaussianise an engineered column whose
                # parent the transform()-time replay cannot materialise first.
                # ``feature_names_in_`` is not yet assigned here; exclude via the
                # ``hybrid_orth_features_`` ledger (hinge-stage pattern).
                if _rg_cols is None:
                    _rg_already = set(getattr(self, "hybrid_orth_features_", None) or [])
                    _rg_cols = [c for c in X.columns if c not in _rg_already] or None
                _X_before_rg_cols = list(X.columns)
                X_rg, _rg_appended, _rg_recipes, _ = hybrid_rankgauss_fe(
                    X, _y_for_rg,
                    num_cols=_rg_cols,
                    top_k=int(getattr(self, "fe_rankgauss_top_k", 10)),
                )
                _rg_appended = [c for c in _rg_appended if c not in _X_before_rg_cols]
                if _rg_appended:
                    X = X_rg
                    self.rankgauss_features_ = list(_rg_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rg_appended)
                    for _r in _rg_recipes:
                        if _r.name in _rg_appended:
                            _rankgauss_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit rankgauss: appended %d engineered " "column(s): %s",
                            len(_rg_appended),
                            _rg_appended[:8],
                        )
            except Exception as _rg_exc:
                logger.warning(
                    "MRMR.fit rankgauss FE raised %s: %s; continuing without " "rankgauss columns.",
                    type(_rg_exc).__name__,
                    _rg_exc,
                )

    # Layer 92: temporal leak-safe grouped aggregations. Carved
    # verbatim into the sibling ``_fe_stage_temporal_agg`` (Tier E partial
    # split); the helper threads self + ``_y_np`` / ``verbose`` /
    # ``_temporal_agg_pre_recipes`` explicitly, mutates self + the recipes dict
    # in place, and RETURNS the (possibly replaced) working ``X`` frame.
    from ._fe_stage_temporal_agg import _fe_stage_temporal_agg
    X = _fe_stage_temporal_agg(self, X, _y_np, verbose, _temporal_agg_pre_recipes)

    # ACCURACY GATE (2026-06-04, default ON via ``fe_accuracy_gate``). The MI-uplift gates inside the FE generators are fooled by plug-in MI's bias inflation: a Fourier / chirp / Hermite transform of a strong RAW signal earns an inflated MI estimate and out-ranks (then evicts) the raw column even when it adds NO real predictive value. The adaptive-Fourier PROTECTION block at support-finalisation then force-readds those hijackers past the MRMR screen, so they survive into support_ AND leak into ``hybrid_orth_features_`` / ``_adaptive_fourier_features_`` even when a genuine raw signal (or its is_missing__ MNAR indicator) carries the information. This gate runs a held-out multivariate linear-probe uplift check per engineered column against its raw source: a column that adds no held-out uplift over its source - or whose source is >2%-missing (MNAR fail-closed, the signal lives in the NaN pattern the probe cannot see) - is dropped here so it can neither evict the raw signal nor leak into the roster. Only orth_* engineered columns with a single resolvable raw source are gated; the is_missing__ / missingness_* indicators are exempt by construction (their recipes live in ``_miss_*_pre_recipes``, never ``_hybrid_orth_pre_recipes``, so they are never routed here). y is read only at fit; transform replays the survivors without y. Best-effort: any failure falls back to keeping the column.
    if bool(getattr(self, "fe_accuracy_gate", True)) and isinstance(X, pd.DataFrame) and (self.hybrid_orth_features_ or []) and _hybrid_orth_pre_recipes:
        try:
            from .._fe_accuracy_gate import (
                _FE_UPLIFT_MIN,
                infer_classification,
                keep_engineered_over_source,
                measure_feature_uplift,
            )

            _y_for_gate = _y_np
            _gate_seed = int(getattr(self, "random_seed", 0) or 0)
            _gate_classif = infer_classification(_y_for_gate)
            _hybrid_set_now = set(self.hybrid_orth_features_ or [])
            _adaptive_set_now = set(getattr(self, "_adaptive_fourier_features_", None) or [])

            def _gate_col_arr(_name):
                """Fetch column ``_name`` from ``X`` as a float64 1-D array for the held-out linear-probe accuracy gate (unwraps a duplicate-label DataFrame slice to its first column)."""
                _v = X[_name]
                if isinstance(_v, pd.DataFrame):
                    _v = _v.iloc[:, 0]
                return np.asarray(_v.to_numpy(), dtype=np.float64)

            # Resolve each engineered column to its single raw source; split into the polynomial/base columns and the adaptive-Fourier/chirp columns (the latter are gated CONDITIONALLY
            # against their surviving base siblings, since a Fourier of x captures the SAME x**2 signal as its He2 sibling and must not dilute the support when the He2 already carries it).
            _gate_cols: list[tuple[str, str, bool]] = []
            for _gc in list(self.hybrid_orth_features_ or []):
                if _gc not in X.columns:
                    continue
                _rec = _hybrid_orth_pre_recipes.get(_gc)
                # No hybrid-orth recipe => not an orth_* engineered column (missingness / TE / count / etc.): exempt.
                _src_names = tuple(getattr(_rec, "src_names", ()) or ()) if _rec is not None else ()
                if len(_src_names) != 1:
                    continue
                _src = _src_names[0]
                if _src not in X.columns or _src in _hybrid_set_now:
                    continue
                _is_fourier = (_gc in _adaptive_set_now) or (str(getattr(_rec, "kind", "")) == "orth_fourier")
                _gate_cols.append((_gc, _src, _is_fourier))

            _gate_drop: list[str] = []
            _gate_drop_set: set[str] = set()
            # Pass 1: base (non-Fourier) columns - uplift over the raw source alone (also the MNAR fail-closed for >2%-missing sources).
            _surviving_base_by_src: dict[str, list[str]] = {}
            for _gc, _src, _is_fourier in _gate_cols:
                if _is_fourier:
                    continue
                _src_arr = _gate_col_arr(_src)
                _eng_arr = _gate_col_arr(_gc)
                if keep_engineered_over_source(_src_arr, _eng_arr, _y_for_gate, seed=_gate_seed):
                    _surviving_base_by_src.setdefault(_src, []).append(_gc)
                else:
                    _gate_drop.append(_gc)
                    _gate_drop_set.add(_gc)
            # Pass 2: adaptive-Fourier / chirp columns - uplift over [raw source + surviving base siblings of that source]. A Fourier redundant with a He2 sibling (both encode x**2)
            # adds ~0 here and is dropped; a genuine oscillation no polynomial sibling captures clears the floor and is kept. MNAR fail-closed first (the probe drops NaN rows).
            for _gc, _src, _is_fourier in _gate_cols:
                if not _is_fourier:
                    continue
                _src_arr = _gate_col_arr(_src)
                if float(np.mean(~np.isfinite(_src_arr))) > 0.02:
                    _gate_drop.append(_gc)
                    _gate_drop_set.add(_gc)
                    continue
                _base_sibs = _surviving_base_by_src.get(_src, [])
                _base_mat = np.column_stack([_src_arr] + [_gate_col_arr(_b) for _b in _base_sibs])
                _eng_arr = _gate_col_arr(_gc)
                _n = _base_mat.shape[0]
                if _n > 5000:
                    _rng_g = np.random.default_rng(_gate_seed)
                    _idx_g = _rng_g.choice(_n, 5000, replace=False)
                    _base_probe, _eng_probe, _y_probe = _base_mat[_idx_g], _eng_arr[_idx_g], _y_for_gate[_idx_g]
                else:
                    _base_probe, _eng_probe, _y_probe = _base_mat, _eng_arr, _y_for_gate
                _cond_uplift = measure_feature_uplift(
                    _base_probe, _eng_probe, _y_probe, classification=_gate_classif, seed=_gate_seed,
                )
                # Fail-open: None == probe could not measure (degenerate / exception);
                # keep the candidate rather than silently dropping it. Only a genuine
                # MEASURED sub-threshold uplift evicts.
                if _cond_uplift is not None and _cond_uplift < _FE_UPLIFT_MIN:
                    _gate_drop.append(_gc)
                    _gate_drop_set.add(_gc)
            if _gate_drop:
                _gate_drop_set = set(_gate_drop)
                X = X.drop(columns=[c for c in _gate_drop if c in X.columns])
                self.hybrid_orth_features_ = [c for c in (self.hybrid_orth_features_ or []) if c not in _gate_drop_set]
                self._adaptive_fourier_features_ = [c for c in (getattr(self, "_adaptive_fourier_features_", None) or []) if c not in _gate_drop_set]
                # Mirror the cleanup for hinge legs: a hinge the accuracy gate
                # drops (no held-out uplift over its raw source) must NOT be
                # re-added by the HINGE-PROTECTION block, so prune it here too.
                self._hinge_features_ = [c for c in (getattr(self, "_hinge_features_", None) or []) if c not in _gate_drop_set]
                for _c in list(_hybrid_orth_pre_recipes.keys()):
                    if _c in _gate_drop_set:
                        _hybrid_orth_pre_recipes.pop(_c, None)
                if verbose:
                    logger.info(
                        "MRMR.fit accuracy gate: dropped %d engineered column(s) " "adding no held-out uplift over their raw source (or MNAR " "source): %s",
                        len(_gate_drop),
                        sorted(_gate_drop),
                    )
        except Exception as _gate_exc:
            logger.warning(
                "MRMR.fit accuracy gate raised %s: %s; continuing without the " "accuracy gate (engineered columns kept).",
                type(_gate_exc).__name__,
                _gate_exc,
            )

    # Layer 27: cross-stage engineered-column dedup. Hybrid and
    # MI-greedy stages run independently; on signals like ``y = sign(x^2 - 1)``
    # hybrid emits ``x__He2`` and MI-greedy emits ``square(x)`` / ``abs(x)`` /
    # ``sqrt_abs(x)`` / ``log_abs(x)`` - all are monotone-in-|x| encodings
    # of the SAME signal (Pearson |corr| ~ 0.99+ on rank-correlated MI binning).
    # MRMR's CMI gate can't tell them apart well enough to prune; the
    # combined support inflates with 4-5 near-identical columns. The cheap
    # cure is a pre-MRMR dedup pass against the engineered cousins: keep the
    # first appended occurrence, drop everything correlating >= 0.999 with an
    # already-kept engineered column. Raw input columns are never deduped
    # here - that's MRMR's job and removing raw cols would change the
    # ``feature_names_in_`` contract.
    # Order-preserving dedup BEFORE we walk the list: when the same
    # engineered name is emitted by both the hybrid_orth and the
    # mi_greedy stages (e.g. both produce ``square(x1)`` under a
    # signal-driven recipe), ``X[name]`` selects a 2-column DataFrame
    # rather than a Series and the downstream ``.rank()`` call
    # explodes with ``Data must be 1-dimensional``. The dedup also
    # short-circuits the inner O(K^2) pairwise rank-correlation loop
    # for the trivial perfect-name-match case.
    _eng_cols_appended_raw = list(self.hybrid_orth_features_ or []) + list(self.mi_greedy_features_ or [])
    _eng_seen: set[str] = set()
    _eng_cols_appended = [_c for _c in _eng_cols_appended_raw if not (_c in _eng_seen or _eng_seen.add(_c))]  # type: ignore[func-returns-value]  # intentional order-preserving-dedup idiom: set.add()'s None return is used as the falsy side of `or`
    # ADAPTIVE-FOURIER columns are NEVER pruned by the cross-stage dedup: the
    # held-out detector already validated the frequency, and a sin/cos pair at
    # one frequency is not monotone-equivalent to a fixed-grid twin, so the
    # Spearman gate would only ever drop them on a spurious near-tie. Keeping
    # them here guarantees they remain in ``cols`` for the protection block.
    _adaptive_fourier_keep = set(getattr(self, "_adaptive_fourier_features_", None) or [])
    # Keep-higher-MI dedup policy: when a near-duplicate cluster spans stages, the survivor must be the column carrying the MOST information about y, NOT merely the first-appended one.
    # The default-on univariate-basis stage writes into ``hybrid_orth_features_`` and is appended BEFORE ``mi_greedy_features_``, so a first-appended policy silently sacrifices a genuine
    # mi_greedy ``|x|``-family signal (``log_abs(x)`` / ``sqrt_abs(x)`` / ``square(x)`` / ``abs(x)``) to a monotone-equivalent basis twin (``x__L2`` / ``x__cos1`` / ...). We score every appended
    # engineered column once with the SAME plug-in MI scorer + quantile binning the FE stages used, then break dedup ties by higher MI, with the mi_greedy / constructor-requested column winning
    # exact MI ties (a monotone twin bins identically, so MI is numerically equal - prefer the explicitly-requested constructor output). MI scoring is best-effort: any failure falls back to the
    # order-preserving first-appended policy so the dedup never crashes a fit.
    _mig_set = set(self.mi_greedy_features_ or [])
    _eng_mi: dict[str, float] = {}
    try:
        from .._orthogonal_univariate_fe import _mi_classif_batch
        _y_for_eng_mi = _y_np
        if _y_for_eng_mi.dtype.kind in "fc":
            _n_unique_eng = int(np.unique(_y_for_eng_mi).size)
            if _n_unique_eng <= 32:
                _y_for_eng_mi = _y_for_eng_mi.astype(np.int64)
            else:
                try:
                    _y_for_eng_mi = pd.qcut(_y_for_eng_mi, q=10, labels=False, duplicates="drop").astype(np.int64)
                except Exception as exc:
                    logger.debug("mrmr: y densification failed for the engineered-MI dedup pass; falling back to truncating int64 cast: %r", exc, exc_info=True)
                    _y_for_eng_mi = _y_for_eng_mi.astype(np.int64)
        else:
            _y_for_eng_mi = _y_for_eng_mi.astype(np.int64)
        if isinstance(X, pd.DataFrame) and len(_eng_cols_appended) >= 2:
            _mi_cols = [_c for _c in _eng_cols_appended if _c in X.columns]
            if _mi_cols:
                _mi_mat = X[_mi_cols].to_numpy(dtype=np.float64)
                _mi_vals = _mi_classif_batch(_mi_mat, _y_for_eng_mi, nbins=10)
                _eng_mi = {_name: float(_v) for _name, _v in zip(_mi_cols, _mi_vals)}
    except Exception as exc:
        logger.debug("mrmr: engineered-MI dict computation failed; treating as empty (no engineered candidates this round): %r", exc, exc_info=True)
        _eng_mi = {}

    def _eng_dedup_prefer(cand: str, kept: str) -> bool:
        """Return True when ``cand`` should DISPLACE the already-kept ``kept`` on a near-duplicate collision.

        Only CROSS-STAGE collisions (exactly one of the pair is an mi_greedy / constructor-requested column) ever flip the survivor: within a single stage we preserve the original
        first-appended policy byte-for-byte, so the dedup stays deterministic on the monotone-twin families a single basis stage emits (a quantile-binned MI tie between ``x__He2`` /
        ``x__cos1`` / ``x__L2`` would otherwise reshuffle non-deterministically). Across stages we keep the column carrying more MI about y, and the explicitly-requested mi_greedy column
        wins an exact MI tie (a monotone twin bins identically, so its MI is numerically equal - without this the default-on basis twin would silently evict the genuine ``|x|``-family signal).
        """
        _cand_mig = cand in _mig_set
        _kept_mig = kept in _mig_set
        if _cand_mig == _kept_mig:
            return False
        _mi_cand = _eng_mi.get(cand)
        _mi_kept = _eng_mi.get(kept)
        if _mi_cand is None or _mi_kept is None:
            return False
        if _mi_cand > _mi_kept + 1e-12:
            return True
        if _mi_cand >= _mi_kept - 1e-12:
            return _cand_mig and not _kept_mig
        return False

    if len(_eng_cols_appended) >= 2 and isinstance(X, pd.DataFrame):
        _eng_keep: list[str] = []
        _eng_drop: set[str] = set()
        _eng_arrs: dict[str, np.ndarray] = {}
        # Cache each column's FULL-column average ranks. When a (candidate, kept) pair is jointly finite
        # over ALL rows (the common no-NaN engineered case) the masked-subset ranks equal these full ranks,
        # so we reuse them instead of re-sorting both columns per pair - removing the O(K^2) rank-sorts the
        # dedup did (only the O(K^2) corrcoef remains). Bit-identical: same arrays -> same average ranks.
        _eng_ranks: dict[str, np.ndarray] = {}
        # Per-column full-finiteness (zero NaN), cached alongside the ranks: the fast-path condition below
        # (``_mask.all()``) can only ever be True when BOTH sides are fully finite, so a fully-finite
        # candidate's O(K) kept-column comparisons can be BATCHED in one parallel call (see
        # ``_eng_dedup_batch_corr.one_vs_many_abs_corr_masked``) instead of K separate ``np.corrcoef``
        # calls - only for the subset of kept columns that are themselves fully finite (and hence carry a
        # buffer row); a NaN-containing kept/candidate pair keeps the original per-pair masked path.
        # APPEND-ONLY rank buffer (2026-07-13, bench-attempt-rejected note in _eng_dedup_batch_corr.py):
        # a naive per-candidate ``np.vstack`` of the CURRENT kept ranks re-copies O(K) rows on EVERY
        # candidate (O(K^2 * n) total memcpy, the SAME order as the corrcoef calls it replaces - measured
        # a NET LOSS). This buffer is written ONCE per fully-finite column (when first admitted) and never
        # copied again; the kernel takes a zero-copy VIEW of it plus a boolean "still live" mask.
        _eng_rank_buf = np.empty((len(_eng_cols_appended), len(X)), dtype=np.float64)
        _eng_row_of: dict[str, int] = {}
        _eng_next_free_row = 0
        _eng_fully_finite: dict[str, bool] = {}
        for _c in _eng_cols_appended:
            if _c in _eng_drop:
                continue
            if _c in _adaptive_fourier_keep:
                # Force-keep adaptive Fourier columns; record their array so
                # later candidates can still be deduped AGAINST them.
                _col_view_a = X[_c]
                if isinstance(_col_view_a, pd.DataFrame):
                    _col_view_a = _col_view_a.iloc[:, 0]
                _eng_keep.append(_c)
                _eng_arrs[_c] = np.asarray(_col_view_a.to_numpy(), dtype=np.float64)
                continue
            # Defense in depth: if X carries duplicate column labels (a
            # caller-side data-quality issue we don't want to silently
            # mask but can't crash on either), ``X[_c]`` returns a
            # DataFrame; collapse to the first column so rank/corrcoef
            # downstream see a 1-D array and the cross-stage dedup
            # still runs.
            _col_view = X[_c]
            if isinstance(_col_view, pd.DataFrame):
                _col_view = _col_view.iloc[:, 0]
            _arr_c = np.asarray(_col_view.to_numpy(), dtype=np.float64)
            _fin_c = np.isfinite(_arr_c)
            _eng_fully_finite[_c] = bool(_fin_c.all())
            if not _fin_c.any() or _arr_c[_fin_c].std() <= 1e-12:
                _eng_keep.append(_c)
                _eng_arrs[_c] = _arr_c
                continue
            # Rank-correlate (Spearman) rather than Pearson: MRMR's plug-in
            # MI scorer quantile-bins each column before computing MI, so
            # two engineered columns related by ANY monotone reshape (square
            # vs |x| vs log|x|) project to identical bin sequences and carry
            # identical information about y. Pearson at 0.999 catches only
            # the perfect linear case (e.g. x^2 vs x^2-1); Spearman at 0.99
            # catches the full monotone-equivalent family that MRMR's
            # downstream gate cannot distinguish.
            # Full-column ranks of the candidate, cached (reused below when a pair is fully finite).
            _ranks_c = pd.Series(_arr_c).rank(method="average").to_numpy()
            _eng_ranks[_c] = _ranks_c
            _colliding_kept: list[str] = []
            # BATCHED FAST PATH: the ``_mask.all()`` fast-path condition below can only ever
            # be True when BOTH the candidate and the kept column are fully finite - so when the candidate
            # itself is fully finite, every currently-kept column that is ALSO fully finite (and hence
            # already has a row in the append-only rank buffer) can be compared in ONE batched+parallel
            # call instead of one ``np.corrcoef`` call per kept column. Kept columns with any NaN (rare
            # per this loop's own comment) fall through to the unchanged per-pair path below, unaffected.
            _fast_kept_set: set = set()
            if _eng_fully_finite[_c] and _arr_c.shape[0] >= 8 and _eng_next_free_row > 0:
                from ._eng_dedup_batch_corr import one_vs_many_abs_corr_masked
                _active_mask = np.zeros(_eng_next_free_row, dtype=np.bool_)
                _row_to_kc: dict[int, str] = {}
                for _kc in _eng_keep:
                    _r = _eng_row_of.get(_kc)
                    if _r is not None:
                        _active_mask[_r] = True
                        _row_to_kc[_r] = _kc
                if _active_mask.any():
                    _fast_corrs = one_vs_many_abs_corr_masked(_ranks_c, _eng_rank_buf[:_eng_next_free_row], _active_mask)
                    for _r, _kc in _row_to_kc.items():
                        _fast_kept_set.add(_kc)
                        if _fast_corrs[_r] >= 0.99:
                            _colliding_kept.append(_kc)
            for _kept_col in _eng_keep:
                if _kept_col in _fast_kept_set:
                    continue
                _arr_k = _eng_arrs[_kept_col]
                _mask = _fin_c & np.isfinite(_arr_k)
                if _mask.sum() < 8:
                    continue
                _a, _b = _arr_c[_mask], _arr_k[_mask]
                if _a.std() <= 1e-12 or _b.std() <= 1e-12:
                    continue
                if bool(_mask.all()):
                    # No-NaN fast path: masked subset == full column, so reuse the cached full-column ranks
                    # (identical values) instead of re-sorting both columns for this pair.
                    _ranks_a = _ranks_c
                    _ranks_b = _eng_ranks.get(_kept_col)
                    if _ranks_b is None:
                        _ranks_b = pd.Series(_arr_k).rank(method="average").to_numpy()
                        _eng_ranks[_kept_col] = _ranks_b
                else:
                    _ranks_a = pd.Series(_a).rank(method="average").to_numpy()
                    _ranks_b = pd.Series(_b).rank(method="average").to_numpy()
                if _ranks_a.std() <= 1e-12 or _ranks_b.std() <= 1e-12:
                    continue
                _rank_corr = abs(float(np.corrcoef(_ranks_a, _ranks_b)[0, 1]))
                if np.isfinite(_rank_corr) and _rank_corr >= 0.99:
                    _colliding_kept.append(_kept_col)
            if _colliding_kept:
                # Keep-higher-MI: the candidate displaces every colliding kept column it out-scores, and is itself dropped only if some colliding kept column wins.
                # ``_eng_dedup_prefer`` returns False when MI is unavailable, so an unscored cluster degrades exactly to the original first-appended policy (candidate dropped).
                _cand_loses = any(not _eng_dedup_prefer(_c, _kept_col) for _kept_col in _colliding_kept)
                if _cand_loses:
                    _eng_drop.add(_c)
                else:
                    for _kept_col in _colliding_kept:
                        _eng_drop.add(_kept_col)
                        _eng_keep.remove(_kept_col)
                        _eng_arrs.pop(_kept_col, None)
                    _eng_keep.append(_c)
                    _eng_arrs[_c] = _arr_c
                    if _eng_fully_finite[_c]:
                        _eng_rank_buf[_eng_next_free_row] = _ranks_c
                        _eng_row_of[_c] = _eng_next_free_row
                        _eng_next_free_row += 1
            else:
                _eng_keep.append(_c)
                _eng_arrs[_c] = _arr_c
                if _eng_fully_finite[_c]:
                    _eng_rank_buf[_eng_next_free_row] = _ranks_c
                    _eng_row_of[_c] = _eng_next_free_row
                    _eng_next_free_row += 1
        if _eng_drop:
            # Dependency-closure guard: never drop an engineered column / recipe that a
            # SURVIVING recipe consumes via src_names (e.g. a cat_pair_cross producer
            # feeding a modular / numeric_decompose recipe). Dropping the producer while
            # keeping the consumer orphans the consumer's source -> KeyError at transform
            # replay. Fixpoint over all recipe dicts so multi-level chains stay intact.
            _all_pre_recipe_dicts = (
                _hybrid_orth_pre_recipes, _mi_greedy_pre_recipes, _kfold_te_pre_recipes,
                _binned_agg_pre_recipes,
                _count_enc_pre_recipes, _freq_enc_pre_recipes, _cat_num_pre_recipes,
                _miss_ind_pre_recipes, _miss_cnt_pre_recipes, _miss_pat_pre_recipes,
                _ratio_pre_recipes, _log_ratio_pre_recipes, _grouped_delta_pre_recipes,
                _lagged_diff_pre_recipes, _grouped_agg_pre_recipes,
                _composite_group_agg_pre_recipes, _grouped_quantile_pre_recipes,
                _cat_pair_pre_recipes, _cat_triple_pre_recipes,
                _numeric_decompose_pre_recipes, _modular_pre_recipes,
                _pairwise_modular_pre_recipes, _integer_lattice_pre_recipes,
                _row_argmax_pre_recipes, _conditional_gate_pre_recipes,
                _group_distance_pre_recipes, _rare_category_pre_recipes,
                _conditional_residual_pre_recipes,
                _conditional_dispersion_pre_recipes, _wavelet_pre_recipes,
                _rankgauss_pre_recipes,
                _temporal_agg_pre_recipes,
                _conditional_quantile_rank_pre_recipes,
                _ordinal_pattern_pre_recipes,
                _random_fourier_pre_recipes,
                _sir_direction_pre_recipes,
                _lof_pre_recipes,
                _mahalanobis_density_pre_recipes,
            )
            while True:
                _protected = {
                    _s for _d in _all_pre_recipe_dicts for _r in _d.values() if _r.name not in _eng_drop for _s in (getattr(_r, "src_names", ()) or ())
                }
                _newly = _eng_drop & _protected
                if not _newly:
                    break
                _eng_drop -= _newly
            X = X.drop(columns=list(_eng_drop))
            self.hybrid_orth_features_ = [c for c in (self.hybrid_orth_features_ or []) if c not in _eng_drop]
            # Mirror cleanup for hinge legs (a hinge near-duplicate of another
            # engineered column the Spearman dedup removed must not be re-added
            # by the HINGE-PROTECTION block).
            self._hinge_features_ = [c for c in (getattr(self, "_hinge_features_", None) or []) if c not in _eng_drop]
            self.mi_greedy_features_ = [c for c in (self.mi_greedy_features_ or []) if c not in _eng_drop]
            # Layer 33: mirror the same cleanup for TE-encoded columns.
            self.kfold_te_features_ = [c for c in (getattr(self, "kfold_te_features_", []) or []) if c not in _eng_drop]
            # Layer 34: mirror cleanup for count / freq / cat_num residual.
            self.count_encoding_features_ = [c for c in (getattr(self, "count_encoding_features_", []) or []) if c not in _eng_drop]
            self.frequency_encoding_features_ = [c for c in (getattr(self, "frequency_encoding_features_", []) or []) if c not in _eng_drop]
            self.cat_num_interaction_features_ = [c for c in (getattr(self, "cat_num_interaction_features_", []) or []) if c not in _eng_drop]
            # Layer 37: mirror cleanup for missingness indicator / count / pattern.
            self.missingness_indicator_features_ = [c for c in (getattr(self, "missingness_indicator_features_", []) or []) if c not in _eng_drop]
            self.missingness_count_features_ = [c for c in (getattr(self, "missingness_count_features_", []) or []) if c not in _eng_drop]
            self.missingness_pattern_features_ = [c for c in (getattr(self, "missingness_pattern_features_", []) or []) if c not in _eng_drop]
            # Layer 38: mirror cleanup for ratio / log_ratio / grouped_delta / lagged_diff.
            self.pairwise_ratio_features_ = [c for c in (getattr(self, "pairwise_ratio_features_", []) or []) if c not in _eng_drop]
            self.pairwise_log_ratio_features_ = [c for c in (getattr(self, "pairwise_log_ratio_features_", []) or []) if c not in _eng_drop]
            self.grouped_delta_features_ = [c for c in (getattr(self, "grouped_delta_features_", []) or []) if c not in _eng_drop]
            self.lagged_diff_features_ = [c for c in (getattr(self, "lagged_diff_features_", []) or []) if c not in _eng_drop]
            # Layer 87: mirror cleanup for grouped_agg.
            self.grouped_agg_features_ = [c for c in (getattr(self, "grouped_agg_features_", []) or []) if c not in _eng_drop]
            # Layer 93: mirror cleanup for composite_group_agg.
            self.composite_group_agg_features_ = [c for c in (getattr(self, "composite_group_agg_features_", []) or []) if c not in _eng_drop]
            # Layer 88: mirror cleanup for grouped_quantile.
            self.grouped_quantile_features_ = [c for c in (getattr(self, "grouped_quantile_features_", []) or []) if c not in _eng_drop]
            # Layer 89: mirror cleanup for cat_pair crosses.
            self.cat_pair_features_ = [c for c in (getattr(self, "cat_pair_features_", []) or []) if c not in _eng_drop]
            # Layer 94: mirror cleanup for cat_triple crosses.
            self.cat_triple_features_ = [c for c in (getattr(self, "cat_triple_features_", []) or []) if c not in _eng_drop]
            # Layer 90: mirror cleanup for numeric-decomposition columns.
            self.numeric_decompose_features_ = [c for c in (getattr(self, "numeric_decompose_features_", []) or []) if c not in _eng_drop]
            for _c in list(_hybrid_orth_pre_recipes.keys()):
                if _c in _eng_drop:
                    _hybrid_orth_pre_recipes.pop(_c, None)
            for _c in list(_mi_greedy_pre_recipes.keys()):
                if _c in _eng_drop:
                    _mi_greedy_pre_recipes.pop(_c, None)
            for _c in list(_kfold_te_pre_recipes.keys()):
                if _c in _eng_drop:
                    _kfold_te_pre_recipes.pop(_c, None)
            for _c in list(_count_enc_pre_recipes.keys()):
                if _c in _eng_drop:
                    _count_enc_pre_recipes.pop(_c, None)
            for _c in list(_freq_enc_pre_recipes.keys()):
                if _c in _eng_drop:
                    _freq_enc_pre_recipes.pop(_c, None)
            for _c in list(_cat_num_pre_recipes.keys()):
                if _c in _eng_drop:
                    _cat_num_pre_recipes.pop(_c, None)
            for _c in list(_miss_ind_pre_recipes.keys()):
                if _c in _eng_drop:
                    _miss_ind_pre_recipes.pop(_c, None)
            for _c in list(_miss_cnt_pre_recipes.keys()):
                if _c in _eng_drop:
                    _miss_cnt_pre_recipes.pop(_c, None)
            for _c in list(_miss_pat_pre_recipes.keys()):
                if _c in _eng_drop:
                    _miss_pat_pre_recipes.pop(_c, None)
            for _c in list(_ratio_pre_recipes.keys()):
                if _c in _eng_drop:
                    _ratio_pre_recipes.pop(_c, None)
            for _c in list(_log_ratio_pre_recipes.keys()):
                if _c in _eng_drop:
                    _log_ratio_pre_recipes.pop(_c, None)
            for _c in list(_grouped_delta_pre_recipes.keys()):
                if _c in _eng_drop:
                    _grouped_delta_pre_recipes.pop(_c, None)
            for _c in list(_lagged_diff_pre_recipes.keys()):
                if _c in _eng_drop:
                    _lagged_diff_pre_recipes.pop(_c, None)
            for _c in list(_grouped_agg_pre_recipes.keys()):
                if _c in _eng_drop:
                    _grouped_agg_pre_recipes.pop(_c, None)
            for _c in list(_composite_group_agg_pre_recipes.keys()):
                if _c in _eng_drop:
                    _composite_group_agg_pre_recipes.pop(_c, None)
            for _c in list(_grouped_quantile_pre_recipes.keys()):
                if _c in _eng_drop:
                    _grouped_quantile_pre_recipes.pop(_c, None)
            for _c in list(_cat_pair_pre_recipes.keys()):
                if _c in _eng_drop:
                    _cat_pair_pre_recipes.pop(_c, None)
            for _c in list(_cat_triple_pre_recipes.keys()):
                if _c in _eng_drop:
                    _cat_triple_pre_recipes.pop(_c, None)
            for _c in list(_numeric_decompose_pre_recipes.keys()):
                if _c in _eng_drop:
                    _numeric_decompose_pre_recipes.pop(_c, None)
            for _c in list(_modular_pre_recipes.keys()):
                if _c in _eng_drop:
                    _modular_pre_recipes.pop(_c, None)
            for _c in list(_pairwise_modular_pre_recipes.keys()):
                if _c in _eng_drop:
                    _pairwise_modular_pre_recipes.pop(_c, None)
            for _c in list(_integer_lattice_pre_recipes.keys()):
                if _c in _eng_drop:
                    _integer_lattice_pre_recipes.pop(_c, None)
            for _c in list(_row_argmax_pre_recipes.keys()):
                if _c in _eng_drop:
                    _row_argmax_pre_recipes.pop(_c, None)
            for _c in list(_conditional_gate_pre_recipes.keys()):
                if _c in _eng_drop:
                    _conditional_gate_pre_recipes.pop(_c, None)
            for _c in list(_group_distance_pre_recipes.keys()):
                if _c in _eng_drop:
                    _group_distance_pre_recipes.pop(_c, None)
            for _c in list(_rare_category_pre_recipes.keys()):
                if _c in _eng_drop:
                    _rare_category_pre_recipes.pop(_c, None)
            for _c in list(_conditional_residual_pre_recipes.keys()):
                if _c in _eng_drop:
                    _conditional_residual_pre_recipes.pop(_c, None)
            for _c in list(_conditional_dispersion_pre_recipes.keys()):
                if _c in _eng_drop:
                    _conditional_dispersion_pre_recipes.pop(_c, None)
            for _c in list(_conditional_quantile_rank_pre_recipes.keys()):
                if _c in _eng_drop:
                    _conditional_quantile_rank_pre_recipes.pop(_c, None)
            for _c in list(_ordinal_pattern_pre_recipes.keys()):
                if _c in _eng_drop:
                    _ordinal_pattern_pre_recipes.pop(_c, None)
            for _c in list(_random_fourier_pre_recipes.keys()):
                if _c in _eng_drop:
                    _random_fourier_pre_recipes.pop(_c, None)
            for _c in list(_sir_direction_pre_recipes.keys()):
                if _c in _eng_drop:
                    _sir_direction_pre_recipes.pop(_c, None)
            for _c in list(_lof_pre_recipes.keys()):
                if _c in _eng_drop:
                    _lof_pre_recipes.pop(_c, None)
            for _c in list(_mahalanobis_density_pre_recipes.keys()):
                if _c in _eng_drop:
                    _mahalanobis_density_pre_recipes.pop(_c, None)
            for _c in list(_wavelet_pre_recipes.keys()):
                if _c in _eng_drop:
                    _wavelet_pre_recipes.pop(_c, None)
            for _c in list(_rankgauss_pre_recipes.keys()):
                if _c in _eng_drop:
                    _rankgauss_pre_recipes.pop(_c, None)
            for _c in list(_temporal_agg_pre_recipes.keys()):
                if _c in _eng_drop:
                    _temporal_agg_pre_recipes.pop(_c, None)
            if verbose:
                logger.info(
                    "MRMR.fit engineered-FE dedup: pruned %d near-duplicate " "engineered column(s) at Spearman |rho| >= 0.99: %s",
                    len(_eng_drop),
                    sorted(_eng_drop),
                )

    # Layer 91: Tier-2 UNIFIED SECOND-PASS CMI GATE. The Layer 27
    # dedup above is UNSUPERVISED (Spearman rank-corr between engineered cousins)
    # and so cannot see cross-mechanism redundancy that only manifests
    # conditional on y - e.g. ``count(cat_a)`` and ``freq(cat_a)`` ARE caught by
    # Spearman (identical rank order), but ``count(cat_a)`` vs a target-encoding
    # of cat_a that carries the same y-signal through a different bin pattern is
    # NOT. This gate runs a single greedy CMI selection over ALL engineered
    # columns (every mechanism) conditioned on the running support seeded from
    # the top raw-MI columns, keeping only columns that add new information about
    # y on top of raw + earlier-selected engineered columns. Default OFF (byte-
    # identical legacy path). y is read only here at fit; transform replays the
    # surviving recipes without y.
    if bool(getattr(self, "fe_unified_second_pass_gate", False)) and isinstance(X, pd.DataFrame):
        try:
            _eng_now = [c for c in (list(self.hybrid_orth_features_ or []) + list(self.mi_greedy_features_ or [])) if c in X.columns]
            # Order-preserving unique.
            _seen_u: set[str] = set()
            _eng_now = [c for c in _eng_now if not (c in _seen_u or _seen_u.add(c))]  # type: ignore[func-returns-value]  # intentional order-preserving-dedup idiom: set.add()'s None return is used as the falsy side of `or`
            if len(_eng_now) >= 2:
                from .._unified_fe_gate import unified_second_pass_gate

                _raw_cols_u = [c for c in X.columns if c not in set(_eng_now)]
                _y_for_u = _y_np
                _keep_u = set(
                    unified_second_pass_gate(
                        X,
                        _y_for_u,
                        raw_cols=_raw_cols_u,
                        engineered_cols=_eng_now,
                        max_keep=getattr(self, "fe_unified_second_pass_max_keep", None),
                        min_cmi_gain=float(getattr(self, "fe_unified_second_pass_min_gain", 0.005)),
                    )
                )
                _eng_drop_u = set(_eng_now) - _keep_u
                if _eng_drop_u:
                    # Record what the FE stages produced BEFORE this pass prunes it (see the roster-
                    # reconciliation snapshot near the end of fit for why the pre-prune view is kept).
                    self.hybrid_orth_candidates_ = list(
                        dict.fromkeys(list(getattr(self, "hybrid_orth_candidates_", None) or []) + list(getattr(self, "hybrid_orth_features_", None) or []))
                    )
                    X = X.drop(columns=list(_eng_drop_u))
                    for _attr in (
                        "hybrid_orth_features_", "mi_greedy_features_",
                        "kfold_te_features_", "count_encoding_features_",
                        "frequency_encoding_features_",
                        "cat_num_interaction_features_",
                        "missingness_indicator_features_",
                        "missingness_count_features_",
                        "missingness_pattern_features_",
                        "pairwise_ratio_features_", "pairwise_log_ratio_features_",
                        "grouped_delta_features_", "lagged_diff_features_",
                        "grouped_agg_features_", "composite_group_agg_features_",
                        "grouped_quantile_features_",
                        "cat_pair_features_", "cat_triple_features_",
                        "numeric_decompose_features_",
                        "modular_features_", "group_distance_features_",
                        "rare_category_features_",
                        "conditional_residual_features_",
                        "conditional_dispersion_features_", "wavelet_features_",
                        "rankgauss_features_",
                        "temporal_agg_features_",
                    ):
                        setattr(self, _attr, [c for c in (getattr(self, _attr, []) or []) if c not in _eng_drop_u])
                    # Private hinge / adaptive-fourier protection rosters are not
                    # in the public-roster loop above; prune them explicitly so a
                    # second-pass-dropped leg is not re-added by its protection.
                    self._hinge_features_ = [c for c in (getattr(self, "_hinge_features_", None) or []) if c not in _eng_drop_u]
                    self._adaptive_fourier_features_ = [c for c in (getattr(self, "_adaptive_fourier_features_", None) or []) if c not in _eng_drop_u]
                    for _pre in (
                        _hybrid_orth_pre_recipes, _mi_greedy_pre_recipes,
                        _kfold_te_pre_recipes, _count_enc_pre_recipes,
                        _freq_enc_pre_recipes, _cat_num_pre_recipes,
                        _miss_ind_pre_recipes, _miss_cnt_pre_recipes,
                        _miss_pat_pre_recipes, _ratio_pre_recipes,
                        _log_ratio_pre_recipes, _grouped_delta_pre_recipes,
                        _lagged_diff_pre_recipes, _grouped_agg_pre_recipes,
                        _composite_group_agg_pre_recipes,
                        _grouped_quantile_pre_recipes, _cat_pair_pre_recipes,
                        _cat_triple_pre_recipes,
                        _numeric_decompose_pre_recipes,
                        _modular_pre_recipes, _pairwise_modular_pre_recipes,
                        _integer_lattice_pre_recipes,
                        _row_argmax_pre_recipes, _conditional_gate_pre_recipes,
                        _group_distance_pre_recipes,
                        _rare_category_pre_recipes,
                        _conditional_residual_pre_recipes,
                        _conditional_dispersion_pre_recipes,
                        _wavelet_pre_recipes,
                        _rankgauss_pre_recipes,
                        _temporal_agg_pre_recipes,
                        _conditional_quantile_rank_pre_recipes,
                        _ordinal_pattern_pre_recipes,
                        _random_fourier_pre_recipes,
                        _sir_direction_pre_recipes,
                        _lof_pre_recipes,
                        _mahalanobis_density_pre_recipes,
                    ):
                        for _c in list(_pre.keys()):
                            if _c in _eng_drop_u:
                                _pre.pop(_c, None)
                    if verbose:
                        logger.info(
                            "MRMR.fit unified second-pass CMI gate: pruned %d " "cross-mechanism redundant engineered column(s): %s",
                            len(_eng_drop_u),
                            sorted(_eng_drop_u),
                        )
        except Exception as _u_exc:
            logger.warning(
                "MRMR.fit unified_second_pass_gate raised %s: %s; continuing " "without the Tier-2 cross-mechanism gate.",
                type(_u_exc).__name__,
                _u_exc,
            )

    # Layer 23: feature_names_in_ MUST exclude hybrid-appended columns so
    # the end-of-fit ``selected_vars_names`` lookup routes hybrid names
    # into ``_engineered_features_`` / ``_engineered_recipes_`` instead of
    # the raw-feature ``original_indices`` path. transform() then replays
    # hybrid columns from recipes and the sklearn ``n_features_in_``
    # contract still matches the user-facing input width.
    # Layer 26: also exclude MI-greedy-appended columns - same routing
    # contract: they're engineered, not raw input.
    _hybrid_names_set = set(self.hybrid_orth_features_ or [])
    _mi_greedy_names_set = set(self.mi_greedy_features_ or [])
    _engineered_names_set = _hybrid_names_set | _mi_greedy_names_set
    _all_cols = X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
    # Defense in depth (Layer 64 finding 2026-05-31): if any FE stage
    # accidentally appended a column under a name already present in
    # X (e.g. two recipe families converging on the same canonical
    # ``square(x1)`` label, or a stage re-emitting an input column it
    # picked up from a previous stage), pandas downstream raises
    # ``cannot reindex on an axis with duplicate labels`` when
    # ``X.loc[:, target_names] = vals`` runs the target injection.
    # Drop in-place: keep the FIRST occurrence (which is the original
    # raw input column or the first stage's emission), drop later
    # duplicate-named columns, and prune the engineered roster of any
    # name that was effectively shadowed so the recipe ledger stays
    # consistent with the column actually surviving in X.
    if isinstance(X, pd.DataFrame) and X.columns.has_duplicates:
        # Layer 64 defense: keep only the FIRST occurrence
        # of each duplicate-label column position in X. The engineered
        # rosters and the recipe ledger are NOT pruned here - the
        # recipe is what the transform path uses to re-emit the column,
        # so dropping the name from the roster would break
        # ``transform`` (it tries to look up the support_ name in the
        # input X, doesn't find the recipe replay output, and raises
        # "MRMR.transform: N/K selected columns missing from input X").
        # The duplicate is purely a fit-time X-frame artefact (one FE
        # stage re-emitted a column another stage already appended);
        # the recipe replay produces a single canonical column at
        # transform time.
        _seen_cols: set[str] = set()
        _keep_positions: list[int] = []
        _shadowed_eng_names: set[str] = set()
        _n_dropped = 0
        for _i, _c in enumerate(_all_cols):
            if _c in _seen_cols:
                if _c in _engineered_names_set:
                    _shadowed_eng_names.add(_c)
                _n_dropped += 1
                continue
            _seen_cols.add(_c)
            _keep_positions.append(_i)
        X = X.iloc[:, _keep_positions].copy()
        _all_cols = X.columns.tolist()
        if verbose:
            logger.warning(
                "MRMR.fit: pruned %d duplicate column label(s) before "
                "target injection; engineered names shadowed (kept "
                "first occurrence + recipe ledger entry intact): %s",
                _n_dropped,
                sorted(_shadowed_eng_names),
            )
    # When embedding/text passthrough narrowed X above, ``_all_cols`` lacks the passthrough columns; ``feature_names_in_`` must still reflect the FULL user-facing
    # input (passthrough columns included, in their original positions) so the sklearn ``n_features_in_`` contract matches transform's input width. The passthrough
    # indices are re-appended to ``support_`` at fit-end so transform re-emits them.
    _names_source = getattr(self, "_passthrough_full_columns_", None) if self._passthrough_features_ else None
    if _names_source is not None:
        _fni = [c for c in _names_source if c not in _engineered_names_set]
    else:
        _fni = [c for c in _all_cols if c not in _engineered_names_set]
    # ndarray (not list) to match sklearn's own feature_names_in_ contract (BaseEstimator._check_feature_names) and
    # every other MRMR fit-path assignment (_mrmr_class_fit_helpers.py); a plain list here was the one straggler
    # that made ``==`` comparisons against a list/array ambiguous for callers expecting the canonical type.
    self.feature_names_in_ = np.asarray(_fni, dtype=object)
    self.n_features_in_ = len(self.feature_names_in_)

    # FE AUTO-ESCALATION fitting target: a RANK transform of the raw
    # numeric y, stashed for the escalation proposers' corr-based warp fits. The FE
    # step's ``classes_y`` are LABEL codes from the internal target quantisation
    # (NOT guaranteed ordinal/monotone in y - measured 37 unordered codes on a
    # heavy-tailed regression y), which destroys a Pearson-corr-validated ALS /
    # periodogram fit; the rank of y is monotone-equivalent to y, heavy-tail-robust,
    # and exactly as leak-safe (a fit-time supervised target; every emitted recipe
    # stays a closed-form function of x). Deleted at fit end (transient, keeps the
    # pickle slim). Non-numeric / multi-output y -> None (escalation falls back to
    # ``classes_y`` codes).
    try:
        _y_esc_arr = _y_np
        if _y_esc_arr.ndim == 1 and _y_esc_arr.dtype.kind in "fiub" and len(_y_esc_arr) == len(X):
            _y_esc_rank = np.argsort(np.argsort(_y_esc_arr, kind="stable"), kind="stable").astype(np.float64)
            self._fe_escalation_y_rank_ = _y_esc_rank / max(len(_y_esc_rank) - 1, 1)
        else:
            self._fe_escalation_y_rank_ = None
    except Exception as exc:
        logger.debug("mrmr: FE-escalation y-rank computation failed; rank unavailable this fit: %r", exc, exc_info=True)
        self._fe_escalation_y_rank_ = None

    # PREWARP ALS RECONSTRUCTION TARGET: stash the RAW CONTINUOUS y so
    # the pair-search rank-1 ALS warp reconstructs against the faithful continuous
    # target rather than the coarse equal-frequency screening codes the target-rebin
    # guard (above) produces. The guard correctly coarsens ``classes_y`` for the MI
    # screen/gates, but a least-squares f(a)*g(b) reconstruction loses fidelity on a
    # non-monotone product when fit to 10-bin codes (measured |corr| 0.97 -> 0.88).
    # Unlike the escalation rank-y this is the raw VALUES (the supervised MDLP-quality
    # signal the ALS needs; rank-y only recovered 0.88 -> 0.88 in benchmarking). Same
    # leak-safety: a fit-time supervised target whose emitted recipe stays a
    # closed-form function of x. Deleted at fit end (transient, keeps the pickle slim).
    # Non-numeric / multi-output y -> None (ALS falls back to ``classes_y`` codes).
    try:
        _y_pw_arr = _y_np
        if _y_pw_arr.ndim == 1 and _y_pw_arr.dtype.kind in "fiub" and len(_y_pw_arr) == len(X):
            self._fe_prewarp_y_continuous_ = np.ascontiguousarray(_y_pw_arr, dtype=np.float64)
        else:
            self._fe_prewarp_y_continuous_ = None
    except Exception as exc:
        logger.debug("mrmr: prewarp continuous-y stash failed; ALS reconstruction target unavailable: %r", exc, exc_info=True)
        self._fe_prewarp_y_continuous_ = None

    # ---------------------------------------------------------------------------------------------------------------
    # Temporarily inject targets
    # ---------------------------------------------------------------------------------------------------------------

    target_prefix = self._resolve_target_prefix()
    y_shape = y.shape
    if len(y_shape) == 2:
        y_shape = y_shape[1]
    else:
        y_shape = 1
    target_names = [target_prefix + "_" + str(i) for i in range(y_shape)]

    vals = _target_to_numpy_values(y)
    vals = self._coerce_target_dtype(vals)

    # Native Polars support - no `.to_pandas()` copy. Production frames are 100+ GB; full materialization
    # would OOM. Use Polars-native ops when the input is pl.DataFrame.
    try:
        import polars as pl  # local alias; safe even if pl is already imported module-scope
        _is_polars_input = isinstance(X, pl.DataFrame)
    except ImportError:
        _is_polars_input = False

    # Track the caller-visible pandas frame so the ``finally`` below can always drop the injected target columns even if
    # ``fit`` raises mid-way (e.g. categorize_dataset / screen_predictors / cat-FE step). Pre-fix code dropped only on
    # the happy path, so a raised exception left ``targ_*`` columns on the caller's frame; downstream pipelines then
    # baked them into ``feature_names_in_`` and crashed on ``transform``.
    _caller_pandas_frame = None
    if _is_polars_input:
        # Polars is immutable; with_columns returns a new frame sharing buffers with X - no data copy.
        target_series = [pl.Series(name, vals[:, i] if vals.ndim == 2 else vals) for i, name in enumerate(target_names)]
        X = X.with_columns(target_series)
    else:
        # Multilabel target (N, K): pass through unchanged so each column maps to its target_names entry.
        # Previous .reshape(-1, 1) only worked for 1-D y; crashed on multilabel with "Must have equal len keys
        # and value when setting with an ndarray".
        _caller_pandas_frame = X
        if vals.ndim == 2:
            X.loc[:, target_names] = vals
        else:
            X.loc[:, target_names] = vals.reshape(-1, 1)
        # Register cleanup with the public ``fit`` wrapper so any later raise still strips ``targ_*``.
        self._pandas_frame_for_target_cleanup = _caller_pandas_frame
        self._target_names_for_cleanup = list(target_names)

    # ---------------------------------------------------------------------------------------------------------------
    # MEM: free per-family FE intermediate frames before discretizing (PEAK-RSS bound).
    # Each of the ~50 FE families above produces a full-width intermediate DataFrame
    # (``X_t``/``X_q``/``X_te``/... - internally ``pd.concat([X, new_cols])``) and the
    # accepted subset is folded into ``X`` while the intermediate stays bound to its own
    # distinct local name. None is reused or deleted, so at this point (the peak: every
    # float frame coexists, ``categorize_dataset`` is about to allocate the int-code
    # ``data`` on top) the process holds ~one full-frame copy PER family that ran.
    # These locals are provably dead here - the only column data that must survive is in
    # ``X`` (consulted by categorize_dataset, the DCD ``X_raw=X`` path, and transform-time
    # recipe replay). Dropping them is SELECTION-NEUTRAL: ``data``/``cols``/``nbins`` and
    # every downstream MI estimate are computed from ``X`` alone, untouched by these names.
    # Names are bound only when that family ran and its helper returned (gated-off /
    # raised families never bind the name), so each drop is an explicit ``del`` guarded by
    # a membership check. For the ``X = X_<fam>`` rebind families the name is an ALIAS of
    # the live ``X`` and ``del`` only removes the alias (X survives); for the concat
    # families it frees a genuine separate full-width frame - the actual memory win.
    # NOTE: ``del locals()[name]`` does NOT free a real local in CPython; a literal ``del``
    # statement is required, hence the explicit per-name lines below.
    _fe_live = set(locals())
    if "X_h" in _fe_live: del X_h
    if "X_e" in _fe_live: del X_e
    # X_t/X_q/X_aa/X_ad/X_rt/X_df/X_cb/X_boot/X_tg/X_ksg/X_copula/X_dcor/X_hsic/X_jmim/X_tc/X_cmim/
    # X_auto/X_ens/X_meta (the 19 hybrid_orth family-variant temp frames) no longer exist as
    # _fit_impl locals at all -- their whole family block was carved into
    # _hybrid_orth_family_variants.py, where each temp frame is now genuinely function-local and
    # reclaimed automatically on that function's return; the manual del here would have been dead
    # (all these `if "X_t" in _fe_live` guards evaluate False now, and ruff's F821 on the bare `del
    # X_t` naming a name unbound anywhere in this scope confirmed it) rather than a memory-safety gap.
    if "X_mg" in _fe_live: del X_mg
    if "X_cmi" in _fe_live: del X_cmi
    if "X_te" in _fe_live: del X_te
    if "X_ba" in _fe_live: del X_ba
    if "X_c" in _fe_live: del X_c
    if "X_f" in _fe_live: del X_f
    if "X_cn" in _fe_live: del X_cn
    if "X_i" in _fe_live: del X_i
    if "X_p" in _fe_live: del X_p
    if "X_r" in _fe_live: del X_r
    if "X_lr" in _fe_live: del X_lr
    if "X_gd" in _fe_live: del X_gd
    if "X_ld" in _fe_live: del X_ld
    if "X_ga" in _fe_live: del X_ga
    if "X_cga" in _fe_live: del X_cga
    if "X_gq" in _fe_live: del X_gq
    if "X_cp" in _fe_live: del X_cp
    if "X_ct" in _fe_live: del X_ct
    if "X_nd" in _fe_live: del X_nd
    if "X_md" in _fe_live: del X_md
    if "X_rc" in _fe_live: del X_rc
    if "X_cr" in _fe_live: del X_cr
    if "X_cd" in _fe_live: del X_cd
    if "X_wv" in _fe_live: del X_wv
    if "X_rg" in _fe_live: del X_rg
    del _fe_live
    import gc as _gc
    _gc.collect()

    # ---------------------------------------------------------------------------------------------------------------
    # Discretize continuous data
    # ---------------------------------------------------------------------------------------------------------------

    logger.info("categorizing dataset...")
    # NaN handling is delegated to `categorize_dataset` via
    # `missing_strategy`. The legacy ffill/bfill path was a temporal-fill
    # workaround that injected fake signal correlated with the row's
    # neighbours; the default "separate_bin" treats NaN as an honest
    # category (its own bin per column), which an MI estimator handles
    # correctly with no special-casing on the receiving side.
    if self.nan_strategy in ("ffill_bfill",):
        # Legacy path retained for reproducibility of pre-2026-05-15 runs.
        if _is_polars_input:
            _x_for_cat = X.fill_null(strategy="forward").fill_null(strategy="backward")
        else:
            _x_for_cat = X.ffill().bfill()
        _strategy_for_categorize = "fillna_zero"  # any residual NaN -> 0 (legacy)
    else:
        _x_for_cat = X
        _strategy_for_categorize = self.nan_strategy
    # Propagate the new ``nbins_strategy`` knob through to
    # categorize_dataset so per-column adaptive bin counts (FD, QS, MDLP, Knuth,
    # OptimalJoint, ...) actually take effect inside fit(). When None,
    # categorize_dataset uses the legacy fixed ``quantization_nbins``.
    _nbins_strategy = getattr(self, "nbins_strategy", None)
    _nbins_strategy_kwargs = getattr(self, "nbins_strategy_kwargs", None)
    # When capping cardinality for the compact-codes int8 goal, also bound the NUMERIC side: the supervised MDLP
    # (fayyad_irani) recursion can emit up to 2**max_depth intervals (default max_depth=8 -> ~256), which would exceed
    # int8 just like a high-card categorical. Cap max_depth to floor(log2(cap)) so numeric bins <= cap too (unless the
    # user pinned max_depth explicitly). This makes max_categorical_cardinality a single knob for a universally-narrow
    # codes matrix - categorical tail folded AND numeric intervals bounded.
    _cap = getattr(self, "max_categorical_cardinality", None)
    if _cap and str(_nbins_strategy).lower() in ("mdlp", "fayyad_irani", "mdlp_validated", "fayyad_irani_validated"):
        _md = max(2, int(np.floor(np.log2(int(_cap)))))
        _nbins_strategy_kwargs = dict(_nbins_strategy_kwargs or {})
        _nbins_strategy_kwargs.setdefault("max_depth", _md)
    # Constructor-level shared adaptive-bin-count ceiling (knuth / bayesian_blocks / freedman_diaconis
    # - see MRMR.__init__'s max_adaptive_nbins docstring and _adaptive_nbins.MAX_ADAPTIVE_NBINS).
    # setdefault so an explicit per-method override in nbins_strategy_kwargs (e.g. "knuth_m_max_cap")
    # still wins.
    _max_adaptive_nbins = getattr(self, "max_adaptive_nbins", None)
    if _max_adaptive_nbins is not None:
        _nbins_strategy_kwargs = dict(_nbins_strategy_kwargs or {})
        _nbins_strategy_kwargs.setdefault("max_adaptive_nbins", int(_max_adaptive_nbins))
    # The supervised strategies (mdlp / optimal_joint) need y. Pull the raw
    # target column from the input frame - categorize_dataset is called with
    # _x_for_cat which is a DataFrame; the target column is one of its members
    # (target injection happens upstream in _mrmr_fit_impl).
    _y_for_strategy = None
    if _nbins_strategy is not None and str(_nbins_strategy).lower() in (
        "mdlp", "fayyad_irani", "mdlp_validated", "fayyad_irani_validated", "optimal_joint", "cv",
        "mah", "mah_sci", "sci", "marx",
    ):
        # Use the first target column as the supervised signal.
        if target_names:
            try:
                if hasattr(_x_for_cat, "to_numpy"):
                    _y_for_strategy = np.asarray(_x_for_cat[target_names[0]])
                else:
                    _y_for_strategy = np.asarray(_x_for_cat[target_names[0]])
            except Exception as exc:
                logger.debug("mrmr: y coercion for discretization-strategy selection failed: %r", exc, exc_info=True)
                _y_for_strategy = None
    data, cols, nbins = categorize_dataset(
        df=_x_for_cat,
        method=self.quantization_method,
        n_bins=self.quantization_nbins,
        dtype=self.quantization_dtype,
        max_categorical_cardinality=getattr(self, "max_categorical_cardinality", None),
        missing_strategy=_strategy_for_categorize,
        nbins_strategy=_nbins_strategy,
        nbins_strategy_kwargs=_nbins_strategy_kwargs,
        y_for_strategy=_y_for_strategy,
        cache_dir=getattr(self, "cache_dir", None),
    )
    logger.info("categorized.")

    # 2026-07-11 perf: speculatively pre-warm the polynom-pair-FE loky pool here, AFTER categorization (not
    # before it). ``run_polynom_pair_fe`` is otherwise the pool's first user in a typical fit (the sibling CPU
    # pair-MI-sweep pool in ``_step_pairmi.py`` only engages when the GPU MI path fails), so production pays a
    # full cold-start (16 fresh worker processes each re-importing mlframe/numba/cupy, measured 28.1s cold vs
    # 0.7s warm) synchronously inside the polynom-pair-FE phase. Placed HERE, not at fit-entry: an earlier
    # placement (right after ``fe_smart_polynom_iters``/``n_jobs`` are read, well before categorization) was
    # measured to REGRESS a full 100k-row production run by ~223s wall-clock - categorization is itself
    # CPU-active (not idle), so the pre-warm's 16 concurrent worker-process spawns contended with it (round-12
    # A/B: the categorization gap grew 85.3s -> 153.1s, with exactly 16 new
    # "NumbaPerformanceWarning: Grid size 1" lines appearing in that window, matching n_jobs=16 workers
    # bootstrapping). The GPU pair-MI screening dispatched right after this point is genuinely GPU-bound
    # (blocks on ``.get()``/``copy_to_host``), leaving CPU free for the pre-warm to actually overlap with idle
    # time instead of stealing it. See ``maybe_prewarm_polynom_loky_pool``'s docstring for the pool-reuse
    # mechanics and the ``idle_worker_timeout`` fix (the pool must survive until ``run_polynom_pair_fe`` uses
    # it, several minutes later).
    from .._joblib_safe import maybe_prewarm_polynom_loky_pool

    maybe_prewarm_polynom_loky_pool(fe_smart_polynom_iters, n_jobs)

    # ``cols`` is a list; per-name ``cols.index`` is an O(len(cols)) scan, so resolving every target /
    # categorical name that way is O(C*P). Build a name->index map once and reuse it for both lookups.
    _name_to_idx = {c: i for i, c in enumerate(cols)}

    target_indices = np.array([_name_to_idx[col] for col in target_names], dtype=np.int64)

    # TARGET REBIN GUARD. The adaptive per-column ``nbins_strategy``
    # (default ``"mdlp"`` since Wave 7) is meant for FEATURE columns; applied to the
    # injected TARGET column it is SELF-REFERENTIAL (MDLP bins y supervised on y) and
    # on a heavy-tailed continuous y it produces a DEGENERATE encoding - measured on
    # the F2 fixture (y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3), n=20000): 37 bins with
    # 83.7% of all rows collapsed into ONE bin (vs the clean 10 x 2000 equal-frequency
    # legacy quantile bins). Every downstream MI/CMI - screening, pair gates, FE
    # acceptance - is computed AGAINST these target codes, so the bulk of the signal
    # becomes invisible (the genuine (c,d) term's measured CMI drops ~6x). Re-bin the
    # CONTINUOUS target columns (raw unique count > quantization_nbins; classification
    # labels are left untouched) with the legacy ``quantization_method`` /
    # ``quantization_nbins`` equal-frequency quantile path. No-op when
    # ``nbins_strategy`` is None (legacy fits already bin the target this way).
    if _nbins_strategy is not None and len(target_indices) > 0:
        from ..discretization import discretize_array as _t_discretize
        for _ti in target_indices:
            _t_name = cols[int(_ti)]
            try:
                _t_raw = np.asarray(_x_for_cat[_t_name].to_numpy() if hasattr(_x_for_cat[_t_name], "to_numpy") else _x_for_cat[_t_name])
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("mrmr: extracting raw values for column %r during cat-handling failed: %r", _t_name, e, exc_info=True)
                continue
            if _t_raw.dtype.kind not in "fiub" or _t_raw.ndim != 1:
                continue
            _t_finite = _t_raw[np.isfinite(_t_raw.astype(np.float64))] if _t_raw.dtype.kind == "f" else _t_raw
            if np.unique(_t_finite).size <= int(self.quantization_nbins):
                continue  # discrete / classification target: keep its native classes
            _t_codes = _t_discretize(
                arr=_t_raw.astype(np.float64),
                n_bins=int(self.quantization_nbins),
                method=str(self.quantization_method),
                dtype=self.quantization_dtype,
            )
            _t_nb = int(np.max(_t_codes)) + 1
            if _t_nb >= 2 and (int(nbins[int(_ti)]) != _t_nb or not np.array_equal(data[:, int(_ti)], _t_codes)):
                if verbose:
                    logger.info(
                        "MRMR.fit target-rebin guard: target %r re-binned from the adaptive "
                        "nbins_strategy=%r encoding (%d bins, max-bin %.1f%%) to the legacy "
                        "%s/%d equal-frequency codes (%d bins) -- the adaptive strategy is "
                        "feature-side only; on the target it degrades MI sensitivity.",
                        _t_name, str(_nbins_strategy), int(nbins[int(_ti)]),
                        100.0 * float(np.bincount(data[:, int(_ti)].astype(np.int64)).max()) / max(1, data.shape[0]),
                        str(self.quantization_method), int(self.quantization_nbins), _t_nb,
                    )
                data[:, int(_ti)] = _t_codes
                nbins[int(_ti)] = _t_nb

    # COMPACT CODES STORAGE. ``data`` holds per-column BIN INDICES (0..nbins-1 + a NaN bin / -1 sentinel), never JOINT
    # ids, so it fits the smallest int that spans its actual code range - int8 for the common nbins<=~127 case, int16
    # for a high-cardinality categorical. The base (n, p) matrix at scale (e.g. 795k x 496) drops 4x / 2x vs the legacy
    # int32. Selection-EQUIVALENT: the code VALUES are unchanged, and every consumer (merge_vars, the GPU path) reads
    # this storage and casts UP to int32 for JOINT math, so deep joints (nbins^order) never overflow. Engineered-code
    # appends downstream re-narrow to this dtype (``_append_codes``). Range-checked directly (one min/max pass) rather
    # than trusting nbins semantics. Opt out: MLFRAME_MRMR_COMPACT_CODES=0.
    if data.size and os.environ.get("MLFRAME_MRMR_COMPACT_CODES", "1").strip().lower() not in ("0", "false", "off", "no"):
        try:
            _dmin = int(data.min()); _dmax = int(data.max())
            _store_dt: Optional[type]
            if -128 <= _dmin and _dmax <= 127:
                _store_dt = np.int8
            elif -32768 <= _dmin and _dmax <= 32767:
                _store_dt = np.int16
            else:
                _store_dt = None
            if _store_dt is not None and data.dtype.itemsize > np.dtype(_store_dt).itemsize:
                data = data.astype(_store_dt, copy=False)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("mrmr: narrowing stored codes dtype failed: %r", e, exc_info=True)
            pass

    # ---------------------------------------------------------------------------------------------------------------
    # Core
    # ---------------------------------------------------------------------------------------------------------------

    if _is_polars_input:
        # Polars schema-driven detection; mirrors categorize_dataset's _is_pl_cat.
        import polars as _pl
        _CAT_DTYPES_FOR_VARS = {_pl.Utf8, _pl.String, _pl.Categorical, _pl.Boolean}
        categorical_vars_names = [name for name, dt in X.schema.items() if dt in _CAT_DTYPES_FOR_VARS or (hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum))]
    else:
        categorical_vars_names = X.head().select_dtypes(include=("category", "object", "string", "bool")).columns.values.tolist()
    categorical_vars = [_name_to_idx[col] for col in categorical_vars_names]

    if fe_max_steps > 0:
        unary_transformations = create_unary_transformations(preset=fe_unary_preset)
        binary_transformations = create_binary_transformations(preset=fe_binary_preset)
        # REPLAY-SAFETY (audit, 2026-06-13): exclude ops that are NOT row-wise pure functions from FE
        # pair candidates. Their value at a row depends on OTHER rows (``np.gradient``: grad1/grad2) or
        # on a whole-column statistic recomputed at apply time (``logn`` uses ``x - np.min(x)``), so a
        # recipe built on them silently produces DIFFERENT values on a row-slice / test frame
        # (slice-replay corruption - the same class as the smart_log BUG2 fix). They appear only in the
        # non-default "maximal" preset; dropping them here means they are never selected as engineered
        # features, while the create_*_transformations registry stays intact (other callers + the
        # registry-coverage test are unaffected). On the default "minimal" preset this is a no-op.
        _FE_NON_ROWWISE_PURE = ("grad1", "grad2", "logn")
        unary_transformations = {k: v for k, v in unary_transformations.items() if k not in _FE_NON_ROWWISE_PURE}
        binary_transformations = {k: v for k, v in binary_transformations.items() if k not in _FE_NON_ROWWISE_PURE}
        if fe_max_polynoms:
            # Generated polynomial coefficients are appended directly to unary_transformations under "poly_<coef>" keys;
            # no separate registry is needed. Use a seeded local Generator so the polynomial recipes are reproducible
            # across reruns with the same ``random_seed`` - prior code used the global ``np.random`` stream, breaking
            # determinism whenever any earlier suite stage advanced it.
            _poly_rng = np.random.default_rng(self.random_seed)
            for _ in range(fe_max_polynoms):
                length = int(_poly_rng.integers(3, 9))
                coef = np.empty(shape=length, dtype=np.float32)
                for i in range(length):
                    coef[i] = _poly_rng.normal((1.0 if i == 1 else 0.0), scale=0.05)

                unary_transformations["poly_" + str(coef)] = coef

        if verbose > 2:
            logger.info("nunary_transformations: %s", f"{len(unary_transformations):_}")
            logger.info("nbinary_transformations: %s", f"{len(binary_transformations):_}")

        engineered_features: set = set()
        checked_pairs: set = set()
    # engineered_recipes (name -> EngineeredRecipe) is initialised unconditionally; the splitter at the bottom
    # of fit() looks it up regardless of fe_max_steps. Stays empty when FE is disabled.
    engineered_recipes: dict = {}
    # PER-GATE FE REJECTION LEDGER (additive, 2026-06-11): the per-fit raw-record list is reset
    # near fit-start (above, before any FE stage records) so it accumulates the gate drops of
    # EVERY FE stage this fit - the recipe-FE families + cluster-basis (which record before this
    # point) AND the pair-search ``_run_fe_step`` loop below. fe_rejection_ledger_ is built from
    # it at fit-end. Stays empty when FE produced no rejected candidates.
    # Layer 23: seed engineered_recipes with hybrid orthogonal-poly recipes
    # built above (before the screening loop). The end-of-fit remap routes
    # any selected_vars_name matching a key here into _engineered_recipes_.
    if _hybrid_orth_pre_recipes:
        engineered_recipes.update(_hybrid_orth_pre_recipes)
    # Layer 26: same routing pattern for MI-greedy recipes.
    if _mi_greedy_pre_recipes:
        engineered_recipes.update(_mi_greedy_pre_recipes)
    # Layer 33: same routing pattern for K-fold target-encoded recipes.
    if _kfold_te_pre_recipes:
        engineered_recipes.update(_kfold_te_pre_recipes)
    if _binned_agg_pre_recipes:
        engineered_recipes.update(_binned_agg_pre_recipes)
    # Layer 34: same routing for count / frequency / cat_num residual recipes.
    if _count_enc_pre_recipes:
        engineered_recipes.update(_count_enc_pre_recipes)
    if _freq_enc_pre_recipes:
        engineered_recipes.update(_freq_enc_pre_recipes)
    if _cat_num_pre_recipes:
        engineered_recipes.update(_cat_num_pre_recipes)
    # Layer 37: same routing for missingness indicator / count / pattern recipes.
    if _miss_ind_pre_recipes:
        engineered_recipes.update(_miss_ind_pre_recipes)
    if _miss_cnt_pre_recipes:
        engineered_recipes.update(_miss_cnt_pre_recipes)
    if _miss_pat_pre_recipes:
        engineered_recipes.update(_miss_pat_pre_recipes)
    # Layer 38: same routing for ratio / log_ratio / grouped_delta / lagged_diff.
    if _ratio_pre_recipes:
        engineered_recipes.update(_ratio_pre_recipes)
    if _log_ratio_pre_recipes:
        engineered_recipes.update(_log_ratio_pre_recipes)
    if _grouped_delta_pre_recipes:
        engineered_recipes.update(_grouped_delta_pre_recipes)
    if _lagged_diff_pre_recipes:
        engineered_recipes.update(_lagged_diff_pre_recipes)
    # Layer 87: same routing for grouped multi-stat aggregate recipes.
    if _grouped_agg_pre_recipes:
        engineered_recipes.update(_grouped_agg_pre_recipes)
    # Layer 93: same routing for composite-key grouped aggregate recipes.
    if _composite_group_agg_pre_recipes:
        engineered_recipes.update(_composite_group_agg_pre_recipes)
    # Layer 88: same routing for grouped-quantile / target-aware-bin recipes.
    if _grouped_quantile_pre_recipes:
        engineered_recipes.update(_grouped_quantile_pre_recipes)
    # Layer 89: same routing for cat x cat synergy-cross recipes.
    if _cat_pair_pre_recipes:
        engineered_recipes.update(_cat_pair_pre_recipes)
    # Layer 94: same routing for cat x cat x cat triple synergy-cross recipes.
    if _cat_triple_pre_recipes:
        engineered_recipes.update(_cat_triple_pre_recipes)
    if _numeric_decompose_pre_recipes:
        engineered_recipes.update(_numeric_decompose_pre_recipes)
    # Layer 95 PART A: same routing for periodic / modular recipes.
    if _modular_pre_recipes:
        engineered_recipes.update(_modular_pre_recipes)
    if _pairwise_modular_pre_recipes:
        engineered_recipes.update(_pairwise_modular_pre_recipes)
    if _integer_lattice_pre_recipes:
        engineered_recipes.update(_integer_lattice_pre_recipes)
    if _row_argmax_pre_recipes:
        engineered_recipes.update(_row_argmax_pre_recipes)
    if _conditional_gate_pre_recipes:
        engineered_recipes.update(_conditional_gate_pre_recipes)
    # Layer 95 PART B: same routing for per-group distribution-distance recipes.
    if _group_distance_pre_recipes:
        engineered_recipes.update(_group_distance_pre_recipes)
    # Layer 104: rare-category / conditional-residual / rankgauss recipes.
    if _rare_category_pre_recipes:
        engineered_recipes.update(_rare_category_pre_recipes)
    if _conditional_residual_pre_recipes:
        engineered_recipes.update(_conditional_residual_pre_recipes)
    if _conditional_dispersion_pre_recipes:
        engineered_recipes.update(_conditional_dispersion_pre_recipes)
    if _conditional_quantile_rank_pre_recipes:
        engineered_recipes.update(_conditional_quantile_rank_pre_recipes)
    if _ordinal_pattern_pre_recipes:
        engineered_recipes.update(_ordinal_pattern_pre_recipes)
    if _random_fourier_pre_recipes:
        engineered_recipes.update(_random_fourier_pre_recipes)
    if _sir_direction_pre_recipes:
        engineered_recipes.update(_sir_direction_pre_recipes)
    if _lof_pre_recipes:
        engineered_recipes.update(_lof_pre_recipes)
    if _mahalanobis_density_pre_recipes:
        engineered_recipes.update(_mahalanobis_density_pre_recipes)
    if _wavelet_pre_recipes:
        engineered_recipes.update(_wavelet_pre_recipes)
    if _rankgauss_pre_recipes:
        engineered_recipes.update(_rankgauss_pre_recipes)
    # Layer 92: same routing for temporal leak-safe aggregation recipes.
    if _temporal_agg_pre_recipes:
        engineered_recipes.update(_temporal_agg_pre_recipes)
    # Reset per fit so a re-fit on the same instance doesn't carry stale cluster-aggregate state.
    self._cluster_aggregate_removals_ = []
    self.cluster_aggregate_ = []  # fitted summary (per-aggregate records) -> meta_info report

    # Cat-FE step (categorical interaction generator). Runs once before the screening loop when
    # ``cat_fe_config.enable=True``; augments data/cols/nbins with ordinal-encoded columns capturing pair
    # (and future k-way) synergies. Engineered cols enter screening as atomic 1-way features.
    cat_fe_cfg = getattr(self, "cat_fe_config", None)
    self._cat_fe_state_ = None
    # ``None`` means "use default CatFEConfig()" which has enable=True. Pass CatFEConfig(enable=False) for legacy.
    if cat_fe_cfg is None:
        from ..cat_fe_state import CatFEConfig as _CatFEConfig
        cat_fe_cfg = _CatFEConfig()
    # include_numeric: collect raw numeric feature values (keyed by data-column index) so the cat-FE step can
    # quantile-bin them into the candidate pool. Extracted from the ORIGINAL ``X`` (NaN visible) - NOT the
    # ffill'd ``_x_for_cat`` - so a NaN-bearing column is correctly skipped at fit (v1 has no NaN bin in the
    # quantile-edge replay) and fit/transform stay consistent (both read the user's raw frame).
    _num_raw_values = None
    if cat_fe_cfg.enable and getattr(cat_fe_cfg, "include_numeric", False):
        from ..engineered_recipes._recipe_extract import _extract_column as _extract_col_for_num
        _cat_idx_set = set(int(c) for c in categorical_vars)
        _tgt_idx_set = set(int(t) for t in target_indices)
        # RAW input columns only: pre-FE recipes (haar / ratio / grouped-agg ...) appended engineered numeric
        # columns to data / cols / X before this step. Crossing those is unreplayable - the engineered source
        # is absent from the user's raw frame at transform time -> NaN column / silent feature drop. Restrict to
        # ``feature_names_in_`` (the raw user columns, set above, excludes engineered names).
        # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
        _fni_raw = getattr(self, "feature_names_in_", None)
        _raw_name_set = set(_fni_raw) if _fni_raw is not None else set()
        _num_raw_values = {}
        for _ci in range(len(cols)):
            if _ci in _cat_idx_set or _ci in _tgt_idx_set:
                continue
            if _raw_name_set and cols[_ci] not in _raw_name_set:
                continue
            # Skip columns the user supplied with NaN (snapshot at fit entry, robust to any downstream impute):
            # the quantile-edge replay has no NaN bin, so crossing them would skew serving.
            if cols[_ci] in _include_numeric_input_nan_cols:
                continue
            try:
                _num_raw_values[_ci] = np.asarray(_extract_col_for_num(X, cols[_ci]))
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("mrmr: extracting raw numeric column %r values failed: %r", cols[_ci], e, exc_info=True)
                continue
    _cat_fe_pool_size = len(categorical_vars) + (len(_num_raw_values) if _num_raw_values else 0)
    if cat_fe_cfg.enable and _cat_fe_pool_size >= 2:
        from ..cat_interactions import run_cat_interaction_step
        from ..info_theory import merge_vars as _merge_vars_for_cat_fe

        # Pre-compute classes_y / freqs_y for cat-FE (avoids re-binning the target inside every kernel call).
        _classes_y, _freqs_y, _ = _merge_vars_for_cat_fe(
            factors_data=data, vars_indices=target_indices,
            var_is_nominal=None, factors_nbins=nbins, dtype=dtype,
        )
        _classes_y_safe = _classes_y.copy()

        # Pull cached cat-FE state from prior fit (if any).
        _prev_cache = getattr(self, "_cat_fe_cache_", None)
        _n_cols_before_cat_fe = data.shape[1]
        data, cols, nbins, cat_fe_state = run_cat_interaction_step(
            data=data, cols=cols, nbins=nbins,
            target_indices=target_indices,
            classes_y=_classes_y, classes_y_safe=_classes_y_safe,
            freqs_y=_freqs_y,
            categorical_vars=categorical_vars,
            cfg=cat_fe_cfg,
            streaming_cache=_prev_cache,
            numeric_raw_values=_num_raw_values,
            dtype=dtype, verbose=verbose,
        )
        self._cat_fe_state_ = cat_fe_state
        # Register engineered cat features as categorical_vars so the downstream numeric-FE step excludes them
        # from numeric_vars_to_consider; without this, k-way cat engineered cols enter prospective_pairs and
        # check_prospective_fe_pairs hits KeyError reading them from X (which lacks engineered cols).
        # Engineered cat cols are appended at the end of data/cols at positions [_n_cols_before_cat_fe..].
        _n_cat_fe_added = data.shape[1] - _n_cols_before_cat_fe
        if _n_cat_fe_added > 0:
            categorical_vars = list(categorical_vars) + list(range(_n_cols_before_cat_fe, data.shape[1]))
        # Persist cache for next fit() call
        if cat_fe_state.streaming_cache_out:
            self._cat_fe_cache_ = cat_fe_state.streaming_cache_out
        # Stamp the fit-time categorical -> integer-code mapping onto every cat-FE recipe whose source columns are
        # categorical / string. Without this, ``transform`` on a raw frame routes string source values through
        # ``astype(int64)`` -> ValueError -> all-zero codes, so the carefully-discovered cat-interaction (factorize /
        # target_encoding) feature collapses to a CONSTANT column at serving time - a silent train/serve skew (the
        # FS-side analog of the 4b299e25 neural ``_apply_cat_codes`` bug). ``categorize_dataset`` codes Categorical via
        # ``.cat.codes`` (category order) and object/string via ``pd.factorize`` (first-appearance order, training-data
        # dependent); only a stored map can reproduce those codes at transform. The map is built ONCE per distinct source
        # column from the raw ``_x_for_cat`` frame and shared across recipes referencing that column.
        if cat_fe_state.recipes and not _is_polars_input and hasattr(_x_for_cat, "columns"):
            from ..engineered_recipes._recipe_extract import build_category_code_map as _build_cat_code_map
            # ``categorize_dataset`` factorises ALL categorical columns as ONE block and applies the NaN +1
            # shift to the WHOLE block when ANY column in it has a NaN. So even a NaN-FREE categorical source
            # gets its codes shifted +1 at fit time. Compute the block-level NaN flag ONCE (mirroring
            # ``categorize_dataset``'s ``select_dtypes`` block selection exactly) and thread it into every map
            # build; a per-column flag would off-by-one the NaN-free partner of a NaN-bearing column - the
            # same silent train/serve skew, for the mixed-block case the per-column path never handled.
            _block_has_nan: bool | None = None
            try:
                _cat_block = _x_for_cat.select_dtypes(include=("category", "object", "string", "bool"))
                if _cat_block.shape[1] > 0:
                    _block_has_nan = bool(_cat_block.isna().to_numpy().any())
            except Exception as exc:
                logger.debug("mrmr: NaN-block detection failed; treating as unknown: %r", exc, exc_info=True)
                _block_has_nan = None
            _src_map_cache: dict = {}
            for _ri, r in enumerate(cat_fe_state.recipes):
                _maps_for_recipe: dict = {}
                for _src in getattr(r, "src_names", ()) or ():
                    if _src not in _src_map_cache:
                        if _src in _x_for_cat.columns:
                            try:
                                _src_map_cache[_src] = _build_cat_code_map(_x_for_cat[_src], block_has_nan=_block_has_nan)
                            except Exception as exc:
                                logger.debug("mrmr: source-map cache build failed for this recipe source; treating as empty: %r", exc, exc_info=True)
                                _src_map_cache[_src] = {}
                        else:
                            _src_map_cache[_src] = {}
                    if _src_map_cache[_src]:
                        _maps_for_recipe[_src] = _src_map_cache[_src]
                if _maps_for_recipe:
                    # ``extra`` is a read-only MappingProxyType on a frozen recipe; ``with_extra`` returns a fresh copy carrying the maps.
                    try:
                        cat_fe_state.recipes[_ri] = r.with_extra(cat_code_maps=_maps_for_recipe)
                    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                        logger.debug("mrmr: attaching cat_code_maps to recipe %r failed: %r", getattr(r, "name", "?"), e, exc_info=True)
                        pass
        # Cat-FE recipes feed the same engineered_recipes dict numeric FE uses; the fit-end splitter copies
        # any recipe whose engineered name appears in selected_vars_names into ``self._engineered_recipes_``.
        for r in cat_fe_state.recipes:
            engineered_recipes[r.name] = r
        if verbose and cat_fe_state.recipes:
            logger.info(
                "MRMR cat-FE produced %d engineered feature(s); " "data extended from %d to %d cols.",
                len(cat_fe_state.recipes),
                data.shape[1] - len(cat_fe_state.recipes),
                data.shape[1],
            )

    # Resolve effective ``min_relevance_gain`` against the target entropy. ``'relative_to_entropy'`` mode uses ``min_relevance_gain_frac * H(y)`` so the stop floor scales with how much information the target actually carries; ``'absolute'`` mode retains the legacy verbatim value. The target is already discretized into bins (``data[:, target_indices[0]]`` with bin count ``nbins[target_indices[0]]``); ``np.bincount`` + Shannon entropy in nats matches the screen_predictors estimator family.
    if self.min_relevance_gain_mode not in ("absolute", "relative_to_entropy"):
        raise ValueError(f"MRMR.min_relevance_gain_mode={self.min_relevance_gain_mode!r} must be 'absolute' or 'relative_to_entropy'.")
    if self.min_relevance_gain_mode == "relative_to_entropy":
        _target_col_idx = int(target_indices[0])
        _y_bins = data[:, _target_col_idx]
        _y_nbins = int(nbins[_target_col_idx])
        _y_counts = np.bincount(_y_bins, minlength=_y_nbins).astype(np.float64)
        _y_total = float(_y_counts.sum())
        if _y_total > 0:
            _p = _y_counts[_y_counts > 0] / _y_total
            _h_y_nats = float(-(_p * np.log(_p)).sum())
        else:
            _h_y_nats = 0.0
        _effective_min_relevance_gain = float(self.min_relevance_gain_frac) * _h_y_nats
        if verbose:
            logger.info(
                "MRMR min_relevance_gain resolution: mode=relative_to_entropy, H(y)=%.4f nats, frac=%.4g, effective floor=%.6g (legacy absolute would have been %.6g).",
                _h_y_nats, self.min_relevance_gain_frac, _effective_min_relevance_gain, self.min_relevance_gain,
            )
    else:
        _effective_min_relevance_gain = float(self.min_relevance_gain)

    num_fs_steps = 0
    # Tracks whether the post-FE confirming re-screen has run, so it
    # fires at most once (see the fe_reselect_after_engineering block below). The
    # re-screen re-selects from the augmented pool (raw + engineered) using the
    # estimator's own use_simple_mode (now defaulting to False = full Fleuret
    # conditional-MI redundancy), which is what drops engineered columns redundant
    # given an already-selected one and records a real gain for every survivor.
    _did_confirm_rescreen = False
    # Carries the DCDState from the prior screen pass into the post-FE
    # confirm-rescreen so cluster discovery (anchor graph, pruned mask,
    # swap_log) accumulates instead of being rebuilt empty each iteration.
    _persisted_dcd_state = None
    # Carries the prior round's relevance/redundancy caches into the next screen_predictors() call
    # - see screen_predictors' ``seed_caches`` docstring. Mirrors the DCD-state
    # threading immediately above; before this fix each round rebuilt all 4 caches from scratch, fully
    # rescoring every raw column's relevance/entropy/conditional-MI even though those values cannot
    # change round-to-round (the data they're computed from is stable; only new columns get appended).
    _persisted_screen_caches = None
    # Cross-round cache for the maxT permutation-null gain floor (2026-07-09 fix; see
    # compute_fdr_gain_floor's ``maxt_floor_cache`` docstring) - a plain dict, mutated in place by
    # every screen_predictors() call this fit, so a raw-pool floor computed in round 1 is not
    # recomputed identically in round 2/3.
    _persisted_maxt_floor_cache: dict = {}
    # Carries the warmed joblib worker pool (n_workers>1 only) into the next screen_predictors() call
    # - see screen_predictors' ``seed_workers_pool`` docstring. ``None`` at n_workers<=1
    # (no pool built) or before round 1.
    _persisted_workers_pool = None
    # Declared BEFORE the loop so per-binary-func timing accumulates across ALL
    # screen/FE rounds of this fit, not just the last one - previously reset to empty at the top of
    # every iteration, so the end-of-fit log only ever reflected the FINAL round's timing even though
    # this loop typically runs 2-3 rounds per fit (raw screen, FE step(s), confirm-rescreen).
    times_spent: defaultdict = defaultdict(float)
    while True:
        n_recommended_features = 0
        # Resolve the fit's ONE shared row draw BEFORE the screen so the order-1 relevance sweep + FDR
        # floor score on it (screen is the first consumer -> caches the draw -> the FE step reuses the
        # SAME rows). None at small n -> full-n screen, unchanged.
        try:
            from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
            _screen_shared_idx = _get_shared_fe_subsample_idx(self, np.asarray(data[:, int(target_indices[0])]), len(data))
        except Exception as _sub_exc:
            # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
            log_throttle(logger, "mrmr_fit_shared_fe_subsample_failed", logging.WARNING, "mrmr: shared FE subsample resolution failed; screening at FULL n: %r", _sub_exc, exc_info=True)
            _screen_shared_idx = None
        (
            selected_vars,
            predictors,
            any_influencing,
            entropy_cache,
            cached_MIs,
            cached_confident_MIs,
            cached_cond_MIs,
            classes_y,
            classes_y_safe,
            freqs_y,
            _dcd_state,
            _persisted_workers_pool,
        ) = screen_predictors(
            factors_data=data,
            y=target_indices,  # type: ignore[arg-type]
            subsample_idx=_screen_shared_idx,
            factors_nbins=nbins,
            factors_names=cols,
            # Layer 23: when hybrid orth FE appended columns, extend the
            # candidate pool to include them so they reach the screening
            # gates. When the caller did not pin factors_names_to_use,
            # screen_predictors uses every column from ``cols`` so the
            # hybrid cols are naturally included.
            factors_names_to_use=(
                list(self.factors_names_to_use) + list(self.hybrid_orth_features_ or []) + list(getattr(self, "mi_greedy_features_", None) or [])
                if (self.factors_names_to_use and (self.hybrid_orth_features_ or getattr(self, "mi_greedy_features_", None)))
                else self.factors_names_to_use
            ),
            factors_to_use=self.factors_to_use,
            # algorithm
            mrmr_relevance_algo=self.mrmr_relevance_algo,
            mrmr_redundancy_algo=self.mrmr_redundancy_algo,
            reduce_gain_on_subelement_chosen=self.reduce_gain_on_subelement_chosen,
            use_simple_mode=self.use_simple_mode,
            # performance
            extra_x_shuffling=self.extra_x_shuffling,
            dtype=self.dtype,
            random_seed=self.random_seed,
            use_gpu=self.use_gpu,
            n_workers=self.n_workers,
            # confidence
            min_occupancy=self.min_occupancy,
            min_nonzero_confidence=self.min_nonzero_confidence,
            full_npermutations=self.full_npermutations,
            baseline_npermutations=self.baseline_npermutations,
            # 2026-06-02 RC2 — sample-size-aware Fleuret confirmation threshold.
            fe_confirm_undersample_rows_per_cell=float(getattr(self, "fe_confirm_undersample_rows_per_cell", 5.0) or 0.0),
            # stopping conditions
            min_relevance_gain=_effective_min_relevance_gain,
            min_relevance_gain_relative_to_first=float(getattr(self, "min_relevance_gain_relative_to_first", 0.0)),
            cardinality_bias_correction=bool(getattr(self, "cardinality_bias_correction", True)),
            max_consec_unconfirmed=self.max_consec_unconfirmed,
            max_runtime_mins=self.max_runtime_mins,
            interactions_min_order=self.interactions_min_order,
            interactions_max_order=self.interactions_max_order,
            interactions_order_reversed=self.interactions_order_reversed,
            max_veteranes_interactions_order=self.max_veteranes_interactions_order,
            only_unknown_interactions=self.only_unknown_interactions,
            # Resolve effective max_confirmation_cand_nbins: user-pinned wins, else formula default.
            max_confirmation_cand_nbins=(
                self.max_confirmation_cand_nbins if self.max_confirmation_cand_nbins is not None else self.quantization_nbins**self.interactions_max_order * 2
            ),
            # FE-on-empty-screen fallback flag (consumed by MRMR.fit).
            fe_fallback_to_all=self.fe_fallback_to_all,
            # verbosity and formatting
            verbose=self.verbose,
            ndigits=self.ndigits,
            parallel_kwargs=self._effective_parallel_kwargs(),
            stop_file=self.stop_file,
            # engineered_lineage from cat-FE step (None when cat-FE didn't run); screen uses it to skip
            # redundant (orig_parent, engineered_col) k-way candidates.
            engineered_lineage=(self._cat_fe_state_.lineage if getattr(self, "_cat_fe_state_", None) is not None and self._cat_fe_state_.lineage else None),
            # 2026-05-30 Wave 9 — DCD config forward. Built only when
            # ``dcd_enable=True`` (per Critic1/F: passed as kwargs, NOT
            # via thread-local, for joblib parallel-backend safety).
            dcd_config=(
                dict(
                    enable=True,
                    tau_cluster=self.dcd_tau_cluster,
                    distance=self.dcd_distance,
                    cluster_size_threshold=self.dcd_cluster_size_threshold,
                    swap_gain_threshold=self.dcd_swap_gain_threshold,
                    swap_method=self.dcd_swap_method,
                    pairwise_cache_max=self.dcd_pairwise_cache_max,
                    min_cluster_size=self.dcd_min_cluster_size,
                    max_cluster_size=self.dcd_max_cluster_size,
                    swap_alpha=self.dcd_swap_alpha,
                    # 2026-06-03 (audit dcd-core-1/dcd-swap-null-1/2):
                    # the swap null draw count, decoupled from
                    # full_npermutations. getattr fallback keeps old
                    # pickles (lacking the attr) loading at the 199 default.
                    swap_npermutations=getattr(self, "dcd_swap_npermutations", 199),
                    warp_tiebreak_prefer_linear=getattr(self, "warp_tiebreak_prefer_linear", True),
                    warp_twin_rank_corr=getattr(self, "warp_twin_rank_corr", 0.99),
                    warp_linear_margin=getattr(self, "warp_linear_margin", 0.05),
                    # Layer 47: forward the auto-tau
                    # calibration knobs (number of sampled feature pairs
                    # and RNG seed) so make_dcd_state can fingerprint
                    # the calibration sweep deterministically.
                    tau_calibration_n_pairs=getattr(
                        self,
                        "dcd_tau_calibration_n_pairs",
                        100,
                    ),
                    tau_calibration_seed=getattr(
                        self,
                        "dcd_tau_calibration_seed",
                        0,
                    ),
                    X_raw=X,
                    quantization_method=self.quantization_method,
                    quantization_nbins=self.quantization_nbins,
                    quantization_dtype=self.quantization_dtype,
                )
                if getattr(self, "dcd_enable", False)
                else None
            ),
            # 2026-05-31 Layer 43 (PART A) — thread the local
            # engineered_recipes dict into screen so DCD's commit_swap can
            # register the PC1 aggregate as a replayable EngineeredRecipe.
            # Pre-fix the dict was inaccessible from screen and the swap
            # silently dropped the aggregate from ``_engineered_recipes_``.
            engineered_recipes=engineered_recipes,
            # 2026-06-02 — directed-FE tie-break: pass the snapshot of the
            # ORIGINAL user input columns (taken before any FE stage appended
            # engineered intermediates). screen_predictors uses it to mark
            # any candidate whose name is not in this set as engineered and,
            # on a near-tie in selection gain, prefer the engineered transform
            # over its raw parent (e.g. x1__He2 over x1 for an even-symmetric
            # target). Applies in BOTH the first screen and the post-FE
            # confirming re-screen (this same call runs in the while-loop).
            raw_feature_names=_raw_input_cols_pre_fe,
            # Thread the prior pass's DCDState so cluster discovery
            # accumulates across the confirm-rescreen (the matrix only
            # grows; raw indices are stable). Without this the rescreen
            # rebuilds an empty state and the published dcd_ summary loses
            # the screen-1 dup cluster (n_pruned/cluster_anchors reset).
            existing_dcd_state=_persisted_dcd_state,
            seed_caches=_persisted_screen_caches,
            seed_maxt_floor_cache=_persisted_maxt_floor_cache,
            seed_workers_pool=_persisted_workers_pool,
        )
        if _dcd_state is not None:
            _persisted_dcd_state = _dcd_state
        _persisted_screen_caches = (entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs)
        # 2026-05-30 Wave 9 — stash DCD summary on the estimator for the
        # public ``dcd_`` attribute (None when DCD was disabled).
        try:
            from .._dynamic_cluster_discovery import dcd_summary as _dcd_summary
            self.dcd_ = _dcd_summary(_dcd_state)
        except Exception as exc:
            logger.debug("mrmr: DCD result attachment failed; dcd_ unavailable: %r", exc, exc_info=True)
            self.dcd_ = None
        # Layer 41: self-describing cluster membership accessor.
        # Mirror ``dcd_["cluster_anchors_names"]`` onto the estimator as a
        # first-class fitted attribute so downstream code can read the
        # discovered clusters without indexing through ``self.dcd_`` (the
        # raw summary dict). ``cluster_members_`` is None when DCD was
        # disabled, matching ``dcd_`` semantics. Pure additive metadata -
        # no effect on ``support_`` or ``transform`` output.
        if isinstance(self.dcd_, dict):
            self.cluster_members_ = dict(self.dcd_.get("cluster_anchors_names", {}))
        else:
            self.cluster_members_ = None
        # Layer 48: hierarchical post-hoc cluster map. Pure
        # additive analyser over ``dcd_["cluster_anchors_names"]`` -
        # surfaces super-cluster ties DCD's greedy single-anchor rule
        # cannot. Empty dict when DCD found <2 anchors / no super-tau
        # crossings. None mirrors ``cluster_members_`` semantics for the
        # DCD-disabled case.
        if isinstance(self.dcd_, dict):
            try:
                from .._cluster_hierarchy import build_cluster_hierarchy
                self.cluster_hierarchy_ = build_cluster_hierarchy(
                    self.dcd_, X,
                    super_tau=float(getattr(self, "dcd_super_tau", 0.5)),
                    max_levels=int(getattr(self, "dcd_hierarchy_max_levels", 3)),
                    distance=str(getattr(self, "dcd_distance", "su")),
                )
            except Exception as exc:
                logger.debug("mrmr: cluster-hierarchy accessor failed; using an empty mapping: %r", exc, exc_info=True)
                self.cluster_hierarchy_ = {}
        else:
            self.cluster_hierarchy_ = None
        # 2026-05-30 Wave 9.1 fix (loop iter 1, agent-found bug):
        # When DCD's ``commit_swap`` extended ``factors_data`` inside screen
        # with PC1 aggregate columns, the swap targets land in ``selected_vars``
        # at indices >= len(nbins) here - the outer-scope ``data/cols/nbins``
        # still point at the pre-swap matrix, so downstream ``_run_fe_step``
        # crashes in ``merge_vars`` with "negative dimensions" once an
        # aggregate index is looked up. Adopt the extended matrices back
        # from DCDState so downstream FE / final remap sees them.
        if _dcd_state is not None:
            try:
                _new_p = int(_dcd_state.factors_data.shape[1])
                _cur_p = int(data.shape[1])
                if _new_p > _cur_p:
                    data = _dcd_state.factors_data
                    cols = list(_dcd_state.cols)
                    nbins = np.asarray(_dcd_state.factors_nbins, dtype=np.int64)
            except Exception as e:  # nosec B110 - non-trivial body
                # Best-effort - if DCDState is malformed, fall through.
                logger.debug("DCDState looked malformed (%s: %s) -- falling through without it", type(e).__name__, e)

        # MEMORY: prune fit-time ``_engineered_continuous_`` scratch for engineered columns that did not
        # survive THIS round's screen - see ``_prune_engineered_continuous_store`` docstring for why this
        # is safe (the FE operand pool only widens beyond ``selected_vars`` on the very first FE step,
        # before any engineered column exists). No-op when the store is empty/absent.
        if getattr(self, "_engineered_continuous_", None):
            from ._helpers import _prune_engineered_continuous_store
            _prune_engineered_continuous_store(self, cols, selected_vars)

        if fe_max_steps == 0 or num_fs_steps >= fe_max_steps:
            break

        if self.max_runtime_mins is not None:
            elapsed_min = (timer() - start_time) / 60.0
            if elapsed_min >= self.max_runtime_mins:
                ran_out_of_time = True
                if verbose:
                    logger.info("MRMR.fit: runtime budget %.1f min exceeded at FE step %d; stopping.", self.max_runtime_mins, num_fs_steps)
                break

        # SUFFICIENT-SUMMARY EARLY-STOP. The user's
        # "compare-to-theoretical-max" idea via a DPI residual test. Once the
        # current selection already captures all the information the observables
        # carry about y - i.e. the residual r = y - E_hat[y|selected] is pure
        # noise w.r.t. EVERY raw feature (all raws at the maxT permutation null)
        # AND small relative to y (Var(r)/Var(y) guard) - any future engineered
        # candidate is, by the Data-Processing Inequality, a function of the raws
        # and CANNOT have more MI with r than the raws do, so the remaining FE
        # search is provably pointless. Skip it. This NEVER changes the final
        # selection (it only skips work that could find nothing - with it OFF the
        # loop would run the remaining steps and engineer nothing new); verified
        # byte-identical on genuine multi-signal fixtures. CONSERVATIVE: stops only
        # when BOTH guards pass, so a genuine unfound second signal (incl. a
        # NONLINEAR leftover the linear E_hat underfits, caught by MI(r; raw))
        # blocks the stop. ``self.sufficient_summary_`` surfaces the verdict.
        if bool(getattr(self, "fe_sufficient_summary_early_stop", True)) and len(selected_vars) > 0:
            from .._fe_sufficient_summary import check_sufficient_summary_for_mrmr
            _ss_verdict = check_sufficient_summary_for_mrmr(
                self,
                data=data, nbins=nbins, cols=cols,
                selected_vars=selected_vars,
                target_indices=target_indices,  # type: ignore[arg-type]
                X=X, y=y, verbose=verbose,
            )
            self.sufficient_summary_ = _ss_verdict
            if _ss_verdict.reached:
                if verbose:
                    logger.info(
                        "MRMR.fit: sufficient-summary early-stop at FE step %d -- %s. " "Skipping the remaining FE search (selection unchanged).",
                        num_fs_steps,
                        _ss_verdict.reason,
                    )
                break

        # Feature engineering iteration delegated to ``_run_fe_step`` (testable / experiment-friendly outside
        # the screening loop). Returns updated state + n_recommended_features; zero breaks the outer loop.
        self._fe_steps_executed_ += 1
        fe_result = self._run_fe_step(
            data=data, cols=cols, nbins=nbins, X=X,
            target_names=target_names, target_indices=target_indices,
            selected_vars=selected_vars,
            categorical_vars=categorical_vars,
            classes_y=classes_y, classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
            unary_transformations=unary_transformations,
            binary_transformations=binary_transformations,
            engineered_features=engineered_features,
            engineered_recipes=engineered_recipes,
            checked_pairs=checked_pairs,
            times_spent=times_spent,
            num_fs_steps=num_fs_steps,
            n_jobs=n_jobs, prefetch_factor=prefetch_factor,
            parallel_kwargs=parallel_kwargs,
            _is_polars_input=_is_polars_input,
            verbose=verbose,
            fe_max_steps=fe_max_steps,
            fe_npermutations=fe_npermutations,
            fe_max_pair_features=fe_max_pair_features,
            fe_print_best_mis_only=fe_print_best_mis_only,
            fe_min_nonzero_confidence=fe_min_nonzero_confidence,
            fe_min_engineered_mi_prevalence=fe_min_engineered_mi_prevalence,
            fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
            fe_max_external_validation_factors=fe_max_external_validation_factors,
            fe_min_pair_mi=fe_min_pair_mi,
            fe_min_pair_mi_prevalence=fe_min_pair_mi_prevalence,
            fe_smart_polynom_iters=fe_smart_polynom_iters,
            fe_smart_polynom_optimization_steps=fe_smart_polynom_optimization_steps,
            fe_min_polynom_degree=fe_min_polynom_degree,
            fe_max_polynom_degree=fe_max_polynom_degree,
            fe_min_polynom_coeff=fe_min_polynom_coeff,
            fe_max_polynom_coeff=fe_max_polynom_coeff,
            fe_unary_preset=fe_unary_preset,
            fe_binary_preset=fe_binary_preset,
        )
        if fe_result is None:
            break  # FE skip: empty screening + fe_fallback_to_all=False
        data, cols, nbins, X, selected_vars, n_recommended_features = fe_result

        # Pack #5 2026-05-18: adaptive threshold relaxation. When the
        # first-pass FE produces 0 engineered features, the most likely
        # culprit on heavily-correlated feature sets is the strict
        # ``fe_min_engineered_mi_prevalence`` gate - pair-level MI is
        # near the individual-MI sum and the engineered candidate
        # cannot beat 98% of pair MI. Retry ONCE with relaxed
        # thresholds (and fe_smart_polynom_iters=0 to skip the
        # already-completed expensive Hermite Optuna phase).
        _adaptive = bool(getattr(self, "fe_adaptive_threshold_relax", True))
        _relax_factor = float(getattr(self, "fe_adaptive_relax_factor", 0.9))
        if n_recommended_features == 0 and _adaptive and fe_max_steps > 0 and num_fs_steps == 0:  # only on the very first FE step
            # fe_min_pair_mi_prevalence may be the sentinel string "auto" (debiased-ratio mode -
            # see _step_core.py's own isinstance check) rather than a float; resolve it to that
            # same established numeric convention (1.05) just for this relaxation arithmetic, same
            # as _step_core.py/_step_pairs_rank.py already do at their own use sites. Passing "auto"
            # straight into `* _relax_factor` raised TypeError: can't multiply sequence by float.
            _pair_mi_prevalence_for_relax = (
                1.05 if isinstance(fe_min_pair_mi_prevalence, str) and fe_min_pair_mi_prevalence.strip().lower() == "auto" else float(fe_min_pair_mi_prevalence)
            )
            _relaxed_engineered = fe_min_engineered_mi_prevalence * _relax_factor
            _relaxed_pair = max(1.001, _pair_mi_prevalence_for_relax * _relax_factor)
            if verbose:
                logger.info(
                    "MRMR FE: first pass found 0 engineered features; "
                    "retrying with relaxed thresholds "
                    "(engineered_mi_prevalence: %.3f -> %.3f, "
                    "pair_mi_prevalence: %.3f -> %.3f). "
                    "Skipping Hermite Optuna re-run (already cached in "
                    "_hermite_features_).",
                    fe_min_engineered_mi_prevalence, _relaxed_engineered,
                    _pair_mi_prevalence_for_relax, _relaxed_pair,
                )
            fe_result_retry = self._run_fe_step(
                data=data, cols=cols, nbins=nbins, X=X,
                target_names=target_names, target_indices=target_indices,
                selected_vars=selected_vars,
                categorical_vars=categorical_vars,
                classes_y=classes_y, classes_y_safe=classes_y_safe,
                freqs_y=freqs_y,
                cached_MIs=cached_MIs, cached_confident_MIs=cached_confident_MIs,
                unary_transformations=unary_transformations,
                binary_transformations=binary_transformations,
                engineered_features=engineered_features,
                engineered_recipes=engineered_recipes,
                checked_pairs=set(),  # reset so pairs re-evaluated under new threshold
                times_spent=times_spent,
                num_fs_steps=num_fs_steps,
                n_jobs=n_jobs, prefetch_factor=prefetch_factor,
                parallel_kwargs=parallel_kwargs,
                _is_polars_input=_is_polars_input,
                verbose=verbose,
                fe_max_steps=fe_max_steps,
                fe_npermutations=fe_npermutations,
                fe_max_pair_features=fe_max_pair_features,
                fe_print_best_mis_only=fe_print_best_mis_only,
                fe_min_nonzero_confidence=fe_min_nonzero_confidence,
                fe_min_engineered_mi_prevalence=_relaxed_engineered,
                fe_good_to_best_feature_mi_threshold=fe_good_to_best_feature_mi_threshold,
                fe_max_external_validation_factors=fe_max_external_validation_factors,
                fe_min_pair_mi=fe_min_pair_mi,
                fe_min_pair_mi_prevalence=_relaxed_pair,
                fe_smart_polynom_iters=0,  # already ran in first pass
                fe_smart_polynom_optimization_steps=fe_smart_polynom_optimization_steps,
                fe_min_polynom_degree=fe_min_polynom_degree,
                fe_max_polynom_degree=fe_max_polynom_degree,
                fe_min_polynom_coeff=fe_min_polynom_coeff,
                fe_max_polynom_coeff=fe_max_polynom_coeff,
                fe_unary_preset=fe_unary_preset,
                fe_binary_preset=fe_binary_preset,
            )
            if fe_result_retry is not None:
                data, cols, nbins, X, selected_vars, n_recommended_features = fe_result_retry
                if verbose:
                    logger.info(
                        "MRMR FE adaptive retry produced %d engineered features.",
                        n_recommended_features,
                    )

        if n_recommended_features == 0:
            break

        num_fs_steps += 1
        if num_fs_steps >= fe_max_steps:
            # CONFIRM-RESCREEN: the FE step appended engineered
            # columns and (legacy) promoted them into ``selected_vars`` BY FIAT,
            # bypassing redundancy filtering + gain accounting. Instead of
            # breaking here, loop ONCE more so the top-of-loop ``screen_predictors``
            # re-selects from the AUGMENTED pool. The engineered columns are
            # already quantised bin-code columns in ``data``/``cols``/``nbins``,
            # so MRMR treats them as ordinary candidates: a redundant engineered
            # feature (e.g. ``1/b - d**2`` whose conditional MI given an
            # already-selected ``a**2/b`` is ~0.03) is dropped by the Fleuret
            # redundancy term, and every surviving column - raw OR engineered -
            # earns a real ``mrmr_gain`` / ``support_rank``. The next iteration
            # hits the ``num_fs_steps >= fe_max_steps`` break at the TOP of the
            # loop (line ~5085) BEFORE the FE step, so FE never runs again - no
            # unbounded recursion, no new engineered columns.
            if getattr(self, "fe_reselect_after_engineering", True) and n_recommended_features > 0 and not _did_confirm_rescreen:
                _did_confirm_rescreen = True
                continue
            break  # uncomment to avoid recheck of single-rounded FE

    # ENGINEERED-OPERAND FEED-FORWARD: the continuous engineered-value
    # store is FIT-TIME SCRATCH (full-length float64 arrays of training data) used
    # only to feed engineered operands into the next FE step's pair search. Drop it
    # once the FE loop is done so it never bloats the fitted estimator or breaks
    # pickle (the replayable composite carries only its parent recipes, never these
    # arrays). No-op when the attr was never created (no engineered columns).
    # SNAPSHOT FIRST: the raw-vs-engineered conditional-redundancy drop
    # below needs the CONTINUOUS engineered values to bin the engineered survivor
    # finely (the ``data`` matrix holds only the lossy ~10-code screening bins, which
    # leave a fully-subsumed denominator operand a spurious residual CMI). Snapshot
    # into a LOCAL (never an attr -> stays out of the pickled estimator) so the del
    # below still keeps the fitted object lean.
    _eng_continuous_snapshot = dict(getattr(self, "_engineered_continuous_", None) or {})
    if hasattr(self, "_engineered_continuous_"):
        try:
            del self._engineered_continuous_
        except Exception as exc:
            logger.debug("mrmr: engineered-continuous store failed; using an empty mapping: %r", exc, exc_info=True)
            self._engineered_continuous_ = {}

    # Surfaced at verbose>=1 (2026-07-09; was gated behind verbose>2, an unrealistically high bar that
    # left this cumulative-per-operator timing breakdown effectively invisible to normal production runs).
    if verbose and times_spent:
        logger.info("MRMR FE time spent by binary func (cumulative across all rounds): %s", sort_dict_by_value(times_spent))
    # Possibly decide on eliminating original features? (if constructed ones cover 90%+ of MI)

    # ---------------------------------------------------------------------------------------------------------------
    # Drop temporary targets
    # ---------------------------------------------------------------------------------------------------------------

    # Fuzz-caught: previous ``X = X.drop(columns=target_names)`` returned a new DataFrame and only rebound the
    # local; for pandas input (where X.loc[:, target_names] = ... mutated the caller's frame), the caller's
    # X was left with the injected ``targ_<id>`` columns, which leaked into downstream sklearn pipeline
    # (imputer/scaler recorded them in feature_names_in_ and raised on transform). Fix: drop in place (pandas)
    # or rebind (polars - immutable, caller's X was never mutated).
    if _is_polars_input:
        X = X.drop(target_names)  # no-copy lazy op; caller's X untouched
    else:
        # option_context silences the conservative SettingWithCopy heuristic (fires when the caller passed a sliced
        # view); the in-place drop reverses this function's own targ_<id> injection on the same object, no copy.
        with pd.option_context("mode.chained_assignment", None):
            X.drop(columns=target_names, inplace=True)  # noqa: PD002 - must mutate the caller's frame OBJECT in place (restores its original schema by identity), not rebind a local; `X = X.drop(...)` would silently stop touching the caller's actual frame

    # DCD orphaned-cluster raw re-attach. A DCD AGGREGATE swap replaces the raw
    # anchor with the (engineered, non-support_) aggregate column; when that
    # anchor was the cluster's only selected raw column the latent disappears
    # from the raw ``support_`` (which indexes feature_names_in_ only) even
    # though the denoised aggregate survives in ``get_feature_names_out`` /
    # ``transform``. Run on the FINAL ``selected_vars`` (after the confirm-
    # rescreen loop has fully settled, so this can never perturb a subsequent
    # re-selection) to re-attach one raw cluster member per orphaned aggregate,
    # keeping each collapsed latent visible in BOTH the raw support and the
    # transform output. Best-effort; never breaks fit.
    if _dcd_state is not None and len(selected_vars):
        try:
            from .._dynamic_cluster_discovery import (
                reattach_raw_representative_after_aggregate_swap as _dcd_reattach_raw,
            )
            _sv_list = list(selected_vars)
            _sv_set = {int(s) for s in _sv_list}
            _agg_indices = [
                int(e.get("new_col_idx"))
                for e in (getattr(_dcd_state, "swap_log", None) or [])
                if str(e.get("branch", "aggregate")) == "aggregate" and e.get("aggregate_name") and e.get("new_col_idx") is not None
            ]
            for _agg_idx in _agg_indices:
                if _agg_idx in _sv_set:
                    _dcd_reattach_raw(_dcd_state, _agg_idx, _sv_list)
            selected_vars = _sv_list
        except Exception as _reattach_exc:
            logger.warning(
                "DCD orphaned-cluster raw re-attach failed (%s); continuing.",
                _reattach_exc,
            )

    # ---------------------------------------------------------------------------------------------------------------
    # Friend-graph post-analysis (diagnostic; optional pruning). Built here, while ``selected_vars``,
    # ``data``, ``nbins`` and ``target_indices`` are all still in cols-space, BEFORE the remap below
    # rebinds ``selected_vars`` to original-frame indices. When pruning is enabled the pruned cols-space
    # list flows through that same remap into ``support_``. Never allowed to break fit - guarded.
    # ---------------------------------------------------------------------------------------------------------------
    from ._friend_graph_and_redundancy import _friend_graph_and_redundancy_passes

    selected_vars, cols, data, nbins = _friend_graph_and_redundancy_passes(
        self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
        _effective_min_relevance_gain=_effective_min_relevance_gain,
        _hinge_deferred_recipes=_hinge_deferred_recipes,
        _hinge_deferred_values=_hinge_deferred_values,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes,
        _miss_ind_pre_recipes=_miss_ind_pre_recipes,
        _persisted_dcd_state=_persisted_dcd_state,
        _y_np=_y_np,
        fe_to_pandas=fe_to_pandas,
        _fe_family_on=_fe_family_on,
    )
    # ---------------------------------------------------------------------------------------------------------------
    # selected_vars: cols-indices -> names -> original-frame indices (categorize_dataset may rearrange cat columns).
    # ---------------------------------------------------------------------------------------------------------------

    selected_vars_names = np.array(cols)[np.array(selected_vars, dtype=np.intp)]

    # BUG2: the cross-fold stability vote in ``_run_fe_step`` pops a
    # fold-unstable engineered recipe AND de-selects its column for that step, but the
    # materialised bin-code column stays in ``cols``/``data``, so the downstream greedy
    # screen (step>1 re-screen / final selection) re-admits it on marginal MI - it then
    # arrives here with NO recipe and was silently DROPPED from transform output (a
    # select-then-drop contract violation: a feature in support_/discovered MUST survive
    # transform). The vote is authoritative, so strip every vote-rejected engineered name
    # from the selection BEFORE finalising support_/discovered: the column never re-enters
    # support_, get_feature_names_out, or _engineered_features_. ``selected_vars`` is filtered
    # in lockstep (by cols-index) so the raw integer support stays consistent.
    _vote_dropped_names = getattr(self, "_fe_stability_vote_dropped_", None)
    if _vote_dropped_names:
        _keep_mask = np.array([nm not in _vote_dropped_names for nm in selected_vars_names], dtype=bool)
        if not _keep_mask.all():
            _kept_idx_positions = np.nonzero(_keep_mask)[0]
            selected_vars = [selected_vars[i] for i in _kept_idx_positions]
            selected_vars_names = selected_vars_names[_keep_mask]
            if verbose:
                logger.info(
                    "MRMR.fit: stripped %d cross-fold-vote-rejected engineered feature(s) from the "
                    "final selection so they cannot re-enter support_ without a replayable recipe.",
                    int((~_keep_mask).sum()),
                )
    # Tolerate FE-engineered names: screening output may include synthetic feature names not in
    # feature_names_in_; record them in self._engineered_features_ instead of raising on the .index() lookup.
    # Also surface matching EngineeredRecipe (built during _run_fe_step) so transform() can replay each
    # engineered column on test data. An engineered name without a recipe (e.g. higher-order interaction
    # whose parents are themselves engineered) is recorded by name only and dropped from transform output.
    self._engineered_features_ = []
    self._engineered_recipes_ = []
    original_indices = []
    engineered_without_recipe = []
    # feature_names_in_ is an ndarray (sklearn convention); name -> index map built once (O(F))
    # instead of an ``in`` test + ``.index()`` rescan per ``col`` (O(F) each) - turns the O(K*F)
    # loop below into O(K+F).
    _fni_idx = {nm: i for i, nm in enumerate(self.feature_names_in_)}
    for col in selected_vars_names:
        _fni_i = _fni_idx.get(col)
        if _fni_i is not None:
            original_indices.append(_fni_i)
        else:
            self._engineered_features_.append(col)
            recipe = engineered_recipes.get(col)
            if recipe is not None:
                self._engineered_recipes_.append(recipe)
            else:
                engineered_without_recipe.append(col)
    if engineered_without_recipe and verbose:
        # Happens with fe_max_steps>1 when a higher-order interaction's parents are themselves engineered features. The recipe replay path can only
        # reconstruct 1-deep engineering; deeper nests are recorded in self._engineered_features_ but DROPPED from transform output. Surface the cost.
        logger.warning(
            "MRMR.fit: %d engineered feature(s) selected without replayable recipe (nested-engineered parents at fe_max_steps=%d); they will be DROPPED from transform output: %s",
            len(engineered_without_recipe), self.fe_max_steps, engineered_without_recipe[:8],
        )
    # ``selected_vars`` is downstream re-bound to the integer indices of the RAW columns only; engineered features are appended in transform() via
    # ``_append_engineered`` using ``self._engineered_recipes_``. This split mirrors the on-disk contract: support_ indexes feature_names_in_; engineered output
    # columns come from the recipes list. n_features_ counts BOTH (see assignment below).
    selected_vars = original_indices

    # PSEUDO-REMIX OPERAND RE-ADD. A surviving conditional-gate / binned-numeric-agg /
    # row-argmax composite (``gate_mask__a__b`` / ``binagg_*(c|qbin(a))`` / ``argmax__a__b``) is a LOSSY
    # threshold/binning re-mix of its raw operands: it survived because it captures the INTERACTION, but
    # it destroys each operand's continuous value that a LINEAR downstream needs (measured: a 5-class
    # LogReg scored macro-F1 0.62 when x2 lived ONLY inside ``gate_mask__x1__x2`` vs >0.70 with raw x2
    # restored). The operands typically have WEAK MARGINAL MI (signal is in the joint), so the screen /
    # marginal retention never surface them. When a CO-operand is ALREADY in the raw support (e.g. x1
    # selected beside ``gate_mask__x1__x2``) the composite is a vouched genuine multi-source interaction,
    # so restore the other raw operand(s). Runs here (engineered roster + raw support both final). A
    # single-operand self-gate gets no vouch; a noise-paired gate has low joint MI and rarely survives.
    # PASSTHROUGH RE-ATTACH. Embedding/text columns excluded from the MI screen above are re-added to the selected set so transform() emits them unchanged. Their
    # indices are looked up in ``feature_names_in_`` (which includes them, in original order). Appended AFTER the screen so they never participate in MI/redundancy
    # but always survive to the estimator (the learnable-embedding network + boundary encoder consume them).
    if self._passthrough_features_:
        _existing = set(selected_vars)
        # Reuse the name -> index map built above (``feature_names_in_`` is fit-invariant).
        for _pname in self._passthrough_features_:
            _pidx = _fni_idx.get(_pname)
            if _pidx is not None and _pidx not in _existing:
                selected_vars.append(_pidx)
                _existing.add(_pidx)

    # ---------------------------------------------------------------------------------------------------------------
    # additional_rfecv run
    # ---------------------------------------------------------------------------------------------------------------

    if self.run_additional_rfecv_minutes:
        """On the factors discarded by MRMR, let's run RFECV to see if any of them participate in interactions"""
        n_unexplored = X.shape[1] - len(selected_vars)
        if n_unexplored > 0:
            if verbose:
                logger.info(
                    "Running RFECV for %s minute(s) over %s feature(s) discarded by MRMR to extract interactions...",
                    self.run_additional_rfecv_minutes,
                    f"{n_unexplored:_}",
                )

            from mlframe.training import get_training_configs

            configs = get_training_configs(has_time=True)

            params = configs.COMMON_RFECV_PARAMS.copy()
            params["max_runtime_mins"] = self.run_additional_rfecv_minutes
            # Wire MRMR.cv / cv_shuffle into the additional RFECV pass; pre-fix they were dead constructor params.
            # ``params`` may already carry ``cv`` from configs.COMMON_RFECV_PARAMS; MRMR's explicit setting wins.
            params.update(self._rfecv_cv_kwargs())
            # Parsimony for the rescue: RFECV's recall-oriented default ('one_se_max') keeps the LARGEST subset within 1 SE, which on a
            # noise-robust booster re-admits ~the whole discarded pool and undoes MRMR's selection. Pin the smallest-within-1-SE rule so the
            # rescue re-adds only discarded features that genuinely lift CV. setdefault lets COMMON_RFECV_PARAMS / additional_rfecv_kwargs win.
            params.setdefault("n_features_selection_rule", getattr(self, "additional_rfecv_selection_rule", "one_se_min"))
            _extra_rfecv = getattr(self, "additional_rfecv_kwargs", None)
            if _extra_rfecv:
                params.update(_extra_rfecv)

            # Classifier-vs-regressor detection. Preference order:
            #   1) Explicit ``target_type`` attribute on self (set by the caller / harness).
            #   2) Honest dtype + cardinality heuristic: float dtype is regression by
            #      construction (zero-inflated targets like ``[0]*900 + [1.7, 2.4, ...]``
            #      satisfy the legacy ratio>100 but are NOT classification). Integer
            #      dtype with ratio>100 AND small absolute cardinality (<=64 unique
            #      values) is classification. Everything else is regression.
            # Pre-fix, the regression else-branch silently skipped the
            # additional-RFECV pass entirely, so regression callers got no benefit
            # from run_additional_rfecv_minutes. The dtype guard prevents misclassifying
            # zero-inflated float targets. fix audit row FS-L-2.
            _explicit_tt = getattr(self, "target_type", None)
            if _explicit_tt is not None:
                _tt_str = str(_explicit_tt).lower()
                _is_classification = "classif" in _tt_str or _tt_str in ("binary", "multiclass", "multilabel")
            else:
                _y_arr = np.asarray(y)
                _n_unique = len(np.unique(_y_arr))
                _ratio = len(_y_arr) / max(1, _n_unique)
                _is_float = _y_arr.dtype.kind == "f"
                _is_classification = (not _is_float) and _ratio > 100 and _n_unique <= 64
                if _ratio > 100 and _is_float:
                    logger.warning(
                        "MRMR.run_additional_rfecv: target is float dtype with %d unique values; "
                        "treating as regression despite samples/unique ratio %.1f>100. Pass "
                        "target_type='classification' explicitly to override.",
                        _n_unique, _ratio,
                    )
            # order-preserving set
            # difference. The prior ``list(set(X.columns) - set(...))``
            # produced a HASH-SEED-DEPENDENT column order because Python's
            # randomized string hashing reorders ``set`` iteration across
            # processes. That order flowed into RFECV's CatBoost feature
            # importances, whose tie-breaks then gave different
            # ``self.support_`` across runs that differed only in
            # ``PYTHONHASHSEED``. Concrete demo: 5/5 distinct orderings
            # observed across seeds 0-4. Breaks the "same random_seed ->
            # identical support_" contract for any user with
            # ``run_additional_rfecv_minutes`` > 0.
            # ``selected_vars`` indexes ``feature_names_in_`` (full, includes passthrough); ``X`` here is the passthrough-narrowed working frame, so map names via
            # ``feature_names_in_`` rather than ``X.columns[...]`` (positional mismatch when passthrough is active). Passthrough columns are never in the narrowed X
            # and never enter the RFECV rescue pool below regardless.
            _sel_names = {self.feature_names_in_[i] for i in selected_vars}
            # Cluster members already folded into a denoised aggregate (post-hoc cluster_aggregate 'replace' mode,
            # _cluster_aggregate_removals_) or into a DCD PC1/mean_z swap (cluster_members_) are REPRESENTED by that
            # aggregate. Excluding them from the rescue pool stops RFECV re-admitting the raw members and re-injecting
            # the very redundancy the aggregation removed - only features dropped for low marginal/joint relevance get reconsidered.
            _excluded_from_rescue = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
            _cm = getattr(self, "cluster_members_", None)
            if isinstance(_cm, dict):
                for _anchor, _members in _cm.items():
                    _excluded_from_rescue.add(_anchor)
                    if isinstance(_members, (list, tuple, set)):
                        _excluded_from_rescue.update(_members)
            # Engineered FE columns (univariate basis a__T2, hybrid/pair/triplet crosses,
            # MI-greedy) survive in X.columns but were deliberately excluded from
            # feature_names_in_ (raw columns only, line above). They cannot be indexed
            # into support_ via feature_names_in_.index() -> ValueError. Exclude them from
            # the rescue pool so RFECV only reconsiders RAW discarded columns.
            _excluded_from_rescue.update(getattr(self, "hybrid_orth_features_", None) or [])
            _excluded_from_rescue.update(getattr(self, "mi_greedy_features_", None) or [])
            # Raw operands the conditional-redundancy sweep judged FULLY SUBSUMED by a
            # surviving engineered child (``_raw_redundancy_dropped_``) must NOT re-enter
            # via the RFECV rescue pool. The n-invariant CMI verdict is authoritative: a
            # raw whose entire y-information is captured by an admitted engineered feature
            # (e.g. ``a`` / ``b`` in ``a**2/b`` once ``div(neg(a),sqrt(b))`` is selected)
            # carries no independent signal, but CatBoost RFECV - which scores raw
            # MARGINAL usefulness, blind to the engineered child's coverage - would re-admit
            # it, resurrecting the exact redundancy the sweep removed (observed at n=2000/5000
            # on ``y=0.30 a**2/b``: the sweep dropped a+b, RFECV re-added a). Excluding the
            # dropped set keeps the redundancy decision consistent across both the FE-step
            # finalisation AND the downstream RFECV rescue.
            _excluded_from_rescue.update(getattr(self, "_raw_redundancy_dropped_", None) or set())
            temp_columns = [c for c in X.columns if c not in _sel_names and c not in _excluded_from_rescue]

            if _is_classification:
                cb_num_rfecv = RFECV(
                    estimator=CatBoostClassifier(**configs.CB_CLASSIF),
                    fit_params=dict(plot=False),
                    cat_features=categorical_vars_names,
                    scoring=make_scorer(score_func=compute_probabilistic_multiclass_error, response_method="predict_proba", greater_is_better=False),
                    **params,
                )
            else:
                # Regression branch: CatBoostRegressor with the same shared params; default scoring lets
                # RFECV pick from the estimator (negative-MSE-like). Keeping the import local avoids
                # paying the CatBoostRegressor import cost when only classification is exercised.
                from catboost import CatBoostRegressor
                cb_num_rfecv = RFECV(
                    estimator=CatBoostRegressor(**configs.CB_REGR),
                    fit_params=dict(plot=False),
                    cat_features=categorical_vars_names,
                    **params,
                )
            cb_num_rfecv.fit(X[temp_columns], y)

            if cb_num_rfecv.n_features_ > 0:
                new_features = np.array(temp_columns)[cb_num_rfecv.support_]
                if verbose:
                    logger.info("RFECV selected %d additional feature(s): %s", cb_num_rfecv.n_features_, new_features)
                # Reuse the name -> index map built above (``feature_names_in_`` is fit-invariant).
                for feature in new_features:
                    selected_vars.append(_fni_idx[feature])
            else:
                if verbose:
                    logger.info("RFECV selected no additional features.")

    # ---------------------------------------------------------------------------------------------------------------
    # Assign support
    # ---------------------------------------------------------------------------------------------------------------

    from ._assign_support import _assign_support

    _assign_support(
        self,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        target_indices=target_indices,
        y=y,
        verbose=verbose,
        fe_max_steps=fe_max_steps,
        cached_MIs=cached_MIs,
        engineered_recipes=engineered_recipes,
        predictors=predictors,
        _eng_continuous_snapshot=_eng_continuous_snapshot,
        selected_vars=selected_vars,
    )

    # ---------------------------------------------------------------------------------------------------------------
    # Report FS results
    # ---------------------------------------------------------------------------------------------------------------
    from ._finalise import _finalise_fs_results

    return _finalise_fs_results(
        self,
        MRMR=MRMR,
        X=X,
        classes_y=classes_y,
        cols=cols,
        data=data,
        nbins=nbins,
        predictors=predictors,
        start_time=start_time,
        verbose=verbose,
        cache_key=_cache_key,
        signature=signature,
        ran_out_of_time=ran_out_of_time,
        hashable_params_signature=_hashable_params_signature,
        mrmr_cache_bytes_total=_mrmr_cache_bytes_total,
        align_mrmr_gains=_align_mrmr_gains,
        fit_cache_lock=_MRMR_FIT_CACHE_LOCK,
    )
