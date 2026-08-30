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


from ._helpers import _mrmr_cache_bytes_total

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
        numeric_column_names,
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

    from ._fe_stage_cascade_early_a import _fe_stage_cascade_early_a
    from ._fe_stage_cascade_early_b import _fe_stage_cascade_early_b

    # Recipe registries for every FE family from Layer 23 through Layer 104: each family owns exactly
    # one dict here, mutated in place by whichever stage (inline below, or one of the _fe_stage_cascade_*
    # siblings) actually runs that family -- declared once, up front, so every consumer downstream (the
    # end-of-fit ``engineered_recipes.update(...)`` remap) sees the SAME dict objects regardless of which
    # sibling module mutated them. Passing a dict into a sibling and having it mutate entries in place is
    # safe (no return needed); a REASSIGNMENT inside a sibling would NOT propagate back -- confirmed via a
    # systematic check that none of these are ever reassigned (only ``[key] = value`` mutated) in the two
    # early-cascade siblings below.
    _hybrid_orth_pre_recipes: dict = {}
    _mi_greedy_pre_recipes: dict = {}
    _kfold_te_pre_recipes: dict = {}
    _binned_agg_pre_recipes: dict = {}
    _count_enc_pre_recipes: dict = {}
    _freq_enc_pre_recipes: dict = {}
    _cat_num_pre_recipes: dict = {}
    _miss_ind_pre_recipes: dict = {}
    _miss_cnt_pre_recipes: dict = {}
    _miss_pat_pre_recipes: dict = {}
    _ratio_pre_recipes: dict = {}
    _log_ratio_pre_recipes: dict = {}
    _grouped_delta_pre_recipes: dict = {}
    _lagged_diff_pre_recipes: dict = {}
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
    _grouped_agg_pre_recipes: dict = {}
    _composite_group_agg_pre_recipes: dict = {}
    _grouped_quantile_pre_recipes: dict = {}

    X, _raw_input_cols_pre_fe, _hinge_deferred_values, _hinge_deferred_recipes = _fe_stage_cascade_early_a(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok,
        _hybrid_orth_pre_recipes=_hybrid_orth_pre_recipes, _mi_greedy_pre_recipes=_mi_greedy_pre_recipes,
    )
    X = _fe_stage_cascade_early_b(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fit_entry_nan_mask=_fit_entry_nan_mask, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _kfold_te_pre_recipes=_kfold_te_pre_recipes, _binned_agg_pre_recipes=_binned_agg_pre_recipes,
        _count_enc_pre_recipes=_count_enc_pre_recipes, _freq_enc_pre_recipes=_freq_enc_pre_recipes,
        _cat_num_pre_recipes=_cat_num_pre_recipes,
        _miss_ind_pre_recipes=_miss_ind_pre_recipes, _miss_cnt_pre_recipes=_miss_cnt_pre_recipes,
        _miss_pat_pre_recipes=_miss_pat_pre_recipes,
        _ratio_pre_recipes=_ratio_pre_recipes, _log_ratio_pre_recipes=_log_ratio_pre_recipes,
        _grouped_delta_pre_recipes=_grouped_delta_pre_recipes, _lagged_diff_pre_recipes=_lagged_diff_pre_recipes,
    )
    from ._fe_stage_cascade_mid_a import _fe_stage_cascade_mid_a

    X = _fe_stage_cascade_mid_a(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _cat_pair_pre_recipes=_cat_pair_pre_recipes, _cat_triple_pre_recipes=_cat_triple_pre_recipes,
        _composite_group_agg_pre_recipes=_composite_group_agg_pre_recipes,
        _conditional_gate_pre_recipes=_conditional_gate_pre_recipes,
        _grouped_agg_pre_recipes=_grouped_agg_pre_recipes,
        _grouped_quantile_pre_recipes=_grouped_quantile_pre_recipes,
        _integer_lattice_pre_recipes=_integer_lattice_pre_recipes,
        _modular_pre_recipes=_modular_pre_recipes,
        _numeric_decompose_pre_recipes=_numeric_decompose_pre_recipes,
        _pairwise_modular_pre_recipes=_pairwise_modular_pre_recipes,
        _row_argmax_pre_recipes=_row_argmax_pre_recipes,
    )
    from ._fe_stage_cascade_mid_b import _fe_stage_cascade_mid_b

    X = _fe_stage_cascade_mid_b(
        self, X=X, y=y, verbose=verbose, fe_max_steps=fe_max_steps, _y_np=_y_np, _fe_family_on=_fe_family_on,
        _fe_budget_ok=_fe_budget_ok, _raw_input_cols_pre_fe=_raw_input_cols_pre_fe,
        _group_distance_pre_recipes=_group_distance_pre_recipes,
        _rare_category_pre_recipes=_rare_category_pre_recipes,
        _conditional_residual_pre_recipes=_conditional_residual_pre_recipes,
        _conditional_dispersion_pre_recipes=_conditional_dispersion_pre_recipes,
        _conditional_quantile_rank_pre_recipes=_conditional_quantile_rank_pre_recipes,
        _ordinal_pattern_pre_recipes=_ordinal_pattern_pre_recipes,
        _random_fourier_pre_recipes=_random_fourier_pre_recipes,
        _sir_direction_pre_recipes=_sir_direction_pre_recipes,
        _lof_pre_recipes=_lof_pre_recipes,
        _mahalanobis_density_pre_recipes=_mahalanobis_density_pre_recipes,
        _wavelet_pre_recipes=_wavelet_pre_recipes,
        _rankgauss_pre_recipes=_rankgauss_pre_recipes,
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
    # MEM: force a GC pass before discretizing (PEAK-RSS bound). This used to also explicitly ``del``
    # ~50 per-family FE intermediate DataFrames (``X_t``/``X_q``/``X_te``/...) that used to live as
    # ``_fit_impl`` locals; every one of those families has since been carved into a cascade sibling
    # (_fe_stage_cascade_early_a.py, _fe_stage_cascade_early_b.py, _fe_stage_cascade_mid_a.py,
    # _fe_stage_cascade_mid_b.py, _hybrid_orth_family_variants.py) across four Tier F waves, so each
    # intermediate is now genuinely function-local to its own sibling and reclaimed automatically on
    # that function's return -- the explicit per-name ``del`` block is fully retired (ruff F821 on
    # each remaining bare ``del X_t``-style name, confirmed unbound anywhere in this scope, is what
    # caught each wave's leftover dead entries). The GC pass itself is still worth keeping: many
    # families above build full-width DataFrames as local variables INSIDE their own sibling
    # function, which Python's refcounting already frees on return, but an explicit collect here
    # still helps against any reference cycle (e.g. a DataFrame's own internal caches) before the
    # ``categorize_dataset`` peak.
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

    # Which columns the cardinality pre-screen may judge by bin count. Under a SUPERVISED nbins strategy (the
    # default MDLP) a numeric column earns bins for explaining the target, so its bin count is a signal-strength
    # measure, not a cardinality one -- feeding it to a "too many levels" guard drops the best feature. Only
    # genuinely categorical columns, whose bins ARE their levels, stay eligible. With no supervised strategy every
    # bin count is unsupervised again and the guard applies to all of them, as before.
    _numeric_names = numeric_column_names(_x_for_cat) if _nbins_strategy else set()

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
            _any_influencing,
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
            # Recomputed from the CURRENT ``cols``: the cat-interaction FE step rebinds data/cols/nbins with
            # engineered columns, and a crossed categorical can carry real cardinality, so it must stay
            # eligible for the ceiling rather than inherit an exemption computed before it existed.
            raw_cardinality_cols=(None if not _nbins_strategy else {c for c in cols if c not in _numeric_names}),
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
    from .._fe_frame_ops import fe_to_pandas

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
