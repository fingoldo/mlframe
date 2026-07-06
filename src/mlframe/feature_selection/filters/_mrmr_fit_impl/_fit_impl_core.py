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
import textwrap
import threading
import warnings
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Guards every read-then-mutate sequence on the process-wide ``MRMR._FIT_CACHE`` (lookup + ``move_to_end`` on
# a hit; ``__setitem__`` + ``move_to_end`` + LRU/byte-cap ``popitem`` on store). Concurrent fits -- multi-target
# discovery, joblib-threading callers, web-service workers -- otherwise race ``popitem``/``__setitem__``/
# ``move_to_end`` on the same OrderedDict and can raise KeyError or evict the wrong entry. RLock so a wrapped
# region may safely re-enter. The companion ``_MRMR_IDENTITY_FP_CACHE`` already had its own lock; this closes the
# same gap for the fit cache. Exposed on the ``MRMR`` class (idempotently, inside the fit body) as
# ``_FIT_CACHE_LOCK`` so any other holder of the cache can take the same canonical lock.
_MRMR_FIT_CACHE_LOCK = threading.RLock()


def _pgn_raw_budget(ceiling: int, n_engineered: int) -> int:
    """Raw-feature budget under the p>=n FP-control cap: the total ``ceiling`` (= ``max(20, p//3)``) minus the
    engineered survivors that already reach the transform output, floored at 0. Engineered features are charged
    against the ceiling so the p>=n total (raw + engineered) never exceeds it; a higher ``n_engineered`` therefore
    tightens the raw budget. Pulled out as a pure function so the cap arithmetic is unit-testable in isolation."""
    return max(0, int(ceiling) - int(n_engineered))

# Above this many bytes of nullable-column data, densify masked columns one-per-``assign`` instead of all at once
# so peak extra RAM stays ~one float64 column rather than ~2x the whole nullable subset (100GB-frame safe).
_NULLABLE_DENSIFY_EAGER_MAX_BYTES = 2 * 1024**3

"""MRMR._fit_impl main fit body.

The irreducible single function _fit_impl (bound onto the MRMR class
at the mrmr package facade) lives here verbatim. It is LOC-budget exempt:
one giant function cannot be split without distorting the fit control flow.
Its many lazy in-body from ..X import ... imports break the
mrmr -> _mrmr_fit_impl -> mrmr cycle; the small free helpers it calls live
in the sibling _helpers.py.
"""


from ._helpers import _dispatch_default_scorer, _mrmr_cache_bytes_total, _orth_fe_numeric_cols, _build_stability_replay_state, fe_decide_on_subsample

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
    # the user supplied with NaN -- its quantile-edge transform replay has no NaN bin, so a NaN test value would
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
    if hasattr(X, "columns"):
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
        except Exception:
            _x_cols_sig = None
    # 2026-05-30 Wave 9.1 fix (loop iter 36): fold X content hash into
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
    # Pre-fix the signature was ``(X.shape, y.shape, y_hash, x_hash, x_cols)`` -- SELECTOR PARAMS were
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
    try:
        _self_params_sig = _hashable_params_signature(self.get_params(deep=True))
    except Exception:
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
    # because the suite copies X between iterations -- different id() but identical content);
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
    except Exception:
        _cache_key = None
    _cached = None
    if _cache_key is not None:
        with _MRMR_FIT_CACHE_LOCK:
            if _cache_key in MRMR._FIT_CACHE:
                _cached = MRMR._FIT_CACHE[_cache_key]
                MRMR._FIT_CACHE.move_to_end(_cache_key)
    if _cached is not None:
        _replayed = _replay_fitted_state(self, _cached)
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

    # Carry an absolute deadline to the OPTIONAL enrichment FE generators (orth / extra-basis / pair-cross) so a single
    # wide-frame enrichment pass that starts before the budget is spent still aborts its per-column / per-pair loop at the
    # deadline instead of running tens of seconds past a tiny max_runtime_mins. Enrichment-only: the core screen / greedy
    # MI is never gated, so an aborted pass still leaves a usable partial selection. Cleared in the finally below.
    from .._fe_deadline import set_fe_deadline as _set_fe_deadline
    _set_fe_deadline((start_time + self.max_runtime_mins * 60.0) if self.max_runtime_mins is not None else None)

    def _fe_budget_ok() -> bool:
        # Pre-FE univariate generators (extra-basis, wavelet, dispersion, ...) run once before the FE loop and the
        # between-step guard below cannot bound a single long stage; gate each heavy default-ON stage on the remaining
        # wall-clock so an oversized fit handed a small max_runtime_mins aborts within a small multiple of the budget.
        if self.max_runtime_mins is None:
            return True
        return (timer() - start_time) / 60.0 < self.max_runtime_mins

    dtype = self.dtype

    parallel_kwargs = self.parallel_kwargs
    n_jobs = self.n_jobs
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
    # 2026-05-30 Wave 9.1 fix (loop iter 27): record a sentinel
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
    # the working frame so the screen / FE / MI never see them, but PASS THEM THROUGH to the transform output unchanged -- the learnable-embedding MLP / recurrent
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
            # Column-subset selection shares the underlying column buffers (no row copy) -- RAM-safe on 100+ GB frames. The original full column order is recovered
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
            # either way, so the caller's frame is never mutated -- the densification stays RAM-safe on 100+ GB frames.
            if int(len(X)) * len(_nullable_num) * 8 <= _NULLABLE_DENSIFY_EAGER_MAX_BYTES:
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
    # ADAPTIVE-FREQUENCY Fourier (2026-06-03): names of the held-out-validated
    # adaptive sin/cos columns the extra-basis stage emitted. Used by the
    # support-finalisation ADAPTIVE-PROTECTION block to re-add any the MRMR
    # screen dropped. Always present (empty when no adaptive freq detected) so
    # transform / pickle / clone never trip on a missing attribute.
    self._adaptive_fourier_features_ = []
    # HINGE / change-point (2026-06-09): names of the held-out-tau-validated
    # hinge legs the change-point stage emitted. Used by the support-
    # finalisation HINGE-PROTECTION block to re-add any the MRMR screen dropped
    # (a single relu leg is MONOTONE -> MI-INVARIANT by the DPI, so the greedy
    # MI screen drops it as redundant with raw x exactly as it drops the adaptive
    # Fourier legs -- its value is downstream linear usability, not MI). Always
    # present (empty when hinge off / no kink) so transform / pickle / clone
    # never trip on a missing attribute.
    self._hinge_features_ = []
    # SUFFICIENT-SUMMARY EARLY-STOP verdict (backlog #22). The fitted-attribute mirror of
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
    # a**2 etc.) closes that -- ``fe_univariate_basis_enable`` (default True)
    # runs JUST the univariate basis FE, uplift-gated via ``min_uplift`` in
    # ``hybrid_orth_mi_fe_with_recipes`` so it is near-no-op when there is no
    # univariate nonlinearity, independent of the heavier pair-CROSS-basis stage
    # which stays behind ``fe_hybrid_orth_enable``. Recovery pinned in
    # ``test_biz_value_mrmr_univariate_basis_fe.py``.
    _hybrid_on = bool(getattr(self, "fe_hybrid_orth_enable", False))
    _univ_basis_on = bool(getattr(self, "fe_univariate_basis_enable", True))
    if (_hybrid_on or _univ_basis_on) and _fe_budget_ok():
        # Polars frames: skip with a warning -- hybrid FE pipeline operates on
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
            #   (a) Float-encoded discrete labels (0.0/1.0) -- safe to cast to int64.
            #   (b) Continuous regression target -- truncating to int destroys the
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
                    except Exception:
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
                    except Exception:
                        _y_for_extra = _y_for_extra.astype(np.int64)
        _top_k_for_extra = int(getattr(self, "fe_hybrid_orth_top_k", 5))
        # Effective extra-basis set. Two independent contributors:
        #   * the EXPLICIT ``fe_hybrid_orth_extra_bases`` config, but only under
        #     the heavy ``fe_hybrid_orth_enable`` master switch (legacy gate -- a
        #     user who set the config but not the master expected a no-op);
        #   * the DEFAULT-ON Fourier univariate basis (``fe_univariate_fourier_enable``),
        #     which runs in the univariate path WITHOUT the master switch so a
        #     pure oscillatory signal (sin/cos) is recovered by default. The
        #     extra-basis stage is uplift + multiple-comparison gated downstream,
        #     so adding Fourier is near-no-op when there is no oscillation.
        _univ_fourier_on = bool(getattr(self, "fe_univariate_fourier_enable", True))
        _eff_extra_bases = tuple(_extra_bases_cfg) if (_extra_bases_cfg and _hybrid_on) else ()
        # The default-on Fourier univariate basis is part of the plug-in univariate dispatch. Under an alternate ``fe_hybrid_orth_default_scorer`` (cmim / jmim / ksg / ...) the routing
        # runs ONLY the univariate basis-selection for that scorer (the pair stage is likewise skipped above); the Fourier extra basis is a plug-in-path addition, so adding it under
        # alternate routing would emit columns the routed scorer never selected and diverge from a direct call to that scorer. Gate it to plug-in routing.
        try:
            _extra_basis_scorer_ok = _default_scorer == "plug_in"
        except NameError:
            _extra_basis_scorer_ok = True
        if _univ_fourier_on and _univ_basis_on and _extra_basis_scorer_ok and "fourier" not in _eff_extra_bases:
            _eff_extra_bases = _eff_extra_bases + ("fourier",)
        if _eff_extra_bases:
            try:
                from .._orthogonal_univariate_fe import (
                    hybrid_orth_extra_basis_fe_with_recipes,
                )

                _fourier_freqs = tuple(float(f) for f in getattr(self, "fe_hybrid_orth_fourier_freqs", (1.0, 2.0)))
                _spline_knots = int(getattr(self, "fe_hybrid_orth_spline_knots", 5))
                _fourier_powers = tuple(int(p) for p in getattr(self, "fe_hybrid_orth_fourier_powers", (1, 2)))
                _X_before_extra_cols = list(X.columns)
                # Build the extra basis (Fourier/spline) on RAW columns only --
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
                # ADAPTIVE-FREQUENCY Fourier (2026-06-03): default ON. The
                # fixed grid {1, 2} misses arbitrary-period oscillations
                # (sin(3.7*x), sin(5.3*x)); the adaptive detector sweeps a
                # coarse z-space grid + local-refines + held-out-validates
                # the dominant frequency per column, n-gated at >= 800 rows
                # (smaller n false-positives a chance frequency). The
                # emitted adaptive sin/cos recipes are tagged adaptive=True
                # and PROTECTED past screening below (a single leg has low
                # marginal MI -- phase -- so the screen would drop the
                # held-out-validated pair otherwise).
                _fourier_adaptive = bool(getattr(self, "fe_univariate_fourier_adaptive", True))
                _fourier_adaptive_mvc = float(
                    getattr(
                        self,
                        "fe_univariate_fourier_adaptive_min_val_corr",
                        0.15,
                    )
                )
                # ADAPTIVE-CHIRP (2026-06-03): second argument-warp path. Runs
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
                # Detect frequencies + rank MI on the shared subsample (native gather, no whole-frame copy -- the
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
    # 2026-06-09 backlog #11 — HINGE / piecewise-linear change-point basis stage.
    # Independent opt-in via ``fe_hinge_enable`` (does NOT require
    # ``fe_hybrid_orth_enable``): captures a SLOPE CHANGE at a data-dependent
    # threshold ``y = a*x + b*max(x-tau,0)`` (pricing tiers / dose-response /
    # saturation) that the catalog cannot -- ``numeric_rounding`` is piecewise-
    # CONSTANT, the cubic B-spline rounds off a sharp kink at its fixed quantile
    # knots, and orth-poly needs a high degree + rings (Gibbs) around the kink.
    # The breakpoint ``tau`` is detected by scanning inner-quantile cuts for the
    # max 2-segment-SSE drop, HELD-OUT-validated on the ``%3`` stride slice (the
    # 2-segment fit must beat plain linear OOS) so a chance kink / pure noise
    # admits no hinge. Emitted ``relu(x-tau)`` / ``relu(tau-x)`` legs carry a
    # genuinely different LINEAR shape from raw x, so they clear the standard
    # MI-uplift gate (unlike the MI-invariant isotonic / RankGauss). Recipes
    # (``hinge_basis``) store only ``{tau, side}`` -- no y -- so replay is the
    # pure function ``np.maximum(x-tau,0)``, leak-free. On a monotone target a
    # hinge can be near-collinear with raw x -> the downstream cross-stage
    # Spearman dedup drops it (no duplicate columns survive).
    if bool(getattr(self, "fe_hinge_enable", False)) and _fe_budget_ok():
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._hinge_basis_fe import hybrid_hinge_fe_with_recipes
            # The hinge detector + admission are REGRESSION-style (2-segment
            # SSE breakpoint search + held-out incremental linear-R^2 gate),
            # so they want the RAW continuous y -- NOT the qcut-to-10-bins
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
                # DEFERRED MATERIALISATION (2026-06-09): the hinge legs are a
                # TERMINAL univariate linear-usability stage -- they must NOT
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
    if bool(getattr(self, "fe_hybrid_orth_triplet_enable", False)) or _gbm_seeded_triplet_names:
        # Format-agnostic since the matrix-native FE seam: the isinstance(X, pd.DataFrame) skip-guard is gone -- the family
        # runs on polars/pandas alike (subsample decision + native replay via fe_decide_on_subsample / _fe_frame_ops).
        try:
            from .._orthogonal_triplet_fe import (
                hybrid_orth_mi_triplet_fe_with_recipes,
            )
            from .._fe_frame_ops import fe_is_numeric_col, fe_append_columns, fe_extract_columns

            _y_for_triplet = _y_np
            if _y_for_triplet.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_triplet).size)
                if _n_unique <= 32:
                    _y_for_triplet = _y_for_triplet.astype(np.int64)
                else:
                    try:
                        _y_for_triplet = pd.qcut(
                            _y_for_triplet, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_triplet = _y_for_triplet.astype(np.int64)
            # Triplet seed pool is restricted to RAW columns -- never
            # the previously-appended hybrid/extra-basis columns,
            # because those are themselves products of source cols and
            # would invalidate the 3-way-interaction interpretation
            # AND create recipes whose src_names reference engineered
            # columns absent at transform time (KeyError on replay).
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _t_cols: list | None = None
            if getattr(self, "factors_names_to_use", None):
                _t_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _t_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            # The triplet stage applies polynomial (Hermite/Legendre) basis transforms that require numeric input; a string / categorical column ('a_1', ...) raises
            # "could not convert string to float" and the broad guard below would then silently drop the ENTIRE triplet stage. Restrict the seed pool to numeric columns
            # (categoricals are handled by the dedicated categorical-encoding FE stages instead).
            _t_cols = [c for c in _t_cols if fe_is_numeric_col(X, c)]
            _t_max_degree = int(getattr(self, "fe_hybrid_orth_triplet_max_degree", 1))
            _t_seed_k = int(getattr(self, "fe_hybrid_orth_triplet_seed_k", 4))
            _t_top_count = int(getattr(self, "fe_hybrid_orth_triplet_top_count", 2))
            _t_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _t_degrees = tuple(int(d) for d in getattr(self, "fe_hybrid_orth_degrees", (2, 3)))
            _t_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _X_before_triplet_cols = list(X.columns)
            # Forward the GBM seeder's order-3-floored explicit triples (raw column-name
            # legs) so the triplet stage enumerates EXACTLY the zero-marginal 3-way needle
            # the univariate seed_k never ranks; the per-triplet uplift/abs-MI gates still
            # filter. Restrict to legs present + numeric in the current X.
            _explicit_triplets = None
            if _gbm_seeded_triplet_names:
                _xcols = set(X.columns)
                _explicit_triplets = [tr for tr in _gbm_seeded_triplet_names if all((c in _xcols and fe_is_numeric_col(X, c)) for c in tr)] or None
            # When the triplet stage runs SOLELY because the GBM seeder forwarded explicit
            # triples (the legacy univariate-seeded triplet path is OFF), SUPPRESS the
            # stage-1 univariate hybrid (``top_k=0``): we want ONLY the seeded 3-way cross
            # features, not univariate transforms of the seeded operands -- on a pure-noise
            # frame the seeded noise triples' univariate stage would otherwise engineer a
            # spurious univariate Fourier/poly on a noise operand (a noise admission). When
            # the user ALSO enabled the legacy triplet path, keep their univariate budget.
            _t_top_k_eff = _t_top_k
            if _explicit_triplets is not None and not bool(getattr(self, "fe_hybrid_orth_triplet_enable", False)):
                _t_top_k_eff = 0
            X_t, _t_uni_sc, _t_triplet_sc, _t_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_triplet_fe_with_recipes,
                X,
                _y_for_triplet,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_t_cols,
                degrees=_t_degrees,
                basis=_t_basis,
                top_k=_t_top_k_eff,
                triplet_max_degree=_t_max_degree,
                top_triplet_seed_k=_t_seed_k,
                top_triplet_count=_t_top_count,
                explicit_triplets=_explicit_triplets,
            )
            _t_appended = [c for c in X_t.columns if c not in _X_before_triplet_cols]
            # Only keep TRUE triplet columns (3 legs joined by '*');
            # the wrapper may also pass univariate winners through
            # which the master hybrid stage already handles when
            # enabled. Filtering here avoids double-appending the
            # same univariate winner.
            _t_triplet_only = [c for c in _t_appended if c.split("__", 1)[0].count("*") == 2]
            if _t_triplet_only:
                # Append only triplet columns onto the (possibly already
                # hybrid-augmented) X. ``hybrid_orth_features_`` was
                # unconditionally seeded to [] at the top of this fn.
                X = fe_append_columns(X, fe_extract_columns(X_t, _t_triplet_only))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_t_triplet_only)
                # ``_hybrid_orth_pre_recipes`` is unconditionally
                # initialised earlier in this function (line ~245); the
                # triplet stage shares the same dict so its recipes
                # merge into ``_engineered_recipes_`` at end-of-fit via
                # the existing remap.
                _kept = set(_t_triplet_only)
                for _r in _t_recipes:
                    if _r.name in _kept:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth triplet: appended %d " "engineered column(s): %s",
                        len(_t_triplet_only),
                        _t_triplet_only[:8],
                    )
        except Exception as _t_exc:
            logger.warning(
                "MRMR.fit hybrid_orth triplet FE raised %s: %s; " "continuing without triplet-FE columns.",
                type(_t_exc).__name__,
                _t_exc,
            )
    # 2026-06-01 Layer 77 — QUADRUPLET (4-way) cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable): captures
    # genuine 4-way interactions like 4-way XOR (every triplet marginal MI
    # is zero by symmetry, only the He_1^4 cell carries signal) and
    # revenue = price*qty*count*discount. O(seed_k^4 * deg^4) candidate
    # count is bounded by seed_k=4 default. Recipes
    # (``orth_quadruplet_cross``) replay from X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_quadruplet_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_quadruplet_fe import (
                hybrid_orth_mi_quadruplet_fe_with_recipes,
            )
            from .._fe_frame_ops import fe_is_numeric_col, fe_append_columns, fe_extract_columns

            _y_for_quad = _y_np
            if _y_for_quad.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_quad).size)
                if _n_unique <= 32:
                    _y_for_quad = _y_for_quad.astype(np.int64)
                else:
                    try:
                        _y_for_quad = pd.qcut(
                            _y_for_quad, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_quad = _y_for_quad.astype(np.int64)
            # Restrict the seed pool to RAW source columns -- engineered
            # columns from prior stages would create recipes whose
            # src_names reference an engineered column absent at
            # transform time (KeyError on replay).
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _q_cols: list | None = None
            if getattr(self, "factors_names_to_use", None):
                _q_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _q_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            # Numeric-only seed pool: the quadruplet stage applies the same polynomial basis transforms as the triplet stage, so a string / categorical column would raise
            # "could not convert string to float" and the broad guard below would silently drop the whole quadruplet stage. Categoricals are handled by the dedicated cat FE stages.
            _q_cols = [c for c in _q_cols if fe_is_numeric_col(X, c)]
            _q_max_degree = int(getattr(self, "fe_hybrid_orth_quadruplet_max_degree", 1))
            _q_seed_k = int(getattr(self, "fe_hybrid_orth_quadruplet_seed_k", 4))
            _q_top_count = int(getattr(self, "fe_hybrid_orth_quadruplet_top_count", 2))
            _q_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _q_degrees = tuple(int(d) for d in getattr(self, "fe_hybrid_orth_degrees", (2, 3)))
            _q_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _X_before_quad_cols = list(X.columns)
            X_q, _q_uni_sc, _q_quad_sc, _q_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_quadruplet_fe_with_recipes,
                X,
                _y_for_quad,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_q_cols,
                degrees=_q_degrees,
                basis=_q_basis,
                top_k=_q_top_k,
                quadruplet_max_degree=_q_max_degree,
                top_quadruplet_seed_k=_q_seed_k,
                top_quadruplet_count=_q_top_count,
            )
            _q_appended = [c for c in X_q.columns if c not in _X_before_quad_cols]
            # Only keep TRUE quadruplet columns (4 legs joined by '*');
            # the wrapper may also pass univariate winners through which
            # the master hybrid stage already handles when enabled.
            _q_quad_only = [c for c in _q_appended if c.split("__", 1)[0].count("*") == 3]
            if _q_quad_only:
                X = fe_append_columns(X, fe_extract_columns(X_q, _q_quad_only))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_q_quad_only)
                _kept = set(_q_quad_only)
                for _r in _q_recipes:
                    if _r.name in _kept:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth quadruplet: appended %d " "engineered column(s): %s",
                        len(_q_quad_only),
                        _q_quad_only[:8],
                    )
        except Exception as _q_exc:
            logger.warning(
                "MRMR.fit hybrid_orth quadruplet FE raised %s: %s; " "continuing without quadruplet-FE columns.",
                type(_q_exc).__name__,
                _q_exc,
            )
    # 2026-06-01 Layer 78 — ADAPTIVE-ARITY cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, the stage enumerates arity 2..max_arity per seed tuple and
    # keeps ONLY the winning arity per maximal signal set (a higher arity
    # is emitted iff its MI strictly beats every lower-arity prefix).
    # Recipes route to the per-arity Layer 22 / 56 / 77 builders.
    if bool(getattr(self, "fe_hybrid_orth_adaptive_arity_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_adaptive_arity_fe import (
                hybrid_orth_mi_adaptive_arity_fe_with_recipes,
            )

            _y_for_aa = _y_np
            if _y_for_aa.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_aa).size)
                if _n_unique <= 32:
                    _y_for_aa = _y_for_aa.astype(np.int64)
                else:
                    try:
                        _y_for_aa = pd.qcut(
                            _y_for_aa, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_aa = _y_for_aa.astype(np.int64)
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _aa_cols: list | None = None
            if getattr(self, "factors_names_to_use", None):
                _aa_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _aa_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            # The orthogonal/polynomial FE converts operands to float; drop non-numeric columns (raw cat / string,
            # e.g. 'B') so it doesn't raise "could not convert string to float" and silently lose the whole FE pass.
            _aa_cols = _orth_fe_numeric_cols(X, _aa_cols)
            _aa_max_arity = int(getattr(self, "fe_hybrid_orth_adaptive_arity_max_arity", 3))
            _aa_max_degree = int(getattr(self, "fe_hybrid_orth_adaptive_arity_max_degree", 1))
            _aa_seed_k = int(getattr(self, "fe_hybrid_orth_adaptive_arity_seed_k", 4))
            _aa_top_count = int(getattr(self, "fe_hybrid_orth_adaptive_arity_top_count", 3))
            _aa_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _aa_degrees = tuple(int(d) for d in getattr(self, "fe_hybrid_orth_degrees", (2, 3)))
            _aa_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _X_before_aa_cols = list(X.columns)
            X_aa, _aa_uni_sc, _aa_adapt_sc, _aa_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_adaptive_arity_fe_with_recipes,
                X,
                _y_for_aa,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_aa_cols,
                degrees=_aa_degrees,
                basis=_aa_basis,
                top_k=_aa_top_k,
                seed_k=_aa_seed_k,
                max_arity=_aa_max_arity,
                max_degree=_aa_max_degree,
                top_count=_aa_top_count,
            )
            _aa_appended = [c for c in X_aa.columns if c not in _X_before_aa_cols]
            # Only keep TRUE cross columns (arity >= 2 -- one or more '*').
            _aa_cross_only = [c for c in _aa_appended if c.split("__", 1)[0].count("*") >= 1]
            if _aa_cross_only:
                X = fe_append_columns(X, fe_extract_columns(X_aa, _aa_cross_only))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_aa_cross_only)
                _kept_aa = set(_aa_cross_only)
                for _r in _aa_recipes:
                    if _r.name in _kept_aa:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth adaptive-arity: appended %d " "engineered column(s): %s",
                        len(_aa_cross_only),
                        _aa_cross_only[:8],
                    )
        except Exception as _aa_exc:
            logger.warning(
                "MRMR.fit hybrid_orth adaptive-arity FE raised %s: %s; " "continuing without adaptive-arity-FE columns.",
                type(_aa_exc).__name__,
                _aa_exc,
            )
    # 2026-05-31 Layer 57 — ADAPTIVE PER-COLUMN DEGREE FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, for each source column we evaluate every degree in
    # ``fe_hybrid_orth_adaptive_degree_range`` and emit ONLY the argmax-MI
    # degree (if it clears the per-col uplift gate). Recipe kind reuses
    # ``orth_univariate`` -- replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_adaptive_degree_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_adaptive_degree_fe import (
                hybrid_orth_mi_adaptive_degree_fe_with_recipes,
            )

            _y_for_adapt = _y_np
            if _y_for_adapt.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_adapt).size)
                if _n_unique <= 32:
                    _y_for_adapt = _y_for_adapt.astype(np.int64)
                else:
                    try:
                        _y_for_adapt = pd.qcut(
                            _y_for_adapt, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_adapt = _y_for_adapt.astype(np.int64)
            # Restrict the seed pool to RAW source columns -- engineered
            # columns from prior stages would create recipes whose
            # src_names reference an engineered column absent at
            # transform time (KeyError on replay).
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _ad_cols: list | None = None
            if getattr(self, "factors_names_to_use", None):
                _ad_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _ad_cols = [
                    c for c in X.columns
                    if c not in _hybrid_already_appended
                ]
            _ad_range = tuple(int(d) for d in getattr(
                self, "fe_hybrid_orth_adaptive_degree_range", (1, 2, 3, 4, 5, 6),
            ))
            _ad_min_uplift = float(getattr(
                self, "fe_hybrid_orth_adaptive_degree_min_uplift", 1.05,
            ))
            _ad_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _X_before_adaptive_cols = list(X.columns)
            X_ad, _ad_scores, _ad_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_adaptive_degree_fe_with_recipes,
                X,
                _y_for_adapt,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_ad_cols,
                degree_range=_ad_range,
                basis=_ad_basis,
                min_uplift=_ad_min_uplift,
            )
            _ad_appended = [c for c in X_ad.columns if c not in _X_before_adaptive_cols]
            if _ad_appended:
                X = fe_append_columns(X, fe_extract_columns(X_ad, _ad_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ad_appended)
                # Merge into the same recipe dict used by the master
                # hybrid stage so the end-of-fit remap into
                # ``_engineered_recipes_`` picks it up.
                for _r in _ad_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth adaptive-degree: appended " "%d engineered column(s): %s",
                        len(_ad_appended),
                        _ad_appended[:8],
                    )
        except Exception as _ad_exc:
            logger.warning(
                "MRMR.fit hybrid_orth adaptive-degree FE raised %s: %s; " "continuing without adaptive-degree columns.",
                type(_ad_exc).__name__,
                _ad_exc,
            )
    # 2026-05-31 Layer 58 — CONDITIONAL BASIS ROUTING FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, we try every (pre_transform, basis, degree) cell per source
    # column and keep the MI-uplift winner; global top-K appended. Recipe
    # kind reuses ``orth_univariate`` (extra carries ``pre_transform``);
    # replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_conditional_routing_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_routing_fe import (
                hybrid_orth_mi_conditional_routing_fe_with_recipes,
            )

            _y_for_route = _y_np
            if _y_for_route.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_route).size)
                if _n_unique <= 32:
                    _y_for_route = _y_for_route.astype(np.int64)
                else:
                    try:
                        _y_for_route = pd.qcut(
                            _y_for_route, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_route = _y_for_route.astype(np.int64)
            # Restrict the seed pool to RAW source columns -- engineered
            # columns from prior stages would create recipes whose
            # src_names reference an engineered column absent at
            # transform time.
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            _rt_cols: list | None = None
            if getattr(self, "factors_names_to_use", None):
                _rt_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _rt_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _rt_top_k = int(
                getattr(
                    self,
                    "fe_hybrid_orth_conditional_routing_top_k",
                    5,
                )
            )
            _rt_min_uplift = float(
                getattr(
                    self,
                    "fe_hybrid_orth_conditional_routing_min_uplift",
                    1.10,
                )
            )
            _rt_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_conditional_routing_degrees",
                    (2, 3),
                )
            )
            _X_before_routing_cols = list(X.columns)
            X_rt, _rt_scores, _rt_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_conditional_routing_fe_with_recipes,
                X,
                _y_for_route,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_rt_cols,
                degrees=_rt_degrees,
                top_k=_rt_top_k,
                min_uplift=_rt_min_uplift,
            )
            _rt_appended = [c for c in X_rt.columns if c not in _X_before_routing_cols]
            if _rt_appended:
                X = fe_append_columns(X, fe_extract_columns(X_rt, _rt_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rt_appended)
                for _r in _rt_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth conditional-routing: appended " "%d engineered column(s): %s",
                        len(_rt_appended),
                        _rt_appended[:8],
                    )
        except Exception as _rt_exc:
            logger.warning(
                "MRMR.fit hybrid_orth conditional-routing FE raised %s: %s; " "continuing without conditional-routing columns.",
                type(_rt_exc).__name__,
                _rt_exc,
            )
    # 2026-05-31 Layer 59 — DIFF-BASIS FE for highly-correlated source pairs.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, the auto-pair detector flags every pair with |Pearson corr| >=
    # threshold, computes the residual diff, and evaluates a basis expansion
    # per requested degree; top-K winners appended. Recipe kind
    # ``orth_diff_basis``; replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_diff_basis_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_diff_basis_fe import (
                hybrid_orth_mi_diff_basis_fe_with_recipes,
            )

            _y_for_diff = _y_np
            if _y_for_diff.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_diff).size)
                if _n_unique <= 32:
                    _y_for_diff = _y_for_diff.astype(np.int64)
                else:
                    try:
                        _y_for_diff = pd.qcut(
                            _y_for_diff, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_diff = _y_for_diff.astype(np.int64)
            # Restrict the seed pool to RAW source columns -- engineered
            # columns from prior stages would create recipes whose
            # src_names reference an engineered column absent at transform.
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _df_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _df_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _df_corr = float(
                getattr(
                    self,
                    "fe_hybrid_orth_diff_basis_corr_threshold",
                    0.7,
                )
            )
            _df_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_diff_basis_degrees",
                    (1, 2, 3),
                )
            )
            _df_top_k = int(
                getattr(
                    self,
                    "fe_hybrid_orth_diff_basis_top_k",
                    3,
                )
            )
            _X_before_diff_cols = list(X.columns)
            X_df, _df_scores, _df_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_diff_basis_fe_with_recipes,
                X,
                _y_for_diff,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_df_cols,
                degrees=_df_degrees,
                pair_corr_threshold=_df_corr,
                top_k=_df_top_k,
            )
            _df_appended = [c for c in X_df.columns if c not in _X_before_diff_cols]
            if _df_appended:
                X = fe_append_columns(X, fe_extract_columns(X_df, _df_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_df_appended)
                for _r in _df_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth diff-basis: appended %d " "engineered column(s): %s",
                        len(_df_appended),
                        _df_appended[:8],
                    )
        except Exception as _df_exc:
            logger.warning(
                "MRMR.fit hybrid_orth diff-basis FE raised %s: %s; " "continuing without diff-basis columns.",
                type(_df_exc).__name__,
                _df_exc,
            )
    # 2026-05-31 Layer 61 — PER-CLUSTER SHARED-BASIS FE. Independent opt-in
    # (does NOT require fe_hybrid_orth_enable). When active, an internal
    # correlation-based cluster detector finds connected components of the
    # |Pearson corr| >= corr_threshold graph among raw numeric columns, then
    # for each cluster reduces to one aggregate column via the configured
    # aggregator (mean_z / median_z / pc1) and evaluates basis_d on the
    # aggregate. The shared-basis path complements Layer 21 (per-member
    # basis) and Layer 7 cluster_aggregate (swaps cluster to PC1/mean_z as a
    # new raw feature WITHOUT a basis expansion). Recipe kind
    # ``orth_cluster_basis``; replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_cluster_basis_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_cluster_basis_fe import (
                hybrid_orth_mi_cluster_basis_fe_with_recipes,
            )
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
            # W6: record abs-MAD floor kills in the cluster-basis stage into
            # the FE rejection ledger (pure-record; selection unchanged).
            _cb_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _cb_reject_sink(**_kw):
                _record_fe_rejection(self, step=_cb_step, **_kw)

            _y_for_cb = _y_np
            if _y_for_cb.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_cb).size)
                if _n_unique <= 32:
                    _y_for_cb = _y_for_cb.astype(np.int64)
                else:
                    try:
                        _y_for_cb = pd.qcut(
                            _y_for_cb, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_cb = _y_for_cb.astype(np.int64)
            # Restrict to RAW source columns -- engineered columns from
            # prior stages would create recipes whose src_names reference
            # an engineered column absent at transform.
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _cb_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _cb_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _cb_aggregator = str(
                getattr(
                    self,
                    "fe_hybrid_orth_cluster_basis_aggregator",
                    "mean_z",
                )
            )
            _cb_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_cluster_basis_degrees",
                    (2, 3),
                )
            )
            _cb_top_k = int(
                getattr(
                    self,
                    "fe_hybrid_orth_cluster_basis_top_k",
                    3,
                )
            )
            # Cluster detection reuses the diff-basis corr threshold as a
            # sensible default (same calibration: 0.7 is the reflection-
            # cluster floor). We deliberately do NOT share the same
            # constructor argument so callers can tune diff-basis and
            # cluster-basis independently.
            _cb_corr = float(
                getattr(
                    self,
                    "fe_hybrid_orth_diff_basis_corr_threshold",
                    0.7,
                )
            )
            _X_before_cb_cols = list(X.columns)
            X_cb, _cb_scores, _cb_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_cluster_basis_fe_with_recipes,
                X,
                _y_for_cb,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_cb_cols,
                aggregator=_cb_aggregator,
                degrees=_cb_degrees,
                corr_threshold=_cb_corr,
                top_k=_cb_top_k,
                reject_sink=_cb_reject_sink,
            )
            _cb_appended = [c for c in X_cb.columns if c not in _X_before_cb_cols]
            if _cb_appended:
                X = fe_append_columns(X, fe_extract_columns(X_cb, _cb_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cb_appended)
                for _r in _cb_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth cluster-basis: appended %d " "engineered column(s): %s",
                        len(_cb_appended),
                        _cb_appended[:8],
                    )
        except Exception as _cb_exc:
            logger.warning(
                "MRMR.fit hybrid_orth cluster-basis FE raised %s: %s; " "continuing without cluster-basis columns.",
                type(_cb_exc).__name__,
                _cb_exc,
            )
    # 2026-05-31 Layer 62 — BOOTSTRAP-STABLE MI ranking for the hybrid
    # orth-poly FE (independent opt-in; does NOT require
    # fe_hybrid_orth_enable). Replaces the Layer 21 point-estimate MI gate
    # with a lower-confidence-bound (mean - 1.96 * std) across n_boot
    # bootstrap subsamples drawn jointly at sample_fraction. The
    # engineered columns are bit-equal to Layer 21 -- only the SELECTION
    # changes -- so recipes reuse the ``orth_univariate`` kind and replay
    # is shared. Restrict to RAW columns to avoid recipes referencing
    # already-engineered columns absent at transform.
    if bool(getattr(self, "fe_hybrid_orth_bootstrap_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_bootstrap_mi_fe import (
                hybrid_orth_mi_bootstrap_fe_with_recipes,
            )

            _y_for_boot = _y_np
            if _y_for_boot.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_boot).size)
                if _n_unique <= 32:
                    _y_for_boot = _y_for_boot.astype(np.int64)
                else:
                    try:
                        _y_for_boot = pd.qcut(
                            _y_for_boot, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_boot = _y_for_boot.astype(np.int64)
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _boot_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _boot_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            # Orthogonal/polynomial bootstrap FE converts operands to float; a raw categorical / string column would raise
            # "could not convert string to float" and (via the broad except below) silently drop the entire bootstrap-stable pass.
            # Scope to numeric/raw columns the same way the conditional-FE families do, instead of swallowing the failure.
            _boot_cols = _orth_fe_numeric_cols(X, _boot_cols)
            _boot_degrees = tuple(int(d) for d in getattr(
                self, "fe_hybrid_orth_degrees", (2, 3),
            ))
            _boot_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _boot_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _boot_n = int(getattr(
                self, "fe_hybrid_orth_bootstrap_n_boot", 10,
            ))
            _boot_frac = float(getattr(
                self, "fe_hybrid_orth_bootstrap_sample_fraction", 0.8,
            ))
            _boot_seed = int(getattr(self, "random_seed", 0) or 0)
            _X_before_boot_cols = list(X.columns)
            X_boot, _boot_scores, _boot_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_bootstrap_fe_with_recipes,
                X,
                _y_for_boot,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_boot_cols,
                degrees=_boot_degrees,
                basis=_boot_basis,
                top_k=_boot_top_k,
                n_boot=_boot_n,
                sample_fraction=_boot_frac,
                seed=_boot_seed,
            )
            _boot_appended = [c for c in X_boot.columns if c not in _X_before_boot_cols]
            if _boot_appended:
                X = fe_append_columns(X, fe_extract_columns(X_boot, _boot_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_boot_appended)
                for _r in _boot_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth bootstrap-stable: appended " "%d engineered column(s): %s",
                        len(_boot_appended),
                        _boot_appended[:8],
                    )
        except Exception as _boot_exc:
            logger.warning(
                "MRMR.fit hybrid_orth bootstrap-stable FE raised %s: %s; " "continuing without bootstrap-stable columns.",
                type(_boot_exc).__name__,
                _boot_exc,
            )
    # 2026-05-31 Layer 63 — THREE-GATE + K-fold OOF MI ranking for the
    # hybrid orth-poly FE (independent opt-in; does NOT require
    # fe_hybrid_orth_enable). Layer 21 ranks engineered columns with a
    # plug-in MI estimate biased upward by ``(K-1) / (2n)``; the absolute
    # floor sometimes admits noise-driven candidates the bias inflated
    # past it. Layer 63 scores with stratified K-fold OOF MI (train-fitted
    # bin edges applied to held-out fold) and adds a Gate 3:
    # ``CMI(candidate; y | current_support) >= cmi_min`` which kills
    # duplicate-signal candidates (``x__T2`` after ``x__He2`` is already
    # selected). When ``current_support`` is empty Gate 3 is skipped --
    # marginal MI from Gate 1 already covers that case. Engineered VALUES
    # are bit-equal to Layer 21 so recipes reuse the ``orth_univariate``
    # kind and replay is shared infrastructure.
    if bool(getattr(self, "fe_hybrid_orth_three_gate_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_three_gate_mi_fe import (
                hybrid_orth_mi_three_gate_fe_with_recipes,
            )

            _y_for_tg = _y_np
            if _y_for_tg.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_tg).size)
                if _n_unique <= 32:
                    _y_for_tg = _y_for_tg.astype(np.int64)
                else:
                    try:
                        _y_for_tg = pd.qcut(
                            _y_for_tg, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_tg = _y_for_tg.astype(np.int64)
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _tg_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _tg_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            # Orthogonal/polynomial FE is numeric-only; drop non-numeric cols (raw cat / string) before the float
            # conversion, else it raises "could not convert string to float" and the whole FE pass is dropped.
            _tg_cols = _orth_fe_numeric_cols(X, _tg_cols)
            _tg_degrees = tuple(int(d) for d in getattr(
                self, "fe_hybrid_orth_degrees", (2, 3),
            ))
            _tg_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _tg_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _tg_n_folds = int(getattr(
                self, "fe_hybrid_orth_three_gate_n_folds", 5,
            ))
            _tg_cmi_min = float(getattr(
                self, "fe_hybrid_orth_three_gate_cmi_min", 0.001,
            ))
            _tg_seed = int(getattr(self, "random_seed", 0) or 0)
            # Build current_support from columns already appended by
            # earlier hybrid stages (cluster-basis / bootstrap /
            # Layer 21). When the support is empty (the common case
            # in single-stage runs) Gate 3 is skipped inside the
            # callee, which preserves Layer 21 behaviour at the
            # selection level (sans the OOF re-ranking on Gate 1/2).
            _tg_support_cols = [c for c in _hybrid_already_appended if c in X.columns]
            _X_before_tg_cols = list(X.columns)
            # The current_support sub-frame is READ-only (``.empty`` / ``.shape`` / per-column ``.to_numpy()`` for the
            # CMI bins). Build it from whatever pandas frame the subsample funnel hands the callee (the subsample block
            # or, on the small-frame fallback, the full frame) so support rows always align with the decision rows.
            def _tg_run(_Xs, _ys, **_kw):
                _cs = _Xs[_tg_support_cols] if _tg_support_cols else None
                return hybrid_orth_mi_three_gate_fe_with_recipes(_Xs, _ys, _cs, **_kw)
            X_tg, _tg_scores, _tg_recipes = fe_decide_on_subsample(
                _tg_run,
                X, _y_for_tg,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_tg_cols,
                degrees=_tg_degrees,
                basis=_tg_basis,
                top_k=_tg_top_k,
                cmi_min=_tg_cmi_min,
                n_folds=_tg_n_folds,
                seed=_tg_seed,
            )
            _tg_appended = [c for c in X_tg.columns if c not in _X_before_tg_cols]
            if _tg_appended:
                X = fe_append_columns(X, fe_extract_columns(X_tg, _tg_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_tg_appended)
                for _r in _tg_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth three-gate: appended " "%d engineered column(s): %s",
                        len(_tg_appended),
                        _tg_appended[:8],
                    )
        except Exception as _tg_exc:
            logger.warning(
                "MRMR.fit hybrid_orth three-gate FE raised %s: %s; " "continuing without three-gate columns.",
                type(_tg_exc).__name__,
                _tg_exc,
            )
    # 2026-05-31 Layer 65 — KSG / k-NN MI ranking for the hybrid orth-poly
    # FE (independent opt-in; does NOT require fe_hybrid_orth_enable).
    # Replaces the Layer 21 plug-in quantile-binned MI estimator with the
    # Kraskov-Stoegbauer-Grassberger k-NN MI estimator via sklearn's
    # ``mutual_info_classif`` (Ross 2014 mixed-KSG for discrete y). The
    # engineered columns are bit-equal to Layer 21 -- only the SCORING
    # (and therefore the selection) changes -- so recipes reuse the
    # ``orth_univariate`` kind and replay is shared infrastructure.
    if bool(getattr(self, "fe_hybrid_orth_ksg_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_ksg_mi_fe import (
                hybrid_orth_mi_ksg_fe_with_recipes,
            )

            _y_for_ksg = _y_np
            if _y_for_ksg.dtype.kind in "fc":
                _n_unique = int(np.unique(_y_for_ksg).size)
                if _n_unique <= 32:
                    _y_for_ksg = _y_for_ksg.astype(np.int64)
                else:
                    try:
                        _y_for_ksg = pd.qcut(
                            _y_for_ksg, q=10, labels=False, duplicates="drop",
                        ).astype(np.int64)
                    except Exception:
                        _y_for_ksg = _y_for_ksg.astype(np.int64)
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _ksg_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _ksg_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _ksg_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _ksg_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
            _ksg_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _ksg_n_neighbors = int(getattr(
                self, "fe_hybrid_orth_ksg_n_neighbors", 3,
            ))
            _ksg_min_uplift = float(getattr(
                self, "fe_hybrid_orth_ksg_min_uplift", 0.95,
            ))
            _ksg_min_abs_mi_frac = float(getattr(
                self, "fe_hybrid_orth_ksg_min_abs_mi_frac", 0.05,
            ))
            _ksg_seed = int(getattr(self, "random_seed", 0) or 0)
            _X_before_ksg_cols = list(X.columns)
            X_ksg, _ksg_scores, _ksg_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_ksg_fe_with_recipes,
                X,
                _y_for_ksg,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_ksg_cols,
                degrees=_ksg_degrees,
                basis=_ksg_basis,
                top_k=_ksg_top_k,
                min_uplift=_ksg_min_uplift,
                min_abs_mi_frac=_ksg_min_abs_mi_frac,
                n_neighbors=_ksg_n_neighbors,
                random_state=_ksg_seed,
            )
            _ksg_appended = [c for c in X_ksg.columns if c not in _X_before_ksg_cols]
            if _ksg_appended:
                X = fe_append_columns(X, fe_extract_columns(X_ksg, _ksg_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ksg_appended)
                for _r in _ksg_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth KSG-MI: appended " "%d engineered column(s): %s",
                        len(_ksg_appended),
                        _ksg_appended[:8],
                    )
        except Exception as _ksg_exc:
            logger.warning(
                "MRMR.fit hybrid_orth KSG-MI FE raised %s: %s; " "continuing without KSG-MI columns.",
                type(_ksg_exc).__name__,
                _ksg_exc,
            )
    # 2026-06-01 Layer 66 — COPULA-MI ranking for the hybrid orth-poly FE
    # (independent opt-in; does NOT require fe_hybrid_orth_enable). Each
    # variable is rank-transformed to a uniform on (0, 1) before MI is
    # estimated, so the score is INVARIANT under any strictly-monotone
    # transform of either variable. Wins on heavy-tailed / skewed signals
    # where the plug-in's qcut on raw values piles tail observations into
    # one bin and hides genuine dependence. Engineered VALUES bit-equal to
    # Layer 21 -> recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_copula_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_copula_mi_fe import (
                hybrid_orth_mi_copula_fe_with_recipes,
            )

            _y_for_copula = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _copula_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _copula_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _copula_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _copula_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _copula_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _copula_n_bins = int(getattr(
                self, "fe_hybrid_orth_copula_n_bins", 20,
            ))
            # Copula MI on rank-uniformised data is less biased than the
            # plug-in on raw values (the rank transform flattens the
            # marginal so the bias-correcting Miller-Madow term works on
            # a uniform target); the gates calibrated for Layer 21 plug-in
            # (1.05 / 0.1) are too tight here -- copula MI lift on a
            # cubic-in-x signal is typically 1.00-1.05x because rank(x)
            # already captures the monotone structure, leaving only the
            # non-monotone residual to lift. 0.95 / 0.05 matches the
            # Layer 65 KSG calibration for the same reason.
            _copula_min_uplift = 0.95
            _copula_min_abs_mi_frac = 0.05
            _X_before_copula_cols = list(X.columns)
            X_copula, _copula_scores, _copula_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_copula_fe_with_recipes,
                X,
                _y_for_copula,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_copula_cols,
                degrees=_copula_degrees,
                basis=_copula_basis,
                top_k=_copula_top_k,
                min_uplift=_copula_min_uplift,
                min_abs_mi_frac=_copula_min_abs_mi_frac,
                n_bins=_copula_n_bins,
            )
            _copula_appended = [c for c in X_copula.columns if c not in _X_before_copula_cols]
            if _copula_appended:
                X = fe_append_columns(X, fe_extract_columns(X_copula, _copula_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_copula_appended)
                for _r in _copula_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth copula-MI: appended " "%d engineered column(s): %s",
                        len(_copula_appended),
                        _copula_appended[:8],
                    )
        except Exception as _copula_exc:
            logger.warning(
                "MRMR.fit hybrid_orth copula-MI FE raised %s: %s; " "continuing without copula-MI columns.",
                type(_copula_exc).__name__,
                _copula_exc,
            )
    # 2026-06-01 Layer 67 — DISTANCE-CORRELATION ranking for the hybrid
    # orth-poly FE (independent opt-in; does NOT require
    # fe_hybrid_orth_enable). Szekely-Rizzo dCor is the only non-MI
    # dependence measure in the layer family -- ``dCor == 0`` iff X and Y
    # are independent on ANY relationship (Pearson lacks this iff
    # guarantee). Naive dCor is O(n^2); the working sample is capped at
    # n=500 via deterministic random subsample. Engineered VALUES bit-equal
    # to Layer 21 -> recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_dcor_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_dcor_fe import (
                hybrid_orth_mi_dcor_fe_with_recipes,
            )

            _y_for_dcor = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _dcor_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _dcor_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _dcor_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _dcor_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _dcor_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _dcor_n_sample = int(getattr(
                self, "fe_hybrid_orth_dcor_n_sample", 500,
            ))
            # dCor on raw x already captures non-monotone structure
            # (Hermite poly basis tracks the same dependence dCor
            # detects), so engineered/baseline uplift on a single
            # source is typically near 1.0; the 0.95 / 0.05 floor
            # matches the Layer 65 / 66 calibration for the same
            # reason.
            _dcor_min_uplift = 0.95
            _dcor_min_abs_mi_frac = 0.05
            _X_before_dcor_cols = list(X.columns)
            X_dcor, _dcor_scores, _dcor_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_dcor_fe_with_recipes,
                X,
                _y_for_dcor,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_dcor_cols,
                degrees=_dcor_degrees,
                basis=_dcor_basis,
                top_k=_dcor_top_k,
                min_uplift=_dcor_min_uplift,
                min_abs_mi_frac=_dcor_min_abs_mi_frac,
                n_sample=_dcor_n_sample,
                random_state=int(getattr(self, "random_seed", 0) or 0),
            )
            _dcor_appended = [c for c in X_dcor.columns if c not in _X_before_dcor_cols]
            if _dcor_appended:
                X = fe_append_columns(X, fe_extract_columns(X_dcor, _dcor_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_dcor_appended)
                for _r in _dcor_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth dCor: appended %d " "engineered column(s): %s",
                        len(_dcor_appended),
                        _dcor_appended[:8],
                    )
        except Exception as _dcor_exc:
            logger.warning(
                "MRMR.fit hybrid_orth dCor FE raised %s: %s; " "continuing without dCor columns.",
                type(_dcor_exc).__name__,
                _dcor_exc,
            )
    # 2026-06-01 Layer 71 — HSIC ranking for hybrid orth-poly FE
    # (independent opt-in; does NOT require fe_hybrid_orth_enable).
    # Kernel-based dependence measure with the universal HSIC == 0 iff
    # independent guarantee under a characteristic kernel (Gaussian RBF
    # with median-heuristic bandwidth). Complementary to Layer 67 dCor:
    # HSIC operates at a kernel-chosen length SCALE, wins on sharp local
    # non-linearities and high-frequency oscillation. Naive HSIC is
    # O(n^2); the working sample is capped at n=500 via deterministic
    # random subsample. Engineered VALUES bit-equal to Layer 21 ->
    # recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_hsic_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_hsic_fe import (
                hybrid_orth_mi_hsic_fe_with_recipes,
            )

            _y_for_hsic = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _hsic_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _hsic_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _hsic_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _hsic_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _hsic_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _hsic_kernel = str(getattr(
                self, "fe_hybrid_orth_hsic_kernel", "rbf",
            ))
            _hsic_n_sample = int(getattr(
                self, "fe_hybrid_orth_hsic_n_sample", 500,
            ))
            # Same calibration as Layers 65 / 66 / 67: HSIC on raw x
            # already captures non-linear structure (the polynomial
            # basis tracks the same dependence the RBF kernel
            # detects), so engineered/baseline uplift on a single
            # source typically sits near 1.0; 0.95 / 0.05 floor
            # keeps genuine borderline wins.
            _hsic_min_uplift = 0.95
            _hsic_min_abs_mi_frac = 0.05
            _X_before_hsic_cols = list(X.columns)
            X_hsic, _hsic_scores, _hsic_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_hsic_fe_with_recipes,
                X,
                _y_for_hsic,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_hsic_cols,
                degrees=_hsic_degrees,
                basis=_hsic_basis,
                top_k=_hsic_top_k,
                min_uplift=_hsic_min_uplift,
                min_abs_mi_frac=_hsic_min_abs_mi_frac,
                kernel=_hsic_kernel,
                n_sample=_hsic_n_sample,
                random_state=int(getattr(self, "random_seed", 0) or 0),
            )
            _hsic_appended = [c for c in X_hsic.columns if c not in _X_before_hsic_cols]
            if _hsic_appended:
                X = fe_append_columns(X, fe_extract_columns(X_hsic, _hsic_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_hsic_appended)
                for _r in _hsic_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth HSIC: appended %d " "engineered column(s): %s",
                        len(_hsic_appended),
                        _hsic_appended[:8],
                    )
        except Exception as _hsic_exc:
            logger.warning(
                "MRMR.fit hybrid_orth HSIC FE raised %s: %s; " "continuing without HSIC columns.",
                type(_hsic_exc).__name__,
                _hsic_exc,
            )
    # 2026-06-01 Layer 72 — JMIM (Bennasar 2015) redundancy-aware ranking
    # for hybrid orth-poly FE (independent opt-in; does NOT require
    # fe_hybrid_orth_enable). Each engineered candidate is scored by
    # ``min over X_j in S of I((X_cand, X_j); Y)`` where S is the raw
    # source column pool. Selection: same two-gate rule as Layers 65 /
    # 66 / 67 / 71. Engineered VALUES bit-equal to Layer 21 -> recipes
    # reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_jmim_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_jmim_fe import (
                hybrid_orth_mi_jmim_fe_with_recipes,
            )

            _y_for_jmim = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _jmim_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _jmim_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _jmim_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _jmim_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _jmim_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _jmim_n_bins = int(getattr(
                self, "fe_hybrid_orth_jmim_n_bins", 10,
            ))
            # Same calibration as Layers 65 / 66 / 67 / 71: 0.95 /
            # 0.05 floor keeps genuine borderline wins.
            _jmim_min_uplift = 0.95
            _jmim_min_abs_mi_frac = 0.05
            _X_before_jmim_cols = list(X.columns)
            X_jmim, _jmim_scores, _jmim_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_jmim_fe_with_recipes,
                X,
                _y_for_jmim,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_jmim_cols,
                degrees=_jmim_degrees,
                basis=_jmim_basis,
                top_k=_jmim_top_k,
                min_uplift=_jmim_min_uplift,
                min_abs_mi_frac=_jmim_min_abs_mi_frac,
                n_bins=_jmim_n_bins,
            )
            _jmim_appended = [c for c in X_jmim.columns if c not in _X_before_jmim_cols]
            if _jmim_appended:
                X = fe_append_columns(X, fe_extract_columns(X_jmim, _jmim_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_jmim_appended)
                for _r in _jmim_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth JMIM: appended %d " "engineered column(s): %s",
                        len(_jmim_appended),
                        _jmim_appended[:8],
                    )
        except Exception as _jmim_exc:
            logger.warning(
                "MRMR.fit hybrid_orth JMIM FE raised %s: %s; " "continuing without JMIM columns.",
                type(_jmim_exc).__name__,
                _jmim_exc,
            )
    # 2026-06-01 Layer 73 — Total Correlation (Watanabe 1960) multivariate-
    # redundancy ranking for hybrid orth-poly FE (independent opt-in; does
    # NOT require fe_hybrid_orth_enable). Each engineered candidate is
    # scored by the FULL-ORDER joint shared information delta against the
    # current support union with y. Selection: same absolute floor as
    # Layers 65 / 66 / 67 / 71 / 72. Engineered VALUES bit-equal to Layer
    # 21 -> recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_tc_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_total_correlation_fe import (
                hybrid_orth_mi_tc_fe_with_recipes,
            )

            _y_for_tc = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _tc_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _tc_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _tc_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _tc_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _tc_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _tc_n_bins = int(getattr(
                self, "fe_hybrid_orth_tc_n_bins", 10,
            ))
            # Same calibration as Layers 65 / 66 / 67 / 71 / 72: 0.95 /
            # 0.05 floor keeps genuine borderline wins.
            _tc_min_uplift = 0.95
            _tc_min_abs_mi_frac = 0.05
            _X_before_tc_cols = list(X.columns)
            X_tc, _tc_scores, _tc_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_tc_fe_with_recipes,
                X,
                _y_for_tc,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_tc_cols,
                degrees=_tc_degrees,
                basis=_tc_basis,
                top_k=_tc_top_k,
                min_uplift=_tc_min_uplift,
                min_abs_mi_frac=_tc_min_abs_mi_frac,
                n_bins=_tc_n_bins,
            )
            _tc_appended = [c for c in X_tc.columns if c not in _X_before_tc_cols]
            if _tc_appended:
                X = fe_append_columns(X, fe_extract_columns(X_tc, _tc_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_tc_appended)
                for _r in _tc_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth TC: appended %d " "engineered column(s): %s",
                        len(_tc_appended),
                        _tc_appended[:8],
                    )
        except Exception as _tc_exc:
            logger.warning(
                "MRMR.fit hybrid_orth TC FE raised %s: %s; " "continuing without TC columns.",
                type(_tc_exc).__name__,
                _tc_exc,
            )
    # 2026-06-01 Layer 74 — CMIM (Conditional Mutual Information
    # Maximisation, Fleuret 2004) redundancy-aware ranking for hybrid
    # orth-poly FE (independent opt-in; does NOT require
    # fe_hybrid_orth_enable). Each engineered candidate is scored by the
    # WORST-CASE conditional MI against EACH selected support member
    # individually: ``min_j CMI(X_cand; Y | X_j)``. Companion to JMIM
    # (Layer 72): CMIM penalises redundancy via the conditioning
    # operator while JMIM rewards complementarity via the joint MI.
    # Selection: same absolute floor as Layers 65 / 66 / 67 / 71 / 72 /
    # 73. Engineered VALUES bit-equal to Layer 21 -> recipes reuse the
    # ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_cmim_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_cmim_fe import (
                hybrid_orth_mi_cmim_fe_with_recipes,
            )

            _y_for_cmim = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _cmim_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _cmim_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _cmim_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _cmim_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _cmim_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _cmim_n_bins = int(getattr(
                self, "fe_hybrid_orth_cmim_n_bins", 10,
            ))
            # Same calibration as Layers 65 / 66 / 67 / 71 / 72 / 73:
            # 0.95 / 0.05 floor keeps genuine borderline wins.
            _cmim_min_uplift = 0.95
            _cmim_min_abs_mi_frac = 0.05
            _X_before_cmim_cols = list(X.columns)
            X_cmim, _cmim_scores, _cmim_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_cmim_fe_with_recipes,
                X,
                _y_for_cmim,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_cmim_cols,
                degrees=_cmim_degrees,
                basis=_cmim_basis,
                top_k=_cmim_top_k,
                min_uplift=_cmim_min_uplift,
                min_abs_mi_frac=_cmim_min_abs_mi_frac,
                n_bins=_cmim_n_bins,
            )
            _cmim_appended = [c for c in X_cmim.columns if c not in _X_before_cmim_cols]
            if _cmim_appended:
                X = fe_append_columns(X, fe_extract_columns(X_cmim, _cmim_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cmim_appended)
                for _r in _cmim_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth CMIM: appended %d " "engineered column(s): %s",
                        len(_cmim_appended),
                        _cmim_appended[:8],
                    )
        except Exception as _cmim_exc:
            logger.warning(
                "MRMR.fit hybrid_orth CMIM FE raised %s: %s; " "continuing without CMIM columns.",
                type(_cmim_exc).__name__,
                _cmim_exc,
            )
    # 2026-06-01 Layer 68 — PER-COLUMN SCORER AUTO-SELECTION across the
    # Layer 21 / 65 / 66 / 67 scorer family (independent opt-in; does NOT
    # require fe_hybrid_orth_enable). For each engineered column the
    # bootstrap-LCB criterion picks the best scorer in
    # {plug-in, KSG, copula, dCor} and uses ITS LCB for the cross-column
    # ranking + selection. Engineered VALUES bit-equal to Layer 21 ->
    # recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_auto_scorer_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_scorer_auto_fe import (
                hybrid_orth_mi_auto_scorer_fe_with_recipes,
            )

            _y_for_auto = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _auto_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _auto_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _auto_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _auto_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _auto_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _auto_n_boot = int(getattr(
                self, "fe_hybrid_orth_auto_scorer_n_boot", 5,
            ))
            # Same calibration as Layers 65 / 66 / 67: the chosen
            # scorer often captures raw-x dependence as cleanly as
            # the engineered column, so single-source uplift sits
            # near 1.0; the 0.95 / 0.05 floors keep the gate from
            # rejecting genuine wins on a sample-noise tick.
            _auto_min_uplift = 0.95
            _auto_min_abs_mi_frac = 0.05
            _X_before_auto_cols = list(X.columns)
            X_auto, _auto_scores, _auto_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_auto_scorer_fe_with_recipes,
                X,
                _y_for_auto,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_auto_cols,
                degrees=_auto_degrees,
                basis=_auto_basis,
                top_k=_auto_top_k,
                min_uplift=_auto_min_uplift,
                min_abs_mi_frac=_auto_min_abs_mi_frac,
                n_boot=_auto_n_boot,
                random_state=int(getattr(self, "random_seed", 0) or 0),
            )
            _auto_appended = [c for c in X_auto.columns if c not in _X_before_auto_cols]
            if _auto_appended:
                X = fe_append_columns(X, fe_extract_columns(X_auto, _auto_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_auto_appended)
                for _r in _auto_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth auto-scorer: appended " "%d engineered column(s): %s",
                        len(_auto_appended),
                        _auto_appended[:8],
                    )
        except Exception as _auto_exc:
            logger.warning(
                "MRMR.fit hybrid_orth auto-scorer FE raised %s: %s; " "continuing without auto-scorer columns.",
                type(_auto_exc).__name__,
                _auto_exc,
            )
    # 2026-06-01 Layer 69 — ENSEMBLE-OF-SCORERS rank-fusion across the
    # Layer 21 / 65 / 66 / 67 scorer family (independent opt-in; does NOT
    # require fe_hybrid_orth_enable). Each requested scorer ranks every
    # engineered column independently; the per-scorer ranks are fused via
    # ``fe_hybrid_orth_ensemble_aggregator`` (mean_rank / borda_count /
    # reciprocal_rank) and the consensus drives selection. Complementary
    # to Layer 68: ensemble wins on AMBIGUOUS frames where the bootstrap-
    # LCB per-column winner is unstable across seeds. Engineered VALUES
    # bit-equal to Layer 21 -> recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_ensemble_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_scorer_auto_fe import (
                hybrid_orth_mi_ensemble_fe_with_recipes,
            )

            _y_for_ens = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _ens_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _ens_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            _ens_degrees = tuple(
                int(d)
                for d in getattr(
                    self,
                    "fe_hybrid_orth_degrees",
                    (2, 3),
                )
            )
            _ens_basis = str(
                getattr(
                    self,
                    "fe_hybrid_orth_basis",
                    "auto",
                )
            )
            _ens_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _ens_aggregator = str(getattr(
                self, "fe_hybrid_orth_ensemble_aggregator", "mean_rank",
            ))
            _ens_scorers = tuple(getattr(
                self, "fe_hybrid_orth_ensemble_scorers",
                ("plug_in", "ksg", "copula", "dcor", "hsic"),
            ))
            # Same gate calibration as Layers 65 / 66 / 67 / 68: the
            # raw-x dependence is captured by the chosen scorers
            # nearly as cleanly as the engineered column, so the
            # uplift floor sits at 0.95 and the abs MI fraction at
            # 0.05 to keep genuine borderline wins.
            _ens_min_uplift = 0.95
            _ens_min_abs_mi_frac = 0.05
            _X_before_ens_cols = list(X.columns)
            X_ens, _ens_scores, _ens_recipes = fe_decide_on_subsample(
                hybrid_orth_mi_ensemble_fe_with_recipes,
                X,
                _y_for_ens,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_ens_cols,
                degrees=_ens_degrees,
                basis=_ens_basis,
                top_k=_ens_top_k,
                min_uplift=_ens_min_uplift,
                min_abs_mi_frac=_ens_min_abs_mi_frac,
                scorers=_ens_scorers,
                aggregator=_ens_aggregator,
                random_state=int(getattr(self, "random_seed", 0) or 0),
            )
            _ens_appended = [c for c in X_ens.columns if c not in _X_before_ens_cols]
            if _ens_appended:
                X = fe_append_columns(X, fe_extract_columns(X_ens, _ens_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ens_appended)
                for _r in _ens_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth ensemble: appended %d " "engineered column(s) via %s aggregator: %s",
                        len(_ens_appended),
                        _ens_aggregator,
                        _ens_appended[:8],
                    )
        except Exception as _ens_exc:
            logger.warning(
                "MRMR.fit hybrid_orth ensemble FE raised %s: %s; " "continuing without ensemble columns.",
                type(_ens_exc).__name__,
                _ens_exc,
            )
    # 2026-06-01 Layer 76 — META-SCORER auto-selection that LEARNS from
    # cheap signal characteristics ("data fingerprints") and dispatches
    # to the predicted-best scorer of the Layer 21 / 65 / 66 / 67 / 71 /
    # 72 / 74 family (sibling module ``_orthogonal_meta_scorer_fe``).
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). Where
    # Layer 68 (per-column bootstrap LCB) and Layer 69 (rank fusion) run
    # ALL scorers and let a meta-criterion pick, Layer 76 spends a small
    # fixed budget on cheap fingerprints + a deterministic 5-rule cascade
    # distilled from the L75 empirical matrix, then runs ONLY the
    # predicted-best scorer. Wall-clock saving roughly n_scorers - 1 vs
    # L68/L69. Engineered VALUES bit-equal to Layer 21 -> recipes reuse
    # the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_meta_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._orthogonal_meta_scorer_fe import (
                hybrid_orth_mi_meta_fe_with_recipes,
            )

            _y_for_meta = _y_np
            _hybrid_already_appended = set(getattr(self, "hybrid_orth_features_", None) or [])
            if getattr(self, "factors_names_to_use", None):
                _meta_cols = [c for c in self.factors_names_to_use if c in X.columns and c not in _hybrid_already_appended]
            else:
                _meta_cols = [c for c in X.columns if c not in _hybrid_already_appended]
            # Orthogonal/polynomial FE is numeric-only; drop non-numeric cols (raw cat / string) before the float
            # conversion, else it raises "could not convert string to float" and the whole FE pass is dropped.
            _meta_cols = _orth_fe_numeric_cols(X, _meta_cols)
            _meta_degrees = tuple(int(d) for d in getattr(
                self, "fe_hybrid_orth_degrees", (2, 3),
            ))
            _meta_basis = str(getattr(
                self, "fe_hybrid_orth_basis", "auto",
            ))
            _meta_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
            _meta_force = getattr(
                self, "fe_hybrid_orth_meta_force_scorer", None,
            )
            # Same calibration as Layers 65 / 66 / 67 / 68 / 69: the
            # scorer captures raw-x dependence nearly as cleanly as
            # the engineered column, so single-source uplift sits near
            # 1.0; 0.95 / 0.05 floors keep the gate from rejecting
            # genuine wins on a sample-noise tick.
            _meta_min_uplift = 0.95
            _meta_min_abs_mi_frac = 0.05
            _X_before_meta_cols = list(X.columns)
            (
                X_meta, _meta_scores, _meta_recipes,
                _meta_chosen, _meta_fp,
            ) = fe_decide_on_subsample(
                hybrid_orth_mi_meta_fe_with_recipes,
                X, _y_for_meta,
                subsample_n=int(getattr(self, "fe_check_pairs_subsample_n", 0) or 0),
                subsample_seed=int(getattr(self, "random_seed", 0) or 0),
                shared_subsample_idx=getattr(self, "_fe_shared_subsample_idx", None),
                cols=_meta_cols,
                degrees=_meta_degrees,
                basis=_meta_basis,
                top_k=_meta_top_k,
                min_uplift=_meta_min_uplift,
                min_abs_mi_frac=_meta_min_abs_mi_frac,
                force_scorer=_meta_force,
                random_state=int(getattr(self, "random_seed", 0) or 0),
            )
            _meta_appended = [c for c in X_meta.columns if c not in _X_before_meta_cols]
            if _meta_appended:
                X = fe_append_columns(X, fe_extract_columns(X_meta, _meta_appended))
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_meta_appended)
                for _r in _meta_recipes:
                    _hybrid_orth_pre_recipes[_r.name] = _r
                if verbose:
                    logger.info(
                        "MRMR.fit hybrid_orth meta-scorer: dispatched " "to %r (force=%r); appended %d engineered " "column(s): %s",
                        _meta_chosen,
                        _meta_force,
                        len(_meta_appended),
                        _meta_appended[:8],
                    )
            # Expose the chosen scorer + fingerprint for downstream
            # audit / debug (also survives pickle because plain attrs).
            self.hybrid_orth_meta_chosen_scorer_ = _meta_chosen
            self.hybrid_orth_meta_fingerprint_ = dict(_meta_fp)
        except Exception as _meta_exc:
            logger.warning(
                "MRMR.fit hybrid_orth meta-scorer FE raised %s: %s; " "continuing without meta-scorer columns.",
                type(_meta_exc).__name__,
                _meta_exc,
            )
    # 2026-05-21 revert of Wave 29 P1 polars->pandas coercion. That
    # coercion was added on the premise that downstream ``X[target_name]
    # = y`` mutation assumed pandas and would raise on polars; but the
    # ``_is_polars_input`` branch immediately below (line ~1326) ALREADY
    # handles polars via ``X.with_columns(target_series)``. The Wave 29
    # coercion was a false-positive fix that killed the zero-copy
    # polars promise (test_mrmr_fe_zero_copy_polars regressed --
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
    if bool(getattr(self, "fe_mi_greedy_enable", False)):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from .._mi_greedy_fe import greedy_mi_fe_construct_with_recipes
            from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
            # W6: record abs-MAD floor kills in the mi_greedy stage into the
            # FE rejection ledger (pure-record; selection unchanged).
            _mig_step = int(getattr(self, "_fe_steps_executed_", -1))

            def _mig_reject_sink(**_kw):
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
                    except Exception:
                        _y_for_mig = _y_for_mig.astype(np.int64)
            # Restrict the MI-greedy seed pool to RAW source columns only
            # (i.e. exclude hybrid-orth-appended columns from the prior
            # stage). Compound transforms like ``log(He2(x))`` would
            # create recipes whose ``src_names`` reference an engineered
            # column that does not exist at transform time -- replay
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
    if bool(getattr(self, "fe_mi_greedy_cmi_enable", False)):
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
                    except Exception:
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
    # only the full-data per-category lookup -- no y at replay time.
    # Engineered columns route through ``hybrid_orth_features_`` so the
    # end-of-fit remap treats them as engineered features (same routing
    # as Layer 23 / 26 / 32).
    self.kfold_te_features_ = []
    _kfold_te_pre_recipes: dict = {}
    _binned_agg_pre_recipes: dict = {}
    if bool(getattr(self, "fe_kfold_te_enable", False)):
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
                    _mig_appended = set(self.mi_greedy_features_ or [])
                    _te_cols = [c for c in _te_cols if c in X.columns and c not in _hybrid_appended and c not in _mig_appended]
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

    # GROUPED AGGREGATION OVER QUANTILE-BINNED NUMERIC CELLS (2026-06-13). Appends leak-safe per-cell
    # mean/std/skew/kurt of numeric columns grouped by quantile-binned cells of other numerics. Runs in the
    # pre-FE region (before categorize_dataset) so the appended columns enter screening like any numeric, and
    # routes recipes through hybrid_orth_features_ so a selected binagg column lands in _engineered_recipes_.
    if bool(getattr(self, "fe_binned_numeric_agg_enable", False)) and fe_polars_exceeds(X):
        warnings.warn(
            "MRMR: fe_binned_numeric_agg_enable=True but X is a large polars frame (> ~2 GiB); binned-agg is an OOF stat "
            "needing a full-frame decision and is skipped to avoid a whole-frame to_pandas copy.",
            UserWarning, stacklevel=3,
        )
    elif bool(getattr(self, "fe_binned_numeric_agg_enable", False)):
        try:
            from .._binned_numeric_agg_fe import binned_numeric_agg_with_recipes
            _ba_y = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).ravel()
            _X_before_ba = list(X.columns)
            X_ba, _ba_appended, _ba_recipes = binned_numeric_agg_with_recipes(
                fe_to_pandas(X), _ba_y,
                stats=tuple(getattr(self, "fe_binned_numeric_agg_stats", ("mean", "std", "skew", "kurt")) or ("mean",)),
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
        bool(getattr(self, "fe_count_encoding_enable", False))
        or bool(getattr(self, "fe_frequency_encoding_enable", False))
        or bool(getattr(self, "fe_cat_num_interaction_enable", False))
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
                _record_fe_rejection(self, step=_l34_step, **_kw)

            _hybrid_appended_l34 = set(self.hybrid_orth_features_ or [])
            _mig_appended_l34 = set(self.mi_greedy_features_ or [])
            _te_appended_l34 = set(self.kfold_te_features_ or [])
            _engineered_seen_l34 = _hybrid_appended_l34 | _mig_appended_l34 | _te_appended_l34

            # ----- Count encoding ----------------------------------------
            if bool(getattr(self, "fe_count_encoding_enable", False)):
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
            if bool(getattr(self, "fe_frequency_encoding_enable", False)):
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
            if bool(getattr(self, "fe_cat_num_interaction_enable", False)):
                try:
                    _cn_cats = tuple(getattr(self, "fe_cat_num_interaction_cat_cols", ()) or ())
                    _cn_nums = tuple(getattr(self, "fe_cat_num_interaction_num_cols", ()) or ())
                    _cn_cats = [c for c in _cn_cats if c in X.columns]
                    _cn_nums = [c for c in _cn_nums if c in X.columns]
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
            # missingness-FE family encodes -- is_missing__ would be all-zeros and missingness_pattern would collapse to a single pattern. The raw
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
                _cfg = tuple(cfg or ())
                if _cfg:
                    return [c for c in _cfg if c in X.columns and c not in _engineered_seen_l37]
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
        bool(getattr(self, "fe_pairwise_ratio_enable", False))
        or bool(getattr(self, "fe_pairwise_log_ratio_enable", False))
        or bool(getattr(self, "fe_grouped_delta_enable", False))
        or bool(getattr(self, "fe_lagged_diff_enable", False))
    ):
        # grouped_delta / lagged_diff are cross-row (group / time ordered) and ratio / log-ratio rank their mi_gate on the
        # full frame, none wired for closed-form subsample-replay -- so this block needs the full frame: gate the materialisation
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
                _record_fe_rejection(self, step=_l38_step, **_kw)

            # ----- Pairwise ratio --------------------------------------------
            if bool(getattr(self, "fe_pairwise_ratio_enable", False)):
                try:
                    _ratio_cols = tuple(getattr(self, "fe_pairwise_ratio_cols", ()) or ())
                    _ratio_cols = [c for c in _ratio_cols if c in X.columns]
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
            if bool(getattr(self, "fe_pairwise_log_ratio_enable", False)):
                try:
                    _lr_cols = tuple(getattr(self, "fe_pairwise_log_ratio_cols", ()) or ())
                    _lr_cols = [c for c in _lr_cols if c in X.columns]
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
            if bool(getattr(self, "fe_grouped_delta_enable", False)):
                try:
                    _gd_group = getattr(self, "fe_grouped_delta_group_col", None)
                    _gd_nums = tuple(getattr(self, "fe_grouped_delta_num_cols", ()) or ())
                    _gd_nums = [c for c in _gd_nums if c in X.columns]
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
            if bool(getattr(self, "fe_lagged_diff_enable", False)):
                try:
                    _ld_time = getattr(self, "fe_lagged_diff_time_col", None)
                    _ld_vals = tuple(getattr(self, "fe_lagged_diff_value_cols", ()) or ())
                    _ld_vals = [c for c in _ld_vals if c in X.columns]
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

    # Layer 87 (2026-06-01): grouped multi-stat aggregator with CMI gate.
    # NVIDIA cuDF Kaggle-Grandmaster technique #1. Per-group statistics of a
    # continuous column broadcast to rows + z-within / ratio residuals, each
    # CMI-gated against the raw support and uplift-gated against the source
    # num_col marginal MI. Routing piggybacks on hybrid_orth_features_ (same
    # Layer 23 remap as Layers 33/34/37/38).
    if bool(getattr(self, "fe_grouped_agg_enable", False)):
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
                        except Exception:
                            _y_for_ga = _y_for_ga.astype(np.int64)

                _ga_groups = tuple(getattr(self, "fe_grouped_agg_group_cols", ()) or ())
                _ga_groups = [c for c in _ga_groups if c in X.columns] or None
                _ga_nums = tuple(getattr(self, "fe_grouped_agg_num_cols", ()) or ())
                _ga_nums = [c for c in _ga_nums if c in X.columns] or None
                _ga_stats = tuple(getattr(self, "fe_grouped_agg_stats", ()) or ("mean", "std", "min", "max", "nunique", "skew", "median"))
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

    # Layer 93 (2026-06-01): COMPOSITE (multi-column) group-key aggregates.
    # Multi-col extension of Layer 87: each composite key is factorized into
    # one integer-coded group and run through the same per-group stat / z /
    # ratio machinery; survivors are CMI-gated against the raw support and
    # uplift-gated against the source num_col marginal MI. Composite keys whose
    # distinct-cell count exceeds 0.5*n are refused (Layer 29 guard). Routing
    # piggybacks on hybrid_orth_features_ (same Layer 23 remap as 33/.../87).
    if bool(getattr(self, "fe_composite_group_agg_enable", False)):
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
                        except Exception:
                            _y_for_cga = _y_for_cga.astype(np.int64)

                # key_sets: each entry is a tuple of >= 2 group cols. Empty =>
                # auto-detect r-combinations of detected group columns.
                _cga_key_sets_raw = tuple(getattr(self, "fe_composite_group_agg_key_sets", ()) or ())
                _cga_key_sets = [tuple(c for c in gset if c in X.columns) for gset in _cga_key_sets_raw]
                _cga_key_sets = [g for g in _cga_key_sets if len(g) >= 2] or None
                _cga_nums = tuple(getattr(self, "fe_composite_group_agg_num_cols", ()) or ())
                _cga_nums = [c for c in _cga_nums if c in X.columns] or None
                _cga_stats = tuple(getattr(self, "fe_composite_group_agg_stats", ()) or ("mean", "std", "count"))
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

    # Layer 88 (2026-06-01): per-group histogram + quantile FE with
    # target-aware edges. NVIDIA cuDF Kaggle-Grandmaster technique #2.
    # Percentile-rank-within-group + per-group IQR / p90-p10 spread, optionally
    # the OOF-fit target-aware supervised bin index; each survivor MI-gated
    # against the source num_col marginal MI. Routing piggybacks on
    # hybrid_orth_features_ (same Layer 23 remap as Layers 33/34/37/38/87).
    if bool(getattr(self, "fe_grouped_quantile_enable", False)):
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
                _gq_groups = [c for c in _gq_groups if c in X.columns] or None
                _gq_nums = tuple(getattr(self, "fe_grouped_quantile_num_cols", ()) or ())
                _gq_nums = [c for c in _gq_nums if c in X.columns] or None
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
                _gq_quantiles = tuple(getattr(self, "fe_grouped_quantile_quantiles", ()) or (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
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

    # Layer 89 (2026-06-01): cat x cat synergy cross with II pre-filter.
    if bool(getattr(self, "fe_cat_pair_enable", False)):
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
                _cp_cols = [c for c in _cp_cols if c in X.columns] or None
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

    # Layer 94 (2026-06-01): cat x cat x cat TRIPLE synergy cross via beam
    # search over three-way interaction information (co-information).
    if bool(getattr(self, "fe_cat_triple_enable", False)):
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
                _ct_cols = [c for c in _ct_cols if c in X.columns] or None
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

    # Layer 90 (2026-06-01): numeric decomposition (multi-precision rounding +
    # decimal-digit extraction) with a bootstrap-stable MI gate.
    if bool(getattr(self, "fe_numeric_decompose_enable", False)):
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

    # Layer 95 PART A (2026-06-01): periodic / modular decomposition. For each
    # (col, period) emit x mod period plus its sin/cos phase encoding; each
    # candidate gated by Layer 62 bootstrap-stable MI (the gate doubles as
    # auto-period detection). Routing piggybacks on hybrid_orth_features_.
    if bool(getattr(self, "fe_modular_enable", False)):
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
    # combination of integer columns -- (a+b) mod m, (a*b) mod m, n-way parity, or a
    # single column's hidden non-calendar period -- which smooth bases cannot fit.
    # Cheap-first / escalate + permutation-null gate; budget-guarded on wide frames.
    # The four discrete-structural families (pairwise-modular / row-argmax / conditional-gate /
    # binned-agg) are gated by their own enable flags and fire INDEPENDENTLY of fe_max_steps>0 (they are a
    # distinct operator group, deliberately usable with fe_max_steps=0 -- the operator-lift biz_value tests
    # rely on exactly that: fe_max_steps=0 + an explicit fe_<op>_enable=True must still build the composite).
    # SMALL-N RELIABILITY FLOOR: their composites are high-cardinality joints (integer lattice/gcd, gated
    # thresholds, row-argmax) whose MI is unreliable at tiny n -- on small-n pure noise a spurious composite
    # clears the relevance gate and is admitted (RC2 pure-noise n=300), and it crowds the clean raw signal.
    # So when FE is otherwise OFF (fe_max_steps==0) require at least ``_DISCRETE_FE_MIN_N_AT_FE0`` rows
    # before building them; with FE enabled (fe_max_steps>=1) the normal FE pipeline competes them down so
    # no floor is needed. Calibrated so RC2 (n=300) stays clean while the operator-lift cases (n=2000) fire.
    _DISCRETE_FE_MIN_N_AT_FE0 = 500
    _discrete_fe_master = bool(getattr(self, "fe_discrete_structural_operators_enable", True)) and (
        fe_max_steps > 0 or (isinstance(X, pd.DataFrame) and len(X) >= _DISCRETE_FE_MIN_N_AT_FE0)
    ) and _fe_budget_ok()
    # OPERATOR SKIP-GATE (2026-06-18, perf). The four discrete-structural operators (pairwise-modular /
    # row-argmax / conditional-gate / binned-agg) hunt for NONLINEAR/regime structure via MI-kernel scans
    # over many candidate combos -- ~58% of an additive-regression fit (cProfile: cheap_conditional_gate_scan
    # 7.2s + binned_numeric_agg 4s of a 19s fit). On an additive-LINEAR regression target there is no such
    # structure to find, so a single cheap linear fit on the raws is a necessary-condition gate: if the raws
    # already explain y (R^2>=0.92), skip the scans. Classification keeps them (R^2 N/A there -> the gate
    # returns False), and any genuine regime/modular/interaction target leaves a large linear residual
    # (low R^2) -> the operators still fire. One ~0.1s linear fit vs ~11s of scans.
    #
    # SCOPE: AUTOMATIC PATH ONLY (fe_max_steps>0). The skip-gate is a perf optimisation for the default FE
    # pipeline, where the operators run automatically alongside the basis/escalation passes and the gate just
    # spares their scans when the raws already explain y. With fe_max_steps==0 the operators are the ONLY FE
    # the user asked for (the deliberate operator-only path documented above) -- skipping them there silently
    # suppresses an explicitly-requested, genuinely-detectable composite. A linearly-explainable target can
    # still hold real MI/operator structure (e.g. y=1[argmax(a,b,c)==0]: raw-only in-sample logistic AUC ~0.98
    # yet argmax__a__b__c is a clean, selectable composite); the in-sample linear/logistic score is NOT a
    # licence to drop the operator the user explicitly enabled. So the gate is confined to fe_max_steps>0.
    if _discrete_fe_master and fe_max_steps > 0:
        try:
            from .._fe_linear_explainability import raws_linearly_explain_y

            if raws_linearly_explain_y(X, y, seed=int(getattr(self, "random_seed", 0) or 0)):
                _discrete_fe_master = False
        except Exception:  # nosec B110 - optional/best-effort path, rationale documented
            pass  # gate is an optimisation; on any failure keep the operators (correct path)

    # Shared class-MI target binning for the four discrete-structural FE operators (pairwise-modular / integer-lattice / row-argmax / conditional-gate).
    # All four gate candidates on the SAME 1D y binned with the SAME quantization_nbins via bin_y_for_class_mi; compute the applicability flag + binned
    # labels ONCE here and reuse, rather than re-quantile-binning the identical target inside each block. _y_np is fixed for the whole fit (never rebound).
    _y_class_mi_applicable = False
    _y_class_mi_binned = None
    if (
        _discrete_fe_master
        and isinstance(X, pd.DataFrame)
        and (
            bool(getattr(self, "fe_pairwise_modular_enable", False))
            or bool(getattr(self, "fe_integer_lattice_enable", False))
            or bool(getattr(self, "fe_row_argmax_enable", False))
            or bool(getattr(self, "fe_conditional_gate_enable", False))
        )
    ):
        from .._fe_accuracy_gate import bin_y_for_class_mi as _bin_y_class_mi, class_mi_fe_applicable as _class_mi_applicable

        _y_class_mi_applicable = _class_mi_applicable(_y_np)
        if _y_class_mi_applicable:
            _y_class_mi_binned = _bin_y_class_mi(_y_np, nbins=int(getattr(self, "quantization_nbins", 10)))

    if _discrete_fe_master and bool(getattr(self, "fe_pairwise_modular_enable", False)):
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
                # (bin_y_for_class_mi, nbins=quantization_nbins) so the kernel sees a discrete target -- the prior int64 cast collapsed continuous y
                # to ~n bogus classes. Only a 2D (multilabel/multi-target) y stays skipped (binning a label matrix is out of scope). Reuses the
                # shared _y_class_mi_* computed once above (identical y + nbins across all four discrete-structural operators).
                _pm_appended, _pm_recipes = ([], [])
                if _y_class_mi_applicable:
                    _y_pm_binned = _y_class_mi_binned
                    # Restrict operands to raw input columns: combining on already-engineered columns yields nested recipes
                    # whose engineered source is not resolvable at replay time (transform() emits NaN and drops the feature).
                    _pm_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]
                    _pm_appended, _pm_recipes = hybrid_pairwise_modular_fe_with_recipes(
                        X, _y_pm_binned,
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
    # divisor (gcd), its dual lcm, or a bit-level co-occurrence (a & b) of integer columns -- structure smooth/arithmetic/
    # modular ops cannot express. Cheap-first pairs-only scan + dual margin/permutation-null gate; budget-guarded.
    if _discrete_fe_master and bool(getattr(self, "fe_integer_lattice_enable", False)):
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
                        X, _y_il_binned,
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

    # Row-argmax FE (frontier pass 2): for a column triple (a, b, c) emit the integer index 0/1/2 of the row-maximum -- an
    # ordinal/comparison pattern the MI/linear path cannot read off marginals or pairwise diffs. ZERO free params, detector-clean;
    # leak-free deterministic replay (np.argmax over the stacked source columns). Budget-guarded on wide frames.
    if _discrete_fe_master and bool(getattr(self, "fe_row_argmax_enable", False)):
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
                        X, _y_am_binned,
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
    if _discrete_fe_master and bool(getattr(self, "fe_conditional_gate_enable", False)):
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
                # (bin_y_for_class_mi) before the tau-grid + conditional-divergence sweep -- the prior int64 cast turned continuous y into ~n
                # distinct classes (the tau-sweep MI exploded / never completed). A 2D y stays skipped (the kernel reads a dead signal).
                # Reuses the shared _y_class_mi_* binned above.
                _cg_appended, _cg_recipes = ([], [])
                if _y_class_mi_applicable:
                    _y_cg_binned = _y_class_mi_binned
                    # Raw-column operands only (see the row-argmax / modular note); engineered operands would orphan at replay.
                    _cg_raw_cols = [c for c in X.columns if c not in set(self.hybrid_orth_features_ or [])]
                    _cg_appended, _cg_recipes = hybrid_conditional_gate_fe_with_recipes(
                        X, _y_cg_binned,
                        cols=_cg_raw_cols,
                        top_k=int(getattr(self, "fe_conditional_gate_top_k", 4)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                        max_cols=int(getattr(self, "fe_conditional_gate_max_cols", 200)),
                        k_gate=int(getattr(self, "fe_conditional_gate_k_gate", 8)),
                        k_operand=int(getattr(self, "fe_conditional_gate_k_operand", 10)),
                        # SCREEN SUBSAMPLE (2026-06-20): subsample the gate-DETECTION scan (tau + MI
                        # ranking are rank-stable; the recipe replays the gate at FULL n). Reuse the
                        # resolved screen-n (fe_check_pairs_subsample_n) UNCONDITIONALLY -- the default-
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

    # Layer 95 PART B (2026-06-01): per-group distribution-distance. For each
    # (group, num) emit the group-level z / KL / Wasserstein-1 distance from the
    # global distribution, broadcast to rows; each survivor MI-gated against the
    # source num_col marginal MI. Routing piggybacks on hybrid_orth_features_.
    if bool(getattr(self, "fe_group_distance_enable", False)):
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
                _gd_groups = [c for c in _gd_groups if c in X.columns] or None
                _gd_nums = tuple(getattr(self, "fe_group_distance_num_cols", ()) or ())
                _gd_nums = [c for c in _gd_nums if c in X.columns] or None
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

    # Layer 104 (2026-06-01): THREE new recipe-based FE families.
    # Family D (backlog #12, 2026-06-09): conditional dispersion / 2nd-moment.
    self.rare_category_features_ = []
    self.conditional_residual_features_ = []
    self.conditional_dispersion_features_ = []
    self.wavelet_features_ = []
    self.rankgauss_features_ = []

    # FAMILY A -- rare-category indicator + frequency-band encoding. A category
    # being RARE is itself predictive; emit is_rare_{col} + freq_band_{col}.
    # MI-gated against the raw-baseline floor. Routing piggybacks on
    # hybrid_orth_features_.
    if bool(getattr(self, "fe_rare_category_enable", False)):
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
                    _record_fe_rejection(self, step=_rc_step, **_kw)

                _y_for_rc = _y_np
                _rc_cols = tuple(getattr(self, "fe_rare_category_cols", ()) or ())
                _rc_cols = [c for c in _rc_cols if c in X.columns] or None
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

    # FAMILY B -- NUM x NUM conditional residual x_i - E[x_i | bin(x_j)].
    # Cardinality-bounded by top raw-MI columns; MI-gated. Routing piggybacks on
    # hybrid_orth_features_.
    if bool(getattr(self, "fe_conditional_residual_enable", False)):
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
                    _record_fe_rejection(self, step=_cr_step, **_kw)

                _y_for_cr = _y_np
                _cr_cols = tuple(getattr(self, "fe_conditional_residual_cols", ()) or ())
                _cr_cols = [c for c in _cr_cols if c in X.columns] or None
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

    # FAMILY D -- NUM x NUM conditional DISPERSION / 2nd-moment (backlog #12).
    # Bin x_j; per bin store conditional STD of x_i; emit |z| / z^2 (conditional
    # dispersion anomaly). DEFAULT-ON: MI-gateable (|z| is a non-monotone fold ->
    # genuine MI on heteroscedastic targets) + SELF-LIMITING (a dual-uplift gate
    # admits a column only when its MI beats BOTH raw x_i AND the |mean-residual|
    # Family-B sibling, so homoscedastic / canonical fixtures admit 0 and the
    # operator does not perturb pair-FE recovery). Routing piggybacks on
    # hybrid_orth_features_; recipes carry no y -> leak-safe replay.
    if bool(getattr(self, "fe_conditional_dispersion_enable", False)):
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
                    _record_fe_rejection(self, step=_cd_step, **_kw)

                _y_for_cd = _y_np
                _cd_cols = tuple(getattr(self, "fe_conditional_dispersion_cols", ()) or ())
                _cd_cols = [c for c in _cd_cols if c in X.columns] or None
                # RAW columns only (2026-06-10 fix, same class as the wavelet stage
                # below): the all-numeric default scope over the already-augmented X
                # builds dispersion features OF engineered columns -> nested recipes
                # the 1-deep replay cannot order at transform() time (KeyError on the
                # engineered parent when it is not selected). Raw scope keeps every
                # conditional-dispersion recipe replayable.
                # ``feature_names_in_`` is not yet assigned here; scope to the raw
                # pre-FE column snapshot (the cat_pair / cat_triple guard's ledger),
                # which is strictly safer than the ``hybrid_orth_features_`` exclusion
                # -- that ledger only tracks orth / hinge / wavelet columns and misses
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

    # HAAR WAVELET / localized multiresolution basis (backlog #13, 2026-06-09).
    # A NEW operator for LOCALIZED bump / multiscale piecewise structure: y jumps
    # only inside a narrow sub-window of x (Fourier Gibbs-rings it, spline's fixed
    # quantile knots smooth it away). Emits a small held-out-scale-selected dyadic
    # set of Haar indicators psi_{j,k} (+1 left / -1 right half of a dyadic
    # interval). DEFAULT-ON + SELF-LIMITING: the noise-aware held-out MAD floor +
    # max-legs cap bound the candidate explosion, and each leg is admitted on its
    # held-out INCREMENTAL MI over raw x AND a complementarity guard (must beat a
    # SMOOTH location-refinement of x) -- so a localized step/bump admits legs, a
    # SMOOTH (sin / monotone) column admits 0 (Fourier owns it, complementary),
    # pure noise admits 0. The leg is NON-monotone -> MI-VISIBLE, so it routes
    # through the MI-based gate (no deferred-materialise / re-add dance the
    # MI-invariant hinge needs). Recipes (``orth_wavelet``) store (lo, span) +
    # dyadic (j, k); replay is the closed-form indicator -- no y, leak-safe.
    # Routing piggybacks on hybrid_orth_features_ (like Family D dispersion).
    if bool(getattr(self, "fe_wavelet_enable", False)) and _fe_budget_ok():
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Haar wavelet FE enabled but X is not a pandas DataFrame; "
                "the features are skipped. Convert via X.to_pandas() before fit() "
                "to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._wavelet_basis_fe import hybrid_wavelet_fe_with_recipes

                _y_for_wv = _y_np
                _wv_cols = tuple(getattr(self, "fe_wavelet_cols", ()) or ())
                _wv_cols = [c for c in _wv_cols if c in X.columns] or None
                # RAW columns only (2026-06-10 fix, mirrors the extra-basis stage's
                # guard at the hybrid_orth call above): by this point X is ALREADY
                # augmented with poly/fourier/spline/hinge engineered columns, so the
                # all-numeric default scope emitted NESTED recipes (e.g.
                # ``x0__p2sin1__haar_j3k5`` -- a Haar leg of an engineered Fourier
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

    # FAMILY C -- RankGauss (rank-Gaussianisation). NOT MI-gated: monotone ->
    # MI-invariant by the data-processing inequality; the pool is bounded by raw
    # marginal MI and the value is downstream (linear / NN). Routing piggybacks
    # on hybrid_orth_features_.
    if bool(getattr(self, "fe_rankgauss_enable", False)):
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
                _rg_cols = [c for c in _rg_cols if c in X.columns] or None
                # RAW columns only (2026-06-10 fix, same class as the wavelet /
                # conditional-dispersion stages): keep rankgauss recipes 1-deep and
                # replayable -- never rank-Gaussianise an engineered column whose
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

    # Layer 92 (2026-06-01): temporal leak-safe grouped aggregations. Carved
    # verbatim into the sibling ``_fe_stage_temporal_agg`` (Tier E partial
    # split); the helper threads self + ``_y_np`` / ``verbose`` /
    # ``_temporal_agg_pre_recipes`` explicitly, mutates self + the recipes dict
    # in place, and RETURNS the (possibly replaced) working ``X`` frame.
    from ._fe_stage_temporal_agg import _fe_stage_temporal_agg
    X = _fe_stage_temporal_agg(self, X, _y_np, verbose, _temporal_agg_pre_recipes)

    # ACCURACY GATE (2026-06-04, default ON via ``fe_accuracy_gate``). The MI-uplift gates inside the FE generators are fooled by plug-in MI's bias inflation: a Fourier / chirp / Hermite transform of a strong RAW signal earns an inflated MI estimate and out-ranks (then evicts) the raw column even when it adds NO real predictive value. The adaptive-Fourier PROTECTION block at support-finalisation then force-readds those hijackers past the MRMR screen, so they survive into support_ AND leak into ``hybrid_orth_features_`` / ``_adaptive_fourier_features_`` even when a genuine raw signal (or its is_missing__ MNAR indicator) carries the information. This gate runs a held-out multivariate linear-probe uplift check per engineered column against its raw source: a column that adds no held-out uplift over its source -- or whose source is >2%-missing (MNAR fail-closed, the signal lives in the NaN pattern the probe cannot see) -- is dropped here so it can neither evict the raw signal nor leak into the roster. Only orth_* engineered columns with a single resolvable raw source are gated; the is_missing__ / missingness_* indicators are exempt by construction (their recipes live in ``_miss_*_pre_recipes``, never ``_hybrid_orth_pre_recipes``, so they are never routed here). y is read only at fit; transform replays the survivors without y. Best-effort: any failure falls back to keeping the column.
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
            # Pass 1: base (non-Fourier) columns -- uplift over the raw source alone (also the MNAR fail-closed for >2%-missing sources).
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
            # Pass 2: adaptive-Fourier / chirp columns -- uplift over [raw source + surviving base siblings of that source]. A Fourier redundant with a He2 sibling (both encode x**2)
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

    # Layer 27 (2026-05-31): cross-stage engineered-column dedup. Hybrid and
    # MI-greedy stages run independently; on signals like ``y = sign(x^2 - 1)``
    # hybrid emits ``x__He2`` and MI-greedy emits ``square(x)`` / ``abs(x)`` /
    # ``sqrt_abs(x)`` / ``log_abs(x)`` -- all are monotone-in-|x| encodings
    # of the SAME signal (Pearson |corr| ~ 0.99+ on rank-correlated MI binning).
    # MRMR's CMI gate can't tell them apart well enough to prune; the
    # combined support inflates with 4-5 near-identical columns. The cheap
    # cure is a pre-MRMR dedup pass against the engineered cousins: keep the
    # first appended occurrence, drop everything correlating >= 0.999 with an
    # already-kept engineered column. Raw input columns are never deduped
    # here -- that's MRMR's job and removing raw cols would change the
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
    _eng_cols_appended = [_c for _c in _eng_cols_appended_raw if not (_c in _eng_seen or _eng_seen.add(_c))]
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
    # exact MI ties (a monotone twin bins identically, so MI is numerically equal -- prefer the explicitly-requested constructor output). MI scoring is best-effort: any failure falls back to the
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
                except Exception:
                    _y_for_eng_mi = _y_for_eng_mi.astype(np.int64)
        else:
            _y_for_eng_mi = _y_for_eng_mi.astype(np.int64)
        if isinstance(X, pd.DataFrame) and len(_eng_cols_appended) >= 2:
            _mi_cols = [_c for _c in _eng_cols_appended if _c in X.columns]
            if _mi_cols:
                _mi_mat = X[_mi_cols].to_numpy(dtype=np.float64)
                _mi_vals = _mi_classif_batch(_mi_mat, _y_for_eng_mi, nbins=10)
                _eng_mi = {_name: float(_v) for _name, _v in zip(_mi_cols, _mi_vals)}
    except Exception:
        _eng_mi = {}

    def _eng_dedup_prefer(cand: str, kept: str) -> bool:
        """Return True when ``cand`` should DISPLACE the already-kept ``kept`` on a near-duplicate collision.

        Only CROSS-STAGE collisions (exactly one of the pair is an mi_greedy / constructor-requested column) ever flip the survivor: within a single stage we preserve the original
        first-appended policy byte-for-byte, so the dedup stays deterministic on the monotone-twin families a single basis stage emits (a quantile-binned MI tie between ``x__He2`` /
        ``x__cos1`` / ``x__L2`` would otherwise reshuffle non-deterministically). Across stages we keep the column carrying more MI about y, and the explicitly-requested mi_greedy column
        wins an exact MI tie (a monotone twin bins identically, so its MI is numerically equal -- without this the default-on basis twin would silently evict the genuine ``|x|``-family signal).
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
        # so we reuse them instead of re-sorting both columns per pair -- removing the O(K^2) rank-sorts the
        # dedup did (only the O(K^2) corrcoef remains). Bit-identical: same arrays -> same average ranks.
        _eng_ranks: dict[str, np.ndarray] = {}
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
            for _kept in _eng_keep:
                _arr_k = _eng_arrs[_kept]
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
                    _ranks_b = _eng_ranks.get(_kept)
                    if _ranks_b is None:
                        _ranks_b = pd.Series(_arr_k).rank(method="average").to_numpy()
                        _eng_ranks[_kept] = _ranks_b
                else:
                    _ranks_a = pd.Series(_a).rank(method="average").to_numpy()
                    _ranks_b = pd.Series(_b).rank(method="average").to_numpy()
                if _ranks_a.std() <= 1e-12 or _ranks_b.std() <= 1e-12:
                    continue
                _rank_corr = abs(float(np.corrcoef(_ranks_a, _ranks_b)[0, 1]))
                if np.isfinite(_rank_corr) and _rank_corr >= 0.99:
                    _colliding_kept.append(_kept)
            if _colliding_kept:
                # Keep-higher-MI: the candidate displaces every colliding kept column it out-scores, and is itself dropped only if some colliding kept column wins.
                # ``_eng_dedup_prefer`` returns False when MI is unavailable, so an unscored cluster degrades exactly to the original first-appended policy (candidate dropped).
                _cand_loses = any(not _eng_dedup_prefer(_c, _kept) for _kept in _colliding_kept)
                if _cand_loses:
                    _eng_drop.add(_c)
                else:
                    for _kept in _colliding_kept:
                        _eng_drop.add(_kept)
                        _eng_keep.remove(_kept)
                        _eng_arrs.pop(_kept, None)
                    _eng_keep.append(_c)
                    _eng_arrs[_c] = _arr_c
            else:
                _eng_keep.append(_c)
                _eng_arrs[_c] = _arr_c
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

    # Layer 91 (2026-06-01): Tier-2 UNIFIED SECOND-PASS CMI GATE. The Layer 27
    # dedup above is UNSUPERVISED (Spearman rank-corr between engineered cousins)
    # and so cannot see cross-mechanism redundancy that only manifests
    # conditional on y -- e.g. ``count(cat_a)`` and ``freq(cat_a)`` ARE caught by
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
            _eng_now = [c for c in _eng_now if not (c in _seen_u or _seen_u.add(c))]
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
    # Layer 26: also exclude MI-greedy-appended columns -- same routing
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
        # Layer 64 (2026-05-31) defense: keep only the FIRST occurrence
        # of each duplicate-label column position in X. The engineered
        # rosters and the recipe ledger are NOT pruned here -- the
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
        self.feature_names_in_ = [c for c in _names_source if c not in _engineered_names_set]
    else:
        self.feature_names_in_ = [c for c in _all_cols if c not in _engineered_names_set]
    self.n_features_in_ = len(self.feature_names_in_)

    # FE AUTO-ESCALATION fitting target (2026-06-10): a RANK transform of the raw
    # numeric y, stashed for the escalation proposers' corr-based warp fits. The FE
    # step's ``classes_y`` are LABEL codes from the internal target quantisation
    # (NOT guaranteed ordinal/monotone in y -- measured 37 unordered codes on a
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
    except Exception:
        self._fe_escalation_y_rank_ = None

    # PREWARP ALS RECONSTRUCTION TARGET (2026-06-11): stash the RAW CONTINUOUS y so
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
    except Exception:
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

    # Native Polars support -- no `.to_pandas()` copy. Production frames are 100+ GB; full materialization
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
        # Polars is immutable; with_columns returns a new frame sharing buffers with X -- no data copy.
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
    # (``X_t``/``X_q``/``X_te``/... -- internally ``pd.concat([X, new_cols])``) and the
    # accepted subset is folded into ``X`` while the intermediate stays bound to its own
    # distinct local name. None is reused or deleted, so at this point (the peak: every
    # float frame coexists, ``categorize_dataset`` is about to allocate the int-code
    # ``data`` on top) the process holds ~one full-frame copy PER family that ran.
    # These locals are provably dead here -- the only column data that must survive is in
    # ``X`` (consulted by categorize_dataset, the DCD ``X_raw=X`` path, and transform-time
    # recipe replay). Dropping them is SELECTION-NEUTRAL: ``data``/``cols``/``nbins`` and
    # every downstream MI estimate are computed from ``X`` alone, untouched by these names.
    # Names are bound only when that family ran and its helper returned (gated-off /
    # raised families never bind the name), so each drop is an explicit ``del`` guarded by
    # a membership check. For the ``X = X_<fam>`` rebind families the name is an ALIAS of
    # the live ``X`` and ``del`` only removes the alias (X survives); for the concat
    # families it frees a genuine separate full-width frame -- the actual memory win.
    # NOTE: ``del locals()[name]`` does NOT free a real local in CPython; a literal ``del``
    # statement is required, hence the explicit per-name lines below.
    _fe_live = set(locals())
    if "X_h" in _fe_live: del X_h
    if "X_e" in _fe_live: del X_e
    if "X_t" in _fe_live: del X_t
    if "X_q" in _fe_live: del X_q
    if "X_aa" in _fe_live: del X_aa
    if "X_ad" in _fe_live: del X_ad
    if "X_rt" in _fe_live: del X_rt
    if "X_df" in _fe_live: del X_df
    if "X_cb" in _fe_live: del X_cb
    if "X_boot" in _fe_live: del X_boot
    if "X_tg" in _fe_live: del X_tg
    if "X_ksg" in _fe_live: del X_ksg
    if "X_copula" in _fe_live: del X_copula
    if "X_dcor" in _fe_live: del X_dcor
    if "X_hsic" in _fe_live: del X_hsic
    if "X_jmim" in _fe_live: del X_jmim
    if "X_tc" in _fe_live: del X_tc
    if "X_cmim" in _fe_live: del X_cmim
    if "X_auto" in _fe_live: del X_auto
    if "X_ens" in _fe_live: del X_ens
    if "X_meta" in _fe_live: del X_meta
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
    # 2026-05-29 Wave 7: propagate the new ``nbins_strategy`` knob through to
    # categorize_dataset so per-column adaptive bin counts (FD, QS, MDLP, Knuth,
    # OptimalJoint, ...) actually take effect inside fit(). When None,
    # categorize_dataset uses the legacy fixed ``quantization_nbins``.
    _nbins_strategy = getattr(self, "nbins_strategy", None)
    _nbins_strategy_kwargs = getattr(self, "nbins_strategy_kwargs", None)
    # When capping cardinality for the compact-codes int8 goal, also bound the NUMERIC side: the supervised MDLP
    # (fayyad_irani) recursion can emit up to 2**max_depth intervals (default max_depth=8 -> ~256), which would exceed
    # int8 just like a high-card categorical. Cap max_depth to floor(log2(cap)) so numeric bins <= cap too (unless the
    # user pinned max_depth explicitly). This makes max_categorical_cardinality a single knob for a universally-narrow
    # codes matrix -- categorical tail folded AND numeric intervals bounded.
    _cap = getattr(self, "max_categorical_cardinality", None)
    if _cap and str(_nbins_strategy).lower() in ("mdlp", "fayyad_irani"):
        _md = max(2, int(np.floor(np.log2(int(_cap)))))
        _nbins_strategy_kwargs = dict(_nbins_strategy_kwargs or {})
        _nbins_strategy_kwargs.setdefault("max_depth", _md)
    # The supervised strategies (mdlp / optimal_joint) need y. Pull the raw
    # target column from the input frame -- categorize_dataset is called with
    # _x_for_cat which is a DataFrame; the target column is one of its members
    # (target injection happens upstream in _mrmr_fit_impl).
    _y_for_strategy = None
    if _nbins_strategy is not None and str(_nbins_strategy).lower() in (
        "mdlp", "fayyad_irani", "optimal_joint", "cv",
        "mah", "mah_sci", "sci", "marx",
    ):
        # Use the first target column as the supervised signal.
        if target_names:
            try:
                if hasattr(_x_for_cat, "to_numpy"):
                    _y_for_strategy = np.asarray(_x_for_cat[target_names[0]])
                else:
                    _y_for_strategy = np.asarray(_x_for_cat[target_names[0]])
            except Exception:
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

    # ``cols`` is a list; per-name ``cols.index`` is an O(len(cols)) scan, so resolving every target /
    # categorical name that way is O(C*P). Build a name->index map once and reuse it for both lookups.
    _name_to_idx = {c: i for i, c in enumerate(cols)}

    target_indices = np.array([_name_to_idx[col] for col in target_names], dtype=np.int64)

    # TARGET REBIN GUARD (2026-06-10). The adaptive per-column ``nbins_strategy``
    # (default ``"mdlp"`` since Wave 7) is meant for FEATURE columns; applied to the
    # injected TARGET column it is SELF-REFERENTIAL (MDLP bins y supervised on y) and
    # on a heavy-tailed continuous y it produces a DEGENERATE encoding -- measured on
    # the F2 fixture (y = 0.2*a**2/b + f/5 + log(2c)*sin(d/3), n=20000): 37 bins with
    # 83.7% of all rows collapsed into ONE bin (vs the clean 10 x 2000 equal-frequency
    # legacy quantile bins). Every downstream MI/CMI -- screening, pair gates, FE
    # acceptance -- is computed AGAINST these target codes, so the bulk of the signal
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
                logger.debug("suppressed in _fit_impl_core.py:5376: %s", e)
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
    # ids, so it fits the smallest int that spans its actual code range -- int8 for the common nbins<=~127 case, int16
    # for a high-cardinality categorical. The base (n, p) matrix at scale (e.g. 795k x 496) drops 4x / 2x vs the legacy
    # int32. Selection-EQUIVALENT: the code VALUES are unchanged, and every consumer (merge_vars, the GPU path) reads
    # this storage and casts UP to int32 for JOINT math, so deep joints (nbins^order) never overflow. Engineered-code
    # appends downstream re-narrow to this dtype (``_append_codes``). Range-checked directly (one min/max pass) rather
    # than trusting nbins semantics. Opt out: MLFRAME_MRMR_COMPACT_CODES=0.
    if data.size and os.environ.get("MLFRAME_MRMR_COMPACT_CODES", "1").strip().lower() not in ("0", "false", "off", "no"):
        try:
            _dmin = int(data.min()); _dmax = int(data.max())
            if -128 <= _dmin and _dmax <= 127:
                _store_dt = np.int8
            elif -32768 <= _dmin and _dmax <= 32767:
                _store_dt = np.int16
            else:
                _store_dt = None
            if _store_dt is not None and data.dtype.itemsize > np.dtype(_store_dt).itemsize:
                data = data.astype(_store_dt, copy=False)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _fit_impl_core.py:5422: %s", e)
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
        # (slice-replay corruption -- the same class as the smart_log BUG2 fix). They appear only in the
        # non-default "maximal" preset; dropping them here means they are never selected as engineered
        # features, while the create_*_transformations registry stays intact (other callers + the
        # registry-coverage test are unaffected). On the default "minimal" preset this is a no-op.
        _FE_NON_ROWWISE_PURE = ("grad1", "grad2", "logn")
        unary_transformations = {k: v for k, v in unary_transformations.items() if k not in _FE_NON_ROWWISE_PURE}
        binary_transformations = {k: v for k, v in binary_transformations.items() if k not in _FE_NON_ROWWISE_PURE}
        if fe_max_polynoms:
            # Generated polynomial coefficients are appended directly to unary_transformations under "poly_<coef>" keys;
            # no separate registry is needed. Use a seeded local Generator so the polynomial recipes are reproducible
            # across reruns with the same ``random_seed`` -- prior code used the global ``np.random`` stream, breaking
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

        engineered_features = set()
        checked_pairs = set()
    # engineered_recipes (name -> EngineeredRecipe) is initialised unconditionally; the splitter at the bottom
    # of fit() looks it up regardless of fe_max_steps. Stays empty when FE is disabled.
    engineered_recipes: dict = {}
    # PER-GATE FE REJECTION LEDGER (additive, 2026-06-11): the per-fit raw-record list is reset
    # near fit-start (above, before any FE stage records) so it accumulates the gate drops of
    # EVERY FE stage this fit -- the recipe-FE families + cluster-basis (which record before this
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
    # quantile-bin them into the candidate pool. Extracted from the ORIGINAL ``X`` (NaN visible) -- NOT the
    # ffill'd ``_x_for_cat`` -- so a NaN-bearing column is correctly skipped at fit (v1 has no NaN bin in the
    # quantile-edge replay) and fit/transform stay consistent (both read the user's raw frame).
    _num_raw_values = None
    if cat_fe_cfg.enable and getattr(cat_fe_cfg, "include_numeric", False):
        from ..engineered_recipes._recipe_extract import _extract_column as _extract_col_for_num
        _cat_idx_set = set(int(c) for c in categorical_vars)
        _tgt_idx_set = set(int(t) for t in target_indices)
        # RAW input columns only: pre-FE recipes (haar / ratio / grouped-agg ...) appended engineered numeric
        # columns to data / cols / X before this step. Crossing those is unreplayable -- the engineered source
        # is absent from the user's raw frame at transform time -> NaN column / silent feature drop. Restrict to
        # ``feature_names_in_`` (the raw user columns, set above, excludes engineered names).
        _raw_name_set = set(getattr(self, "feature_names_in_", None) or [])
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
                logger.debug("suppressed in _fit_impl_core.py:5600: %s", e)
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
        # target_encoding) feature collapses to a CONSTANT column at serving time -- a silent train/serve skew (the
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
            # build; a per-column flag would off-by-one the NaN-free partner of a NaN-bearing column -- the
            # same silent train/serve skew, for the mixed-block case the per-column path never handled.
            _block_has_nan: bool | None = None
            try:
                _cat_block = _x_for_cat.select_dtypes(include=("category", "object", "string", "bool"))
                if _cat_block.shape[1] > 0:
                    _block_has_nan = bool(_cat_block.isna().to_numpy().any())
            except Exception:
                _block_has_nan = None
            _src_map_cache: dict = {}
            for _ri, r in enumerate(cat_fe_state.recipes):
                _maps_for_recipe: dict = {}
                for _src in getattr(r, "src_names", ()) or ():
                    if _src not in _src_map_cache:
                        if _src in _x_for_cat.columns:
                            try:
                                _src_map_cache[_src] = _build_cat_code_map(_x_for_cat[_src], block_has_nan=_block_has_nan)
                            except Exception:
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
                        logger.debug("suppressed in _fit_impl_core.py:5680: %s", e)
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
    # 2026-06-02: tracks whether the post-FE confirming re-screen has run, so it
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
    while True:
        n_recommended_features = 0
        times_spent = defaultdict(float)
        # Resolve the fit's ONE shared row draw BEFORE the screen so the order-1 relevance sweep + FDR
        # floor score on it (screen is the first consumer -> caches the draw -> the FE step reuses the
        # SAME rows). None at small n -> full-n screen, unchanged.
        try:
            from .._fe_sufficient_summary import _get_shared_fe_subsample_idx
            _screen_shared_idx = _get_shared_fe_subsample_idx(self, np.asarray(data[:, int(target_indices[0])]), int(len(data)))
        except Exception as _sub_exc:
            # Full-n fallback is safe but ~33x slower at n~1M -> log so it is never a silent mystery.
            logger.warning("mrmr: shared FE subsample resolution failed; screening at FULL n: %r", _sub_exc, exc_info=True)
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
        ) = screen_predictors(
            factors_data=data,
            y=target_indices,
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
            parallel_kwargs=self.parallel_kwargs,
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
                    # Layer 47 (2026-05-31): forward the auto-tau
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
        )
        if _dcd_state is not None:
            _persisted_dcd_state = _dcd_state
        # 2026-05-30 Wave 9 — stash DCD summary on the estimator for the
        # public ``dcd_`` attribute (None when DCD was disabled).
        try:
            from .._dynamic_cluster_discovery import dcd_summary as _dcd_summary
            self.dcd_ = _dcd_summary(_dcd_state)
        except Exception:
            self.dcd_ = None
        # Layer 41 (2026-05-31): self-describing cluster membership accessor.
        # Mirror ``dcd_["cluster_anchors_names"]`` onto the estimator as a
        # first-class fitted attribute so downstream code can read the
        # discovered clusters without indexing through ``self.dcd_`` (the
        # raw summary dict). ``cluster_members_`` is None when DCD was
        # disabled, matching ``dcd_`` semantics. Pure additive metadata --
        # no effect on ``support_`` or ``transform`` output.
        if isinstance(self.dcd_, dict):
            self.cluster_members_ = dict(self.dcd_.get("cluster_anchors_names", {}))
        else:
            self.cluster_members_ = None
        # Layer 48 (2026-05-31): hierarchical post-hoc cluster map. Pure
        # additive analyser over ``dcd_["cluster_anchors_names"]`` --
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
            except Exception:
                self.cluster_hierarchy_ = {}
        else:
            self.cluster_hierarchy_ = None
        # 2026-05-30 Wave 9.1 fix (loop iter 1, agent-found bug):
        # When DCD's ``commit_swap`` extended ``factors_data`` inside screen
        # with PC1 aggregate columns, the swap targets land in ``selected_vars``
        # at indices >= len(nbins) here -- the outer-scope ``data/cols/nbins``
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
            except Exception:  # nosec B110 - non-trivial body
                # Best-effort -- if DCDState is malformed, fall through.
                pass

        if fe_max_steps == 0 or num_fs_steps >= fe_max_steps:
            break

        if self.max_runtime_mins is not None:
            elapsed_min = (timer() - start_time) / 60.0
            if elapsed_min >= self.max_runtime_mins:
                ran_out_of_time = True
                if verbose:
                    logger.info("MRMR.fit: runtime budget %.1f min exceeded at FE step %d; stopping.", self.max_runtime_mins, num_fs_steps)
                break

        # SUFFICIENT-SUMMARY EARLY-STOP (backlog #22, DEFAULT-ON). The user's
        # "compare-to-theoretical-max" idea via a DPI residual test. Once the
        # current selection already captures all the information the observables
        # carry about y -- i.e. the residual r = y - E_hat[y|selected] is pure
        # noise w.r.t. EVERY raw feature (all raws at the maxT permutation null)
        # AND small relative to y (Var(r)/Var(y) guard) -- any future engineered
        # candidate is, by the Data-Processing Inequality, a function of the raws
        # and CANNOT have more MI with r than the raws do, so the remaining FE
        # search is provably pointless. Skip it. This NEVER changes the final
        # selection (it only skips work that could find nothing -- with it OFF the
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
                target_indices=target_indices,
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
        # ``fe_min_engineered_mi_prevalence`` gate -- pair-level MI is
        # near the individual-MI sum and the engineered candidate
        # cannot beat 98% of pair MI. Retry ONCE with relaxed
        # thresholds (and fe_smart_polynom_iters=0 to skip the
        # already-completed expensive Hermite Optuna phase).
        _adaptive = bool(getattr(self, "fe_adaptive_threshold_relax", True))
        _relax_factor = float(getattr(self, "fe_adaptive_relax_factor", 0.9))
        if n_recommended_features == 0 and _adaptive and fe_max_steps > 0 and num_fs_steps == 0:  # only on the very first FE step
            _relaxed_engineered = fe_min_engineered_mi_prevalence * _relax_factor
            _relaxed_pair = max(1.001, fe_min_pair_mi_prevalence * _relax_factor)
            if verbose:
                logger.info(
                    "MRMR FE: first pass found 0 engineered features; "
                    "retrying with relaxed thresholds "
                    "(engineered_mi_prevalence: %.3f -> %.3f, "
                    "pair_mi_prevalence: %.3f -> %.3f). "
                    "Skipping Hermite Optuna re-run (already cached in "
                    "_hermite_features_).",
                    fe_min_engineered_mi_prevalence, _relaxed_engineered,
                    fe_min_pair_mi_prevalence, _relaxed_pair,
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
            # CONFIRM-RESCREEN (2026-06-02): the FE step appended engineered
            # columns and (legacy) promoted them into ``selected_vars`` BY FIAT,
            # bypassing redundancy filtering + gain accounting. Instead of
            # breaking here, loop ONCE more so the top-of-loop ``screen_predictors``
            # re-selects from the AUGMENTED pool. The engineered columns are
            # already quantised bin-code columns in ``data``/``cols``/``nbins``,
            # so MRMR treats them as ordinary candidates: a redundant engineered
            # feature (e.g. ``1/b - d**2`` whose conditional MI given an
            # already-selected ``a**2/b`` is ~0.03) is dropped by the Fleuret
            # redundancy term, and every surviving column -- raw OR engineered --
            # earns a real ``mrmr_gain`` / ``support_rank``. The next iteration
            # hits the ``num_fs_steps >= fe_max_steps`` break at the TOP of the
            # loop (line ~5085) BEFORE the FE step, so FE never runs again -- no
            # unbounded recursion, no new engineered columns.
            if getattr(self, "fe_reselect_after_engineering", True) and n_recommended_features > 0 and not _did_confirm_rescreen:
                _did_confirm_rescreen = True
                continue
            break  # uncomment to avoid recheck of single-rounded FE

    # ENGINEERED-OPERAND FEED-FORWARD (2026-06-08): the continuous engineered-value
    # store is FIT-TIME SCRATCH (full-length float64 arrays of training data) used
    # only to feed engineered operands into the next FE step's pair search. Drop it
    # once the FE loop is done so it never bloats the fitted estimator or breaks
    # pickle (the replayable composite carries only its parent recipes, never these
    # arrays). No-op when the attr was never created (no engineered columns).
    # SNAPSHOT FIRST (2026-06-08): the raw-vs-engineered conditional-redundancy drop
    # below needs the CONTINUOUS engineered values to bin the engineered survivor
    # finely (the ``data`` matrix holds only the lossy ~10-code screening bins, which
    # leave a fully-subsumed denominator operand a spurious residual CMI). Snapshot
    # into a LOCAL (never an attr -> stays out of the pickled estimator) so the del
    # below still keeps the fitted object lean.
    _eng_continuous_snapshot = dict(getattr(self, "_engineered_continuous_", None) or {})
    if hasattr(self, "_engineered_continuous_"):
        try:
            del self._engineered_continuous_
        except Exception:
            self._engineered_continuous_ = {}

    if verbose > 2:
        logger.info("time spent by binary func: %s", sort_dict_by_value(times_spent))
    # Possibly decide on eliminating original features? (if constructed ones cover 90%+ of MI)

    # ---------------------------------------------------------------------------------------------------------------
    # Drop temporary targets
    # ---------------------------------------------------------------------------------------------------------------

    # Fuzz-caught: previous ``X = X.drop(columns=target_names)`` returned a new DataFrame and only rebound the
    # local; for pandas input (where X.loc[:, target_names] = ... mutated the caller's frame), the caller's
    # X was left with the injected ``targ_<id>`` columns, which leaked into downstream sklearn pipeline
    # (imputer/scaler recorded them in feature_names_in_ and raised on transform). Fix: drop in place (pandas)
    # or rebind (polars -- immutable, caller's X was never mutated).
    if _is_polars_input:
        X = X.drop(target_names)  # no-copy lazy op; caller's X untouched
    else:
        # option_context silences the conservative SettingWithCopy heuristic (fires when the caller passed a sliced
        # view); the in-place drop reverses this function's own targ_<id> injection on the same object, no copy.
        with pd.option_context("mode.chained_assignment", None):
            X.drop(columns=target_names, inplace=True)  # restores caller's original schema

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
    # list flows through that same remap into ``support_``. Never allowed to break fit -- guarded.
    # ---------------------------------------------------------------------------------------------------------------
    self.friend_graph_ = None
    # ``len(...)`` not truthiness: by this point ``selected_vars`` may be a numpy array (the empty-screen
    # FE fallback rebinds it), and ``and <array>`` raises "truth value ... ambiguous". Empty list AND empty
    # array both give len 0, so the guard reads "build the graph only when something was selected".
    # build_friend_graph defaults OFF (diagnostic-display only); friend_graph_prune REQUIRES the graph, so auto-build
    # it whenever pruning is on even if the diagnostic build was left off.
    if (getattr(self, "build_friend_graph", False) or getattr(self, "friend_graph_prune", False)) and len(selected_vars) > 0:
        try:
            from ..friend_graph import build_friend_graph as _build_fg, prune_by_friend_graph as _prune_fg

            _fg = _build_fg(
                selected_vars=selected_vars,
                factors_data=data,
                factors_nbins=nbins,
                target_indices=target_indices,
                feature_names=cols,
                mi_eps=self.friend_graph_mi_eps,
                edge_significance=self.friend_graph_edge_significance,
                garbage_min_degree=self.friend_graph_garbage_min_degree,
                garbage_unique_ratio=self.friend_graph_unique_ratio,
                unique_max_degree=self.friend_graph_unique_max_degree,
                max_nodes=self.friend_graph_max_nodes,
                seed=self.random_seed,
                gpu_backend=getattr(self, "friend_graph_gpu_backend", None),
            )
            if self.friend_graph_prune:
                # Protect cluster-aggregate columns from pruning: they are correlated with all their
                # members by construction, so the sink classifier could mis-flag them.
                _ca_protect = [v for v in selected_vars if getattr(engineered_recipes.get(cols[v]), "kind", None) == "cluster_aggregate"]
                _pruned, _reasons = _prune_fg(_fg, selected_vars, protect_indices=_ca_protect)
                if _reasons:
                    if verbose:
                        logger.info(
                            "MRMR friend-graph pruned %d suspected-sink feature(s): %s",
                            len(_fg.pruned), _fg.pruned,
                        )
                    selected_vars = _pruned
            self.friend_graph_ = _fg
        except Exception as _fg_exc:
            logger.warning(
                "MRMR friend-graph post-analysis failed (%s: %s); continuing without it.",
                type(_fg_exc).__name__, _fg_exc,
            )

    # Clustered-feature aggregation, replace mode: drop the aggregated cluster MEMBERS from
    # selected_vars (cols-space) so only the denoised aggregate survives into support_. Idempotent
    # set-difference (composes with the friend-graph prune above). The aggregate itself is an
    # engineered name and is routed into _engineered_recipes_ by the remap below.
    _ca_removed = getattr(self, "_cluster_aggregate_removals_", None)
    if _ca_removed:
        _removed_set = set(_ca_removed)
        selected_vars = [v for v in selected_vars if cols[v] not in _removed_set]

    # STANDALONE CROSS-GROUP GATE PRUNE (2026-06-15; un-gated to BOTH paths 2026-06-22). The conditional-gate
    # pre-pass appends gate_mask columns into the screening pool and the greedy can select a STANDALONE one;
    # the FE-step cross-group prune cannot reach it (it filters prospective_additions, not selected_vars).
    # Drop a selected standalone gate column iff its gate pair is CROSS-GROUP (no single clean ENGINEERED
    # survivor jointly covers its raw sources) AND those sources are already covered by the clean survivors'
    # union -- the spurious gate_mask__c__b / gate_mask__b__d on y=a**2/b+log(c)*sin(d) (b,d from the two
    # different groups, both already in div(sqr(a),neg(b))+mul(log(c),sin(d))). A genuine warped (c,d) carrier
    # is WITHIN-pair (or embedded in a composite, not a standalone gate name) so it is KEPT. Originally scoped
    # to fe_fast_search on the assumption the exhaustive path's extra passes would remove these; they do NOT
    # (the canonical exhaustive fit selected TWO such standalone cross-group gates -> over the <=4 cap). The
    # discriminator is purely structural and never empties the support, so it now runs in BOTH paths.
    if len(selected_vars) and getattr(self, "_gate_col_src_vars_", None):
        try:
            import re as _re_sg
            _gmap_sg = dict(self._gate_col_src_vars_)
            _tok_sg = _re_sg.compile(r"(?<![A-Za-z0-9_])([a-z](?:[a-z]?\d+)?)(?![A-Za-z0-9_])")
            _sel_names_sg = [cols[v] for v in selected_vars]
            # Clean engineered survivors = selected composites that are NOT a bare gate column and NOT raw.
            _clean_tok_sets_sg = [set(_tok_sg.findall(nm)) for nm in _sel_names_sg if nm not in _gmap_sg and ("(" in nm) and ("gate_mask" not in nm)]
            _clean_union_sg = set().union(*_clean_tok_sets_sg) if _clean_tok_sets_sg else set()
            # "Genuine single-pair carrier" anchors for the within-one test are PURE-PAIR survivors only
            # (<= 2 distinct raw vars). A FULL-TARGET fused compound spans every var ({a,b,c,d}) and would
            # otherwise make EVERY gate look within-one, defeating the cross-group test -- so the canonical
            # ``add(sqrt(div(sqr(a),neg(b))),sin(mul(log(c),sin(d))))`` is excluded as an anchor; the clean
            # pure pairs ``div(sqr(a),neg(b))`` / ``mul(log(c),sin(d))`` are the real carriers and a
            # cross-group gate over {b,d} (whose pair no single pure survivor covers) is correctly dropped.
            _pair_tok_sets_sg = [_ts for _ts in _clean_tok_sets_sg if len(_ts) <= 2]
            _drop_sg = set()
            for nm in _sel_names_sg:
                if nm not in _gmap_sg:
                    continue  # only standalone bare gate columns
                _src = set(str(s) for s in _gmap_sg.get(nm, ()))
                if len(_src) < 2:
                    continue
                _within_one = any(_src <= _ts for _ts in _pair_tok_sets_sg)
                if (not _within_one) and _src and _src <= _clean_union_sg:
                    _drop_sg.add(nm)
            if _drop_sg:
                selected_vars = [v for v in selected_vars if cols[v] not in _drop_sg]
                if getattr(self, "verbose", 0):
                    logger.info(
                        "MRMR FE fast-search: pruned %d standalone cross-group gate column(s) covered by "
                        "clean engineered survivors: %s", len(_drop_sg), sorted(_drop_sg),
                    )
        except Exception as _sg_exc:
            logger.warning("MRMR fast-search standalone-gate prune skipped (%s); continuing.", type(_sg_exc).__name__)

    # N-WAY SYNERGY SEEDING (2026-06-17). The greedy screen assembles features one-at-a-time by
    # CONDITIONAL gain, which cannot climb a PURE-synergy gradient: on a 3-way XOR every operand has
    # ~0 marginal AND ~0 conditional gain until ALL members are present, so the genuine {x0,x1,x2}
    # interaction is never assembled (test_3way_screening, documented tracked-red whose specced fix is
    # exactly this -- evaluate the n-way JOINT directly and surface it). When the user opted into n-way
    # interactions (interactions_max_order>=2), evaluate candidate raw COMBOS by their MM-corrected
    # JOINT MI vs the SUM of member marginals and SEED the members of combos showing strong synergy +
    # clearing an absolute joint-MI floor. The Miller-Madow correction keeps a noise combo's joint MI
    # ~0 (verified: on 3-way XOR among 5 noise vars ONLY {x0,x1,x2} fires), so noise is never seeded.
    # Off when interactions_max_order<2 (the default) -> byte-identical there. Bounded combo
    # enumeration (candidate + order caps) so wide-p fits stay tractable.
    if int(getattr(self, "interactions_max_order", 1) or 1) >= 2 and len(selected_vars) >= 0:
        try:
            from .._fe_synergy_screen import detect_synergy_combos
            _raw_set_syn = set(self.feature_names_in_)
            _cand_syn = [i for i, _nm in enumerate(cols) if _nm in _raw_set_syn]
            if 2 <= len(_cand_syn) <= 60:
                _yc_syn = np.asarray(classes_y).astype(np.int64).ravel()
                _code_cols_syn = {i: np.asarray(data[:, i]).astype(np.int64).ravel() for i in _cand_syn}
                _combos_syn = detect_synergy_combos(
                    _code_cols_syn, _yc_syn, _cand_syn,
                    max_order=int(getattr(self, "interactions_max_order", 3) or 3),
                    min_order=max(2, int(getattr(self, "interactions_min_order", 2) or 2)),
                )
                _sv_syn = set(selected_vars)
                _seed_syn = []
                for _combo_syn, _jmi_syn in _combos_syn:
                    for _ci_syn in _combo_syn:
                        if _ci_syn not in _sv_syn:
                            _seed_syn.append(_ci_syn)
                            _sv_syn.add(_ci_syn)
                if _seed_syn:
                    selected_vars = list(selected_vars) + _seed_syn
                    if verbose:
                        logger.info(
                            "MRMR n-way synergy seeding: added %d raw operand(s) of synergy combo(s) the "
                            "greedy could not assemble (joint MI >> sum of marginals): %s",
                            len(_seed_syn), [cols[i] for i in _seed_syn],
                        )
        except Exception as _syn_exc:
            logger.warning("MRMR n-way synergy seeding failed: %s; keeping support.", _syn_exc)

    # RAW-RETENTION (2026-06-03): re-add SCREENING-confirmed genuine raw features
    # that the post-FE re-selection dropped, UNLESS a SINGLE-PARENT engineered child
    # substitutes them (the prefer-engineered raw->transform swap, which is a
    # legitimate, intended replacement). Screening permutation-validated these raw
    # columns as genuine; at small n an engineered feature can absorb a weak genuine
    # one as a redundant near-duplicate and the re-selection then drops the clean raw
    # signal entirely (measured: a genuine X5 at n=500, and both operands of a
    # pair-interaction target, dropped from support_). A raw feature only legitimately
    # leaves the support when a sole-parent transform of it survives.
    _prefe_raw = getattr(self, "_prefe_screened_raw_", None)
    if _prefe_raw and len(selected_vars):
        from .._confirm_predictor import _extract_single_raw_parent  # noqa: E501
        _raw_names_set = set(self.feature_names_in_)
        _cur_names = set(np.asarray(cols)[np.asarray(selected_vars, dtype=np.intp)])
        # Raw parents already represented by a SOLE-parent engineered survivor:
        _substituted = set()
        for _v in selected_vars:
            if cols[_v] in _raw_names_set:
                continue
            _p = _extract_single_raw_parent([_v], cols, _raw_names_set)
            if _p is not None:
                _substituted.add(_p)
        # Cluster members folded into a denoised MULTI-parent aggregate (cluster_aggregate 'replace' mode -> _cluster_aggregate_removals_, or a DCD PC1/mean_z swap -> cluster_members_) are
        # ALREADY represented by that aggregate. _extract_single_raw_parent only recognises a SOLE-parent transform substitute, so without this exclusion raw-retention would resurrect the
        # very members 'replace' mode just removed and re-inject the redundancy the aggregation collapsed. Same exclusion the additional-RFECV rescue pool applies below.
        for _ca_member in getattr(self, "_cluster_aggregate_removals_", None) or []:
            _substituted.add(_ca_member)
        _cm_for_raw_retention = getattr(self, "cluster_members_", None)
        if isinstance(_cm_for_raw_retention, dict):
            for _anchor, _members in _cm_for_raw_retention.items():
                _substituted.add(_anchor)
                if isinstance(_members, (list, tuple, set)):
                    _substituted.update(_members)
        # MULTI-PARENT OPERAND SCOPE (2026-06-08 regression fix): a raw feature that
        # is an OPERAND of a SURVIVING multi-parent engineered feature is NOT covered
        # by the sole-parent ``_substituted`` exclusion above, so the original blanket
        # re-add resurrected EVERY such operand -- including ones whose entire signal
        # flowed into the engineered child (e.g. ``y = a**2/b + log(c)*sin(d)``: raw
        # ``a, c, d`` carry NO information about ``y`` beyond ``div(sqr(a),abs(b))`` and
        # ``mul(log(c),sin(d))``, yet were re-added with ``support_rank -1`` and no gain,
        # padding the support with three redundant columns). The post-FE re-selection
        # ALREADY judged them redundant via the Fleuret conditional-MI redundancy term.
        # We restore the OLD CORRECT behaviour by deferring to that verdict for such
        # operands at large n (where the conditional-MI estimate is reliable), while
        # keeping the protective unconditional re-add at small n (the regime the
        # protection was built and validated for) and for raws NOT consumed by any
        # surviving engineered feature (the originally-intended absorbed-by-unrelated case).
        from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _RR_TOK_SPLIT
        # Map each raw-operand name -> list of surviving ENGINEERED survivor column indices that consume it.
        _eng_operands_of = {}  # raw_name -> list[engineered survivor col idx]
        for _v in selected_vars:
            _vname = cols[_v]
            if _vname in _raw_names_set:
                continue
            for _tok in _RR_TOK_SPLIT.split(_vname):
                if not _tok:
                    continue
                _base = _tok if _tok in _raw_names_set else (_tok.split("__", 1)[0] if "__" in _tok else None)
                if _base in _raw_names_set:
                    _eng_operands_of.setdefault(_base, []).append(_v)
        # Sample-size scope (2026-06-08): the small-n regime the protection was BUILT and
        # validated for (n=500 / 2000 / 3000 fixtures). At large n the post-FE re-selection's
        # conditional-MI redundancy term is statistically reliable -- its drop of a redundant
        # operand IS the OLD CORRECT behaviour -- so we do NOT override it there. ``_RR_PROTECT_MAX_N``
        # sits well above the largest validated fixture (3000) and far below the regression case (1e5).
        _RR_PROTECT_MAX_N = int(getattr(self, "fe_raw_retention_max_n", 20000) or 0)
        _n_rows_rr = int(data.shape[0])

        def _rr_raw_is_relevant_given_engineered(_raw_idx, _eng_cols):
            """Whether a raw operand of a surviving engineered child carries signal the
            engineered set does NOT capture, so raw-retention should OVERRIDE the re-selection's
            redundancy drop. Two regimes:

            * small n (``n <= _RR_PROTECT_MAX_N``): the conditional-MI redundancy estimate the
              re-selection used is unreliable at small n (the protection's whole reason to exist),
              so keep the protective re-add unconditionally -- preserves the n<=3000 contracts.
            * large n: the re-selection's conditional-MI redundancy verdict is trustworthy, so we
              DEFER to it -- an operand it dropped is genuinely redundant given the engineered
              child (``a`` in ``div(sqr(a),abs(b))`` for ``y=a**2/b`` carries no signal about ``y``
              beyond the ratio). We do NOT re-add it, restoring the pre-2026-06-03 selection. Note a
              bare ``CMI >= relevance_floor`` check does NOT work here: a coarsely-binned (~10-bin)
              engineered child leaves a small but above-floor residual conditional MI on its operand
              purely from the binning gap (measured: redundant ``a/c/d`` sit at CMI 0.002-0.023, the
              floor is ~0.0013), so the absolute floor cannot separate residual-binning-noise from a
              real independent term -- only the re-selection's RELATIVE redundancy criterion can, and
              it already ran.

            CMI-estimator import/edge failures fall back to the protective re-add (never drop a
            screening-confirmed raw on an estimator error)."""
            if not _eng_cols:
                return True  # absorbed by an UNRELATED engineered feature -> original intent
            if _n_rows_rr <= _RR_PROTECT_MAX_N:
                return True  # small-n protective regime: keep the unconditional re-add
            # large n: defer to the re-selection's redundancy drop for engineered operands.
            return False

        # PERMUTATION-SIGNIFICANCE GATE on the re-add (2026-06-08): a raw column the
        # screen flagged as ``_prefe_screened_raw_`` can be a small-n FALSE POSITIVE --
        # the coarse-binning plug-in MI is upward-biased, so a PURE-NOISE column (one
        # NOT in the target equation, e.g. CC4's ``e`` in ``y=log(a)*c+0.4*f``) can leave
        # a tiny residual debiased MI that the screen confirms and retention then
        # re-injects, padding the support with noise. Gate the re-add on the SAME
        # within-data permutation-significance test the empty-RAW rescue uses (computed on
        # the screen's own ``data`` / ``nbins`` so it matches ``cached_MIs``): a candidate
        # that sits WITHIN its own null (p >= alpha) is genuine-screen noise and is NOT
        # re-added. A genuinely weak-BUT-real raw (above its null) still passes. Best-
        # effort: a kernel failure falls through to the permissive re-add (never drop a
        # screening-confirmed raw on an estimator error).
        try:
            from ..permutation import mi_direct as _mi_direct_rr
        except Exception:
            _mi_direct_rr = None
        _rr_signif_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
        _rr_q_dtype = getattr(self, "quantization_dtype", np.int32)

        def _rr_raw_is_significant(_idx):
            """True iff the raw column at cols-index ``_idx`` sits ABOVE its permutation
            null against y (genuine signal). Pure-screen-noise sits within (p>=alpha)."""
            if _mi_direct_rr is None:
                return True
            try:
                _sig = _mi_direct_rr(
                    data, x=np.array([int(_idx)], dtype=np.int64), y=target_indices,
                    factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                    return_null_mean=True, parallelism="none", dtype=_rr_q_dtype, prefer_gpu=False,
                )
                return float(_sig[3]) < _rr_signif_alpha
            except Exception:
                return True  # significance unavailable -> permissive re-add

        _sv_set = set(selected_vars)
        # C2 ADDITIVE-FUSION EXCLUSION (2026-06-24): a raw operand the FE step's
        # additive-fusion proposer judged FULLY subsumed by the fused ``add(...)`` compound
        # (recorded in ``_raw_redundancy_dropped_`` via the production keep-probe) must NOT
        # be resurrected here -- the fused compound carries its additive term, so re-adding
        # it would re-inject a redundant single-group fragment beside the clean compound
        # (the FUSION-blocked goal's leftover raw). The fusion ran the same n-invariant
        # conditional-excess verdict ``drop_redundant_raw_operands`` uses, so this is the
        # authoritative drop. Byte-identical when no fusion fired (the set is empty).
        _fused_dropped_raw = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
        _readd = []
        _dropped_redundant = []
        _dropped_insignificant = []
        for _rn in _prefe_raw:
            if _rn in _cur_names or _rn in _substituted:
                continue
            if _rn in _fused_dropped_raw:
                _dropped_redundant.append(_rn)
                continue
            try:
                _idx = cols.index(_rn)
            except ValueError:
                continue
            if _idx in _sv_set:
                continue
            _eng_cols = _eng_operands_of.get(_rn)
            if _eng_cols and not _rr_raw_is_relevant_given_engineered(_idx, _eng_cols):
                # Fully captured by a surviving engineered child -> respect the
                # re-selection's redundancy verdict (the OLD CORRECT behaviour).
                _dropped_redundant.append(_rn)
                continue
            if not _rr_raw_is_significant(_idx):
                # Screen false positive (pure noise within its own null) -> do not re-add.
                _dropped_insignificant.append(_rn)
                continue
            _readd.append(_idx)
            _sv_set.add(_idx)
        if _dropped_insignificant and verbose:
            logger.info(
                "MRMR raw-retention: withheld %d screening-flagged raw feature(s) that "
                "sit WITHIN their permutation null (p>=%.2f -- genuine-screen noise, not "
                "re-added): %s",
                len(_dropped_insignificant), _rr_signif_alpha, _dropped_insignificant,
            )
        if _readd:
            selected_vars = list(selected_vars) + _readd
            if verbose:
                logger.info(
                    "MRMR raw-retention: re-added %d screening-confirmed raw feature(s) "
                    "dropped by the post-FE re-selection (carry conditional signal beyond "
                    "their engineered children): %s",
                    len(_readd), [cols[i] for i in _readd],
                )
        if _dropped_redundant and verbose:
            logger.info(
                "MRMR raw-retention: kept %d raw feature(s) DROPPED -- fully captured by a "
                "surviving engineered child (conditional MI given the engineered set below "
                "the relevance floor): %s",
                len(_dropped_redundant), _dropped_redundant,
            )

    # ADAPTIVE-FOURIER PROTECTION (2026-06-03): re-add held-out-validated
    # ADAPTIVE Fourier columns the MRMR screen dropped. The adaptive detector
    # already confirmed the column's dominant frequency on a held-out slice;
    # the screen drops it anyway because a SINGLE sin OR cos has low marginal MI
    # (the phase is split across the two legs, so neither alone clears the
    # relevance floor and the screen prefers a lower-MI fixed-freq twin). We
    # re-add the index of every adaptive name that is a column in ``cols`` but
    # absent from ``selected_vars``; its recipe is already in
    # ``engineered_recipes`` (merged from ``_hybrid_orth_pre_recipes`` above)
    # and survives into ``self._engineered_recipes_`` via the remap below, so
    # transform() replays the fit-time column byte-for-byte. Runs BEFORE the
    # ``selected_vars_names`` remap so the re-added index is routed correctly.
    _adaptive_fourier = getattr(self, "_adaptive_fourier_features_", None)
    if _adaptive_fourier and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _readd_adaptive = []
        for _an in _adaptive_fourier:
            _idx = _cols_index.get(_an)
            if _idx is None:
                continue
            if _idx not in _sv_set:
                _readd_adaptive.append(_idx)
                _sv_set.add(_idx)
        if _readd_adaptive:
            selected_vars = list(selected_vars) + _readd_adaptive
            if verbose:
                logger.info(
                    "MRMR adaptive-fourier protection: re-added %d held-out-" "validated adaptive Fourier feature(s) dropped by the screen: %s",
                    len(_readd_adaptive),
                    [cols[i] for i in _readd_adaptive],
                )

    # MISSINGNESS-INDICATOR PROTECTION (2026-06-04): re-add the clean ``is_missing__{col}`` indicator the MRMR screen dropped IN FAVOUR OF its raw source. Under ``nan_strategy='separate_bin'``
    # the raw column's NaN bin already encodes the MNAR pattern, so the binned MI of the indicator and the raw source are near-identical (a true tie); the greedy screen keeps the raw column
    # and discards the indicator as redundant. But the raw column is mostly NaN -- the downstream model cannot consume the missingness signal from it, only from the standalone numeric
    # indicator (the whole point of Layer 37). When the raw source IS selected, the indicator carries the SAME signal in a clean, model-ready form, so we re-add it. Gating on "the raw source
    # survived the screen" keeps a pure-noise indicator (MAR column the screen never selects) out of support. The count / pattern encoders have no single raw source and are screened normally.
    _miss_indicators = list(getattr(self, "missingness_indicator_features_", None) or [])
    if _miss_indicators and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _sel_names_now = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        _readd_miss = []
        for _mn in _miss_indicators:
            _idx = _cols_index.get(_mn)
            if _idx is None or _idx in _sv_set:
                continue
            _rec_mi = _miss_ind_pre_recipes.get(_mn)
            _src_mi = tuple(getattr(_rec_mi, "src_names", ()) or ())
            # Re-add only when the indicator's raw source survived the screen (i.e. the signal is real and the screen kept the redundant raw twin in its place).
            if _src_mi and _src_mi[0] in _sel_names_now:
                _readd_miss.append(_idx)
                _sv_set.add(_idx)
        if _readd_miss:
            selected_vars = list(selected_vars) + _readd_miss
            if verbose:
                logger.info(
                    "MRMR missingness-indicator protection: re-added %d clean "
                    "is_missing__ indicator(s) the screen dropped in favour of "
                    "the redundant raw NaN-bin source: %s",
                    len(_readd_miss), [cols[i] for i in _readd_miss],
                )

    # HINGE / CHANGE-POINT DEFERRED MATERIALISATION (2026-06-09): the hinge stage
    # ran BEFORE the pair-FE loop (it needs the raw source columns) but DEFERRED
    # appending its legs so they could not perturb composite recovery. Now that the
    # FE loop has settled (composites recovered untouched), materialise the buffered
    # legs into the candidate matrix (``data`` bin-codes / ``cols`` / ``nbins``),
    # the augmented frame ``X``, and the recipe registry, then let the protection
    # block below re-add the deserving ones into ``selected_vars``. Skipped wholesale
    # when nothing was detected (legacy / no-kink path: the buffer is empty).
    if _hinge_deferred_values and isinstance(X, pd.DataFrame):
        try:
            from ..mrmr import discretize_array
            _hinge_added_names = []
            _n_cols_before_hinge = len(cols)
            _new_hinge_codes = []
            _new_hinge_nbins = []
            for _hn, _vals in _hinge_deferred_values.items():
                if _hn in X.columns:
                    continue  # already present (defensive)
                _vals = np.asarray(_vals, dtype=np.float64)
                if _vals.shape[0] != data.shape[0]:
                    continue
                _codes = discretize_array(
                    arr=_vals,
                    n_bins=self.quantization_nbins,
                    method=self.quantization_method,
                    dtype=self.quantization_dtype,
                )
                _new_hinge_codes.append(np.asarray(_codes).reshape(-1, 1))
                _new_hinge_nbins.append(int(self.quantization_nbins))
                X[_hn] = _vals
                cols = cols + [_hn]
                _hinge_added_names.append(_hn)
                _r = _hinge_deferred_recipes.get(_hn)
                if _r is not None:
                    _hybrid_orth_pre_recipes[_hn] = _r
                    engineered_recipes[_hn] = _r
            if _new_hinge_codes:
                data = np.append(
                    data, np.hstack(_new_hinge_codes).astype(data.dtype), axis=1,
                )
                nbins = np.concatenate(
                    [
                        np.asarray(nbins),
                        np.asarray(_new_hinge_nbins, dtype=nbins.dtype),
                    ]
                )
                self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_hinge_added_names)
                self._hinge_features_ = list(getattr(self, "_hinge_features_", None) or []) + list(_hinge_added_names)
                if verbose:
                    logger.info(
                        "MRMR.fit hinge change-point FE: materialised %d deferred " "leg(s) post-loop: %s",
                        len(_hinge_added_names),
                        _hinge_added_names[:8],
                    )
        except Exception as _h_mat_exc:
            logger.warning(
                "MRMR.fit hinge deferred materialisation raised %s: %s; " "continuing without hinge columns.",
                type(_h_mat_exc).__name__,
                _h_mat_exc,
            )

    # HINGE / CHANGE-POINT PROTECTION (2026-06-09): re-add the held-out-tau-
    # validated hinge legs the MRMR screen dropped. A single relu leg
    # ``max(x-tau,0)`` is MONOTONE in x, hence MI-INVARIANT by the data-processing
    # inequality, and near-collinear with raw x -- so the greedy MI screen drops
    # it as redundant with its raw source, EXACTLY as it drops a single adaptive
    # Fourier leg (low marginal MI) and the clean missingness indicator (tied MI
    # with its raw NaN-bin twin). But the hinge's value is NOT marginal MI: it is
    # the SECOND SLOPE it hands a downstream linear / shallow model
    # (``[1, x, relu(x-tau)]`` fits a two-slope kink ``[1, x]`` cannot). The
    # generating stage already (a) detected the breakpoint, (b) HELD-OUT-validated
    # it (2-segment beats 1-segment OOS R^2 on the %3 slice), and (c) admitted the
    # leg only on its held-out INCREMENTAL linear usability over raw x -- so a
    # candidate ``_hinge_features_`` name is a confirmed univariate win. Without
    # this re-add, default-on hinge would GENERATE-then-DROP every leg (wasted
    # compute + the project's MI-vs-linear-usability rule violated, the same fix
    # the adaptive-Fourier protection block applies). TWO-PART SELF-LIMITING GATE
    # (the legs were deferred + just materialised above, so neutral data adds zero
    # cols): (1) the raw SOURCE must have survived the screen (a hinge on a never-
    # selected noise column is left out); (2) the leg must lift a HELD-OUT linear
    # fit over the ALREADY-SELECTED feature set PLUS the source + its degree-2 poly
    # ``[src, src^2]`` -- so a leg subsumed by a surviving pair composite (b/d on
    # ``y=a**2/b+log(c)*sin(d)``) or a smooth curve a quadratic already fits
    # (``y=x^2``) adds ~0 and is rejected, while a genuine slope change with no
    # competing composite clears the floor. Runs BEFORE the ``selected_vars_names``
    # remap so the re-added index routes correctly; the recipe is in
    # ``engineered_recipes`` -> transform() replays it byte-for-byte.
    _hinge_feats = getattr(self, "_hinge_features_", None)
    if _hinge_feats and len(selected_vars):
        _cols_index = {c: i for i, c in enumerate(cols)}
        _sv_set = set(selected_vars)
        _sel_names_now = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        # SELECTED-SET INCREMENTAL-R^2 GATE (the principled self-limit). A hinge
        # leg is admitted on its held-out linear usability over raw x in the FE
        # stage, but on a MULTI-SIGNAL frame the SELECTED pair composite may
        # already capture the source's structure better than a univariate kink
        # (e.g. on y=a**2/b+log(c)*sin(d) the hinge fires on b / d, but
        # div(sqr(a),abs(b)) / mul(log(c),sin(d)) subsume them). So the protection
        # re-adds a leg ONLY when it lifts a held-out linear fit over the ALREADY-
        # SELECTED feature set -- a leg whose value is subsumed by a surviving
        # composite adds ~0 and is dropped (no spurious cols on multi-signal data),
        # while a genuine slope-change leg with no competing composite clears the
        # floor (the hidden-champion win is kept). y is read only here at fit.
        _y_for_hinge_gate = None
        try:
            _yv = _y_np
            _yv = np.asarray(_yv, dtype=np.float64).reshape(-1)
            if _yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_yv)):
                _y_for_hinge_gate = _yv
        except Exception:
            _y_for_hinge_gate = None
        # Continuous values of the currently-selected columns (engineered from the
        # snapshot, raw from X) -> the baseline design the leg must beat OOS.
        _sel_value_cols = []
        if _y_for_hinge_gate is not None and isinstance(X, pd.DataFrame):
            for _sn in _sel_names_now:
                _cv = _eng_continuous_snapshot.get(_sn)
                if _cv is None and _sn in X.columns:
                    _cv = X[_sn].to_numpy()
                if _cv is None:
                    continue
                try:
                    _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue  # a raw categorical/string selected column (e.g. under skip_categorical_encoding) is not a numeric R^2-baseline regressor -- exclude it from the linear design
                if _cv.shape[0] == _y_for_hinge_gate.shape[0] and np.all(np.isfinite(_cv)):
                    _sel_value_cols.append(_cv)

        def _heldout_incr_over_selected(_leg_vals, _src_vals=None) -> float:
            """Held-out R^2 gain of adding ``_leg_vals`` to the selected design
            PLUS the source and its degree-2 poly, scored on the %3 stride slice.

            Including ``[src, src^2]`` in the baseline is the SMOOTH-CURVE guard:
            a parabola (y=x^2) is captured by ``src^2`` so a kink adds ~0 over it
            and is rejected (no spurious hinge on a smooth target -- matches the
            biz_value complementarity contract); a GENUINE slope change still beats
            ``[src, src^2]`` OOS (a quadratic cannot fit a sharp two-slope kink) so
            the hidden-champion leg is kept."""
            if _y_for_hinge_gate is None:
                return 1.0  # gate disabled -> fall back to the source-survived rule
            leg = np.asarray(_leg_vals, dtype=np.float64).reshape(-1)
            n = leg.shape[0]
            if n != _y_for_hinge_gate.shape[0] or not np.all(np.isfinite(leg)):
                return 0.0
            idx = np.arange(n); va = (idx % 3) == 0; tr = ~va
            if int(tr.sum()) < 32 or int(va.sum()) < 16:
                return 1.0
            yv = _y_for_hinge_gate[va]
            ss = float(np.sum((yv - yv.mean()) ** 2))
            if ss < 1e-24:
                return 0.0
            base = [np.ones(n)] + _sel_value_cols
            if _src_vals is not None:
                _sv = np.asarray(_src_vals, dtype=np.float64).reshape(-1)
                if _sv.shape[0] == n and np.all(np.isfinite(_sv)):
                    base = base + [_sv, _sv * _sv]
            def _r2(design_cols):
                A = np.column_stack(design_cols)
                try:
                    coef, *_ = np.linalg.lstsq(A[tr], _y_for_hinge_gate[tr], rcond=None)
                except Exception:
                    return -np.inf
                pred = A[va] @ coef
                return 1.0 - float(np.sum((yv - pred) ** 2)) / ss
            r2_base = _r2(base)
            r2_full = _r2(base + [leg])
            if not (np.isfinite(r2_base) and np.isfinite(r2_full)):
                return 0.0
            return float(r2_full - r2_base)

        _HINGE_PROTECT_MIN_INCR_R2 = 0.003
        _readd_hinge = []
        for _hn in _hinge_feats:
            _idx = _cols_index.get(_hn)
            if _idx is None or _idx in _sv_set:
                continue
            _rec_h = _hybrid_orth_pre_recipes.get(_hn)
            _src_h = tuple(getattr(_rec_h, "src_names", ()) or ())
            # Self-limit #1: source must have survived the screen (real signal).
            if not (_src_h and _src_h[0] in _sel_names_now):
                continue
            # Self-limit #2: the leg must lift a held-out linear fit OVER the
            # already-selected set + the source and its degree-2 poly (not
            # subsumed by a surviving composite, and a genuine kink not a smooth
            # curve a quadratic already fits).
            _leg_vals = _hinge_deferred_values.get(_hn)
            if _leg_vals is None and isinstance(X, pd.DataFrame) and _hn in X.columns:
                _leg_vals = X[_hn].to_numpy()
            _src_vals_gate = None
            if isinstance(X, pd.DataFrame) and _src_h and _src_h[0] in X.columns:
                _src_vals_gate = X[_src_h[0]].to_numpy()
            if _leg_vals is not None:
                if _heldout_incr_over_selected(_leg_vals, _src_vals_gate) < _HINGE_PROTECT_MIN_INCR_R2:
                    continue
            _readd_hinge.append(_idx)
            _sv_set.add(_idx)
        if _readd_hinge:
            selected_vars = list(selected_vars) + _readd_hinge
            if verbose:
                logger.info(
                    "MRMR hinge change-point protection: re-added %d held-out-"
                    "validated hinge leg(s) the MI screen dropped (MI-invariant; "
                    "value is downstream linear usability): %s",
                    len(_readd_hinge), [cols[i] for i in _readd_hinge],
                )

    # ORTH-BASIS UNIVARIATE PROTECTION (2026-06-15): re-add a single-source orthogonal-basis univariate column
    # (``a__T2`` ~ a**2, ``a__He4`` ~ a Hermite degree-4, ...) the MRMR screen dropped. Like a hinge leg, an
    # orth basis column is a DETERMINISTIC function of ONE raw source, so the greedy MI screen drops it as
    # redundant with that raw source under the data-processing inequality -- EVEN WHEN raw ``a`` carries ~0
    # linear/monotone signal about an even target (``exp(-a**2)`` / ``a**2``) and the basis column carries the
    # whole recoverable nonlinearity (|corr| ~0.85). The basis value is downstream LINEAR usability, not
    # marginal MI (the same MI-vs-linear-usability rule the hinge / adaptive-Fourier protections enforce). The
    # generating univariate-basis stage already uplift-gated each column, so a candidate is a confirmed
    # univariate win. SELF-LIMITING GATE mirrors the hinge block: (1) the raw source survived the screen (a
    # basis on a never-selected noise column is left out); (2) the basis lifts a HELD-OUT linear fit over the
    # ALREADY-SELECTED feature set (which already contains the raw source as a linear term) -- so a basis
    # subsumed by a surviving composite/raw adds ~0 and is rejected, while a genuine single-var nonlinearity
    # the screen DPI-dropped clears the floor. NO ``[src, src^2]`` smooth-curve term in the baseline (unlike
    # the hinge gate): for the basis the curve IS the win, so adding ``src^2`` would self-reject the very
    # quadratic basis we want. Reuses ``_heldout_incr_over_selected`` with ``_src_vals=None``.
    _orth_feats = getattr(self, "hybrid_orth_features_", None)
    if _orth_feats and len(selected_vars) and ("_heldout_incr_over_selected" in locals()):
        _cols_index_o = {c: i for i, c in enumerate(cols)}
        _sv_set_o = set(selected_vars)
        _sel_names_o = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
        _ORTH_PROTECT_MIN_INCR_R2 = 0.01  # wider than hinge 0.003: a genuine single-var basis lifts held-out R^2 by >>0.01 (~0.7 for exp(-a**2)); keeps noise-fit basis out
        _readd_orth = []
        for _on in _orth_feats:
            _oidx = _cols_index_o.get(_on)
            if _oidx is None or _oidx in _sv_set_o:
                continue
            _rec_o = _hybrid_orth_pre_recipes.get(_on)
            # Hinge legs (``kind="hinge_basis"``) are routed through hybrid_orth_features_ too, but they have a
            # DEDICATED protection block above that gates them against a ``[src, src^2]`` baseline (the smooth-
            # curve guard: a parabola is fit by src^2 so a kink adds ~0 and is rejected). This orth-basis block
            # deliberately OMITS that guard (for a curved basis the curve IS the win), so re-handling a hinge leg
            # here would bypass the smooth-curve guard and re-add spurious legs on y=x^2 data. Skip them -- the
            # hinge block already made the correct keep/drop decision (2026-06-16 regression fix).
            if getattr(_rec_o, "kind", None) == "hinge_basis":
                continue
            _src_o = tuple(getattr(_rec_o, "src_names", ()) or ())
            # Self-limit #1: single-source basis whose raw source survived the screen.
            if len(_src_o) != 1 or _src_o[0] not in _sel_names_o:
                continue
            _basis_vals = _eng_continuous_snapshot.get(_on)
            if _basis_vals is None and isinstance(X, pd.DataFrame) and _on in X.columns:
                _basis_vals = X[_on].to_numpy()
            if _basis_vals is None:
                continue
            # Self-limit #2: lifts a held-out linear fit over the already-selected design (raw source already
            # present there as a linear term) -- not subsumed by a surviving composite/raw.
            if _heldout_incr_over_selected(_basis_vals, None) < _ORTH_PROTECT_MIN_INCR_R2:
                continue
            _readd_orth.append(_oidx)
            _sv_set_o.add(_oidx)
        if _readd_orth:
            selected_vars = list(selected_vars) + _readd_orth
            if verbose:
                logger.info(
                    "MRMR orth-basis univariate protection: re-added %d single-source basis column(s) the "
                    "MI screen DPI-dropped (value is downstream linear usability over the raw source): %s",
                    len(_readd_orth), [cols[i] for i in _readd_orth],
                )

    # RAW-FEATURE FLOOR-DROP PROTECTION (Fix-B, 2026-06-16). The Westfall-Young maxT relevance floor is computed
    # over the FULL candidate pool; when the all-FE-on config widens that pool to hundreds of (already FE-stage-
    # gated) engineered columns, the per-shuffle MAX corrected MI inflates and the acceptance bar rises ABOVE a
    # genuine raw feature's true marginal MI -- so a real linear signal (e.g. x1 ~ y at binned-MI 0.057, ~30x
    # noise) is dropped from the screen entirely (confirmed root-cause of test_biz_value_mrmr_underselection).
    # LOWERING the floor would surface x1 but ALSO admit high-cardinality raw NOISE (a 50-level pure-noise
    # categorical whose finite-sample MI is inflated) -- a regression. Instead, KEEP the floor (noise stays
    # rejected) and re-add a raw feature the screen dropped IFF it lifts a HELD-OUT linear fit over the already-
    # selected design -- the SAME MI-vs-linear-usability protection the hinge / orth-basis blocks use. A genuine
    # linear/monotone raw signal clears the lift; a high-card noise categorical (no held-out linear usability)
    # does not, so it stays out. Conditioned on _y_for_hinge_gate (the held-out scorer); no-op when it is None.
    # Self-contained held-out scorer (the hinge block's _y_for_hinge_gate / _heldout_incr_over_selected only
    # exist when hinge legs were generated; this protection must run regardless). Baseline = intercept + the
    # continuous values of the ALREADY-SELECTED columns (engineered from the snapshot, raw from X), so a raw
    # feature SUBSUMED by a selected composite adds ~0 and is NOT re-added (no raw-redundancy regression).
    if isinstance(X, pd.DataFrame) and len(selected_vars):
        _rp_y = None
        try:
            _rp_yv = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).reshape(-1)
            if _rp_yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_rp_yv)):
                _rp_y = _rp_yv
        except Exception:
            _rp_y = None
        if _rp_y is not None:
            _RAW_PROTECT_MIN_INCR_R2 = 0.005  # genuine linear raw signal lifts held-out R^2 >> 0.005; noise ~0
            _rp_n = _rp_y.shape[0]
            _rp_idx = np.arange(_rp_n); _rp_va = (_rp_idx % 3) == 0; _rp_tr = ~_rp_va
            _rp_sel_names = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
            _rp_base = [np.ones(_rp_n)]
            for _sn in _rp_sel_names:
                _cv = _eng_continuous_snapshot.get(_sn)
                if _cv is None and _sn in X.columns:
                    _cv = X[_sn].to_numpy()
                if _cv is None:
                    continue
                try:
                    _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                except (TypeError, ValueError):
                    continue  # raw categorical/string selected column -- not a numeric R^2 regressor
                if _cv.shape[0] == _rp_n and np.all(np.isfinite(_cv)):
                    _rp_base.append(_cv)

            # Hoist the fold- and candidate-INVARIANT pieces out of the per-candidate R^2 (each call below
            # re-used the SAME held-out target, its centered SS, and the SAME base design rows): the val
            # target ``_yv`` / its SS, the train target, and the base design already sliced into train/val
            # blocks. Every call scores ``[base | one candidate column]``, so only the single candidate
            # column is stacked/sliced per call instead of rebuilding + row-slicing the full base at n rows.
            _yv = _rp_y[_rp_va]
            _rp_ss = float(np.sum((_yv - _yv.mean()) ** 2))
            _rp_y_tr = _rp_y[_rp_tr]
            _rp_base_mat = np.column_stack(_rp_base)
            _rp_base_tr = _rp_base_mat[_rp_tr]
            _rp_base_va = _rp_base_mat[_rp_va]

            def _rp_r2(_extra=None):
                """Held-out R^2 of ``[base | extra]``; ``_extra`` is a single full-length column or None.
                Numerically identical to the prior ``_rp_r2(_design)`` (same columns in the same order,
                same train/val rows, same lstsq)."""
                if _rp_ss < 1e-24:
                    return 0.0
                if _extra is None:
                    _A_tr, _A_va = _rp_base_tr, _rp_base_va
                else:
                    _A_tr = np.column_stack((_rp_base_tr, _extra[_rp_tr]))
                    _A_va = np.column_stack((_rp_base_va, _extra[_rp_va]))
                try:
                    _coef, *_ = np.linalg.lstsq(_A_tr, _rp_y_tr, rcond=None)
                except Exception:
                    return -np.inf
                return 1.0 - float(np.sum((_yv - _A_va @ _coef) ** 2)) / _rp_ss

            if int(_rp_tr.sum()) >= 32 and int(_rp_va.sum()) >= 16:
                _rp_r2_base = _rp_r2()
                _cols_index_r = {c: i for i, c in enumerate(cols)}
                _sv_set_r = set(selected_vars)
                _readd_raw = []
                # RELEVANCE GATE on the re-add. The held-out single-split R^2 increment alone is an UNCORRECTED linear-usability test: an
                # unregularised regressor overfits idiosyncratic noise on one ~n/3 val split enough to clear the loose 0.005 floor for a
                # feature the relevance screen correctly rejected as within-null (e.g. decoy = x_real**2 on y = sign(x_real): MI ~ 0.00014,
                # below the effective floor, corr -0.04, yet R^2 incr ~0.011). Require the candidate to ALSO clear the SAME marginal-MI
                # relevance floor the screen used (absolute effective floor AND the relative-to-strongest floor) so a below-null raw cannot
                # be resurrected by linear-usability alone -- this re-opened exactly the hole the screen floor closes.
                _rp_rel_floor = float(_effective_min_relevance_gain) if "_effective_min_relevance_gain" in dir() else float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
                _rp_rel_frac = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
                _rp_max_mi = max((float(_v) for _v in cached_MIs.values()), default=0.0) if isinstance(cached_MIs, dict) else 0.0
                _rp_floor = max(_rp_rel_floor, _rp_max_mi * _rp_rel_frac)
                for _rn in getattr(self, "feature_names_in_", None) or []:
                    _ridx = _cols_index_r.get(_rn)
                    if _ridx is None or _ridx in _sv_set_r or _rn not in X.columns:
                        continue
                    _rp_cand_mi = float(cached_MIs.get((_ridx,), 0.0)) if isinstance(cached_MIs, dict) else 0.0
                    if _rp_cand_mi <= _rp_floor:
                        continue  # within-null / below the screen's relevance floor -> not a genuine signal, do not resurrect
                    try:
                        _rv = np.asarray(X[_rn].to_numpy(), dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue  # non-numeric raw (categorical/string) -> not a linear-usability candidate
                    if _rv.shape[0] != _rp_n or not np.all(np.isfinite(_rv)):
                        continue
                    if _rp_r2(_rv) - _rp_r2_base < _RAW_PROTECT_MIN_INCR_R2:
                        continue
                    _readd_raw.append(_ridx)
                    _sv_set_r.add(_ridx)
                if _readd_raw:
                    selected_vars = list(selected_vars) + _readd_raw
                    if verbose:
                        logger.info(
                            "MRMR raw-feature floor-drop protection: re-added %d held-out-validated raw "
                            "feature(s) the maxT relevance floor dropped (genuine linear usability, not "
                            "high-card noise): %s",
                            len(_readd_raw), [cols[i] for i in _readd_raw],
                        )

    # CAT-FE FLOOR-DROP PROTECTION (Fix-C, 2026-06-16). The Westfall-Young maxT relevance floor (computed over
    # the FULL widened candidate pool when many FE families are on) routinely rises above the marginal binned-MI
    # of a genuine categorical-FE encoding -- a K-fold target encoding (``cat__te``), a count/frequency encoding,
    # or a cat-num residual (``price__resid_by__cat_region``) -- so the greedy screen drops it after 2 features
    # EVEN THOUGH it carries strong LINEAR usability to y (the MI-vs-linear-usability gap, a recurring mlframe
    # theme). The cat-num residual on the kitchen-sink frame has univariate corr ~0.27 / held-out R^2-incr ~0.06
    # over the selected design yet is screened out, so downstream LogReg loses ~0.6% AUC. This is the SAME class
    # of false-drop the raw-feature / orth-basis / hinge protections already correct -- but those iterate only
    # over raw ``feature_names_in_`` / single-source orth bases / hinge legs, so an engineered cat-FE column falls
    # through every one of them. Mirror the raw protection here: KEEP the floor (sub-null noise stays rejected)
    # and re-add a dropped cat-FE column IFF it lifts a HELD-OUT linear fit over the already-selected design by
    # >= the same R^2 floor. The cat-FE columns live as quantized codes in ``data[:, idx]`` (the continuous
    # snapshot is only populated by the fe_max_steps>0 path); the binned codes preserve the monotone/linear
    # signal well enough for the usability test (a genuine encoding lifts R^2 >> floor; a noise encoding ~0).
    if isinstance(X, pd.DataFrame) and len(selected_vars):
        _cf_names = []
        for _attr in ("kfold_te_features_", "count_encoding_features_", "frequency_encoding_features_", "cat_num_interaction_features_"):
            _cf_names.extend(getattr(self, _attr, None) or [])
        _cf_names = [n for n in dict.fromkeys(_cf_names)]  # dedup, preserve order
        if _cf_names:
            _cf_y = None
            try:
                _cf_yv = np.asarray(y.to_numpy() if hasattr(y, "to_numpy") else y, dtype=np.float64).reshape(-1)
                if _cf_yv.shape[0] == int(data.shape[0]) and np.all(np.isfinite(_cf_yv)):
                    _cf_y = _cf_yv
            except Exception:
                _cf_y = None
            if _cf_y is not None:
                _CF_PROTECT_MIN_INCR_R2 = 0.005  # genuine encoding lifts held-out R^2 >> 0.005; noise ~0 (same bar as raw protection)
                _cf_n = _cf_y.shape[0]
                _cf_idx = np.arange(_cf_n); _cf_va = (_cf_idx % 3) == 0; _cf_tr = ~_cf_va
                _cf_cols_index = {c: i for i, c in enumerate(cols)}
                _cf_sv_set = set(selected_vars)
                _cf_sel_names = {cols[i] for i in selected_vars if 0 <= i < len(cols)}
                # Baseline design = intercept + continuous/binned values of the ALREADY-SELECTED columns, so a
                # cat-FE column subsumed by a selected feature adds ~0 and is NOT re-added (no redundancy regression).
                _cf_base = [np.ones(_cf_n)]
                for _sn in _cf_sel_names:
                    _cv = _eng_continuous_snapshot.get(_sn)
                    if _cv is None and _sn in X.columns:
                        try:
                            _cv = X[_sn].to_numpy()
                        except Exception:
                            _cv = None
                    if _cv is None:
                        _si = _cf_cols_index.get(_sn)
                        if _si is not None:
                            _cv = data[:, _si]
                    if _cv is None:
                        continue
                    try:
                        _cv = np.asarray(_cv, dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if _cv.shape[0] == _cf_n and np.all(np.isfinite(_cv)):
                        _cf_base.append(_cv)

                def _cf_r2(_design):
                    _A = np.column_stack(_design)
                    _yv = _cf_y[_cf_va]
                    _ss = float(np.sum((_yv - _yv.mean()) ** 2))
                    if _ss < 1e-24:
                        return 0.0
                    try:
                        _coef, *_ = np.linalg.lstsq(_A[_cf_tr], _cf_y[_cf_tr], rcond=None)
                    except Exception:
                        return -np.inf
                    return 1.0 - float(np.sum((_yv - _A[_cf_va] @ _coef) ** 2)) / _ss

                if int(_cf_tr.sum()) >= 32 and int(_cf_va.sum()) >= 16:
                    _cf_r2_base = _cf_r2(_cf_base)
                    _readd_cf = []
                    for _cn in _cf_names:
                        _cidx = _cf_cols_index.get(_cn)
                        if _cidx is None or _cidx in _cf_sv_set or _cn in _cf_sel_names:
                            continue
                        try:
                            _cvv = np.asarray(data[:, _cidx], dtype=np.float64).reshape(-1)
                        except (TypeError, ValueError, IndexError):
                            continue
                        if _cvv.shape[0] != _cf_n or not np.all(np.isfinite(_cvv)):
                            continue
                        if _cf_r2(_cf_base + [_cvv]) - _cf_r2_base < _CF_PROTECT_MIN_INCR_R2:
                            continue  # no held-out linear usability over the selected design -> stays out
                        _readd_cf.append(_cidx)
                        _cf_sv_set.add(_cidx)
                    if _readd_cf:
                        selected_vars = list(selected_vars) + _readd_cf
                        if verbose:
                            logger.info(
                                "MRMR cat-FE floor-drop protection: re-added %d held-out-validated categorical-FE "
                                "encoding(s) the maxT relevance floor dropped (genuine linear usability, not "
                                "sub-null noise): %s",
                                len(_readd_cf), [cols[i] for i in _readd_cf],
                            )

    # POST-SELECTION DCD CLUSTER DISCOVERY. DCD's in-screen hook (``screen_dcd_discover_and_swap``) anchors a cluster ONLY on a column the greedy screen actually SELECTED. On a duplicate-feature
    # fixture the greedy screen selects ONE representative (a strong column or an engineered composite) and gates the redundant duplicates out as mutually-redundant, so no duplicate is ever an
    # anchor: DCD discovers 0 clusters and ``dcd_["n_pruned"]`` stays 0 even though the duplicates re-enter ``selected_vars`` via the floor-drop / retention rescues above. The cluster the screen never
    # saw is exactly the one DCD exists to own. Run a discovery pass over the FINAL selected RAW columns (anchoring on each in selection order, growing from the other selected raws by SU >= tau): the
    # duplicate cluster is found and its redundant members pruned by DCD BEFORE the raw-redundancy / monotone-twin drops below -- DCD owns exact-duplicate clusters, the redundancy drops own engineered-
    # child subsumption. When the grown cluster reaches ``dcd_cluster_size_threshold`` the same anchor->aggregate swap the screen would have evaluated is evaluated + committed here (registering the
    # ``_dcd_pc1_`` cluster_aggregate recipe into ``engineered_recipes`` so it lands in the ``_produced_recipes_`` ledger snapshotted below). Pruned duplicate members are removed from ``selected_vars``
    # (mirroring the in-screen prune); ``dcd_`` is re-published so ``n_pruned`` / ``n_swaps`` / ``cluster_anchors`` reflect the discovered cluster.
    # GATE: fire ONLY when the in-screen DCD discovered NOTHING -- no pool member pruned AND no swap committed. That is exactly the duplicate-cluster-missed case (screen selected one representative +
    # engineered children, never anchored on a duplicate). When the in-screen DCD already clustered (FE-rich sensor-mesh / financial / embedding fixtures) it owns the support-shrinkage contract; re-
    # discovering here would double-act and could GROW support (an extra aggregate the screen-time bake-off deliberately did not add), violating the "DCD must not grow support" invariant. ``cluster_anchors``
    # is NOT a usable signal: ``discover_cluster_members`` does ``setdefault(anchor, set())`` for every selected predictor, so it carries EMPTY anchor entries even when no member joined -- the real
    # "discovered a cluster" signal is a non-zero pruned-mask / a non-empty swap_log.
    if (_persisted_dcd_state is not None and len(selected_vars) >= 2
            and _persisted_dcd_state.pool_pruned_mask is not None
            and int(_persisted_dcd_state.pool_pruned_mask.sum()) == 0
            and not (getattr(_persisted_dcd_state, "swap_log", None) or [])):
        try:
            from .._dynamic_cluster_discovery import (
                discover_cluster_members as _post_dcd_discover,
                evaluate_swap_candidate as _post_dcd_eval_swap,
                commit_swap as _post_dcd_commit_swap,
                dcd_summary as _post_dcd_summary,
            )
            _dcd_st = _persisted_dcd_state
            _mask_w0 = int(_dcd_st.pool_pruned_mask.shape[0]) if _dcd_st.pool_pruned_mask is not None else 0
            _raw_name_set_dcd = set(self.feature_names_in_)
            # Selected RAW columns: stable low indices within the DCD mask width, NUMERIC only (a
            # string/categorical raw can never enter the PC1/Pearson aggregate -- it would raise
            # "could not convert string to float" in the swap's combiner -- and is not a numeric
            # duplicate cluster anyway), in selection order.
            _num_cols_dcd = set(getattr(self, "numeric_features_in_", None) or [])
            _sel_raw_dcd = [
                int(v) for v in selected_vars
                if 0 <= int(v) < _mask_w0 and cols[int(v)] in _raw_name_set_dcd
                and (not _num_cols_dcd or cols[int(v)] in _num_cols_dcd)
                and np.issubdtype(np.asarray(data[:, int(v)]).dtype, np.number)
            ]
            _newly_pruned_dcd: set = set()
            _did_swap_dcd = False
            for _anchor in list(_sel_raw_dcd):
                if _dcd_st.pool_pruned_mask[_anchor]:
                    continue  # already pruned as a member of an earlier anchor's cluster
                _pool_dcd = [c for c in _sel_raw_dcd if c != _anchor and not _dcd_st.pool_pruned_mask[c]]
                if not _pool_dcd:
                    continue
                _added = _post_dcd_discover(
                    _dcd_st, _anchor, _pool_dcd,
                    entropy_cache=None,
                    factors_data=data,
                    factors_nbins=np.asarray(nbins, dtype=np.int64),
                    selected_vars=selected_vars,
                )
                _newly_pruned_dcd |= set(int(a) for a in _added)
                # Mirror the in-screen anchor->aggregate swap: when the grown cluster reaches the size
                # threshold, evaluate + commit the PC1/mean_z aggregate swap so n_swaps / swap_log /
                # the cluster_aggregate recipe are produced exactly as the screen would have.
                _members = _dcd_st.cluster_anchors.get(int(_anchor), set())
                if len(_members) >= int(_dcd_st.cluster_size_threshold):
                    # Sync the state's matrix to the LIVE (post-FE) matrix so the swap's S\{anchor}
                    # conditioning set -- which may reference engineered columns appended AFTER the
                    # screen built the state's matrix -- indexes valid columns (else conditional_mi
                    # raises "negative dimensions" on an out-of-range column).
                    if int(data.shape[1]) >= int(_dcd_st.factors_data.shape[1]):
                        _dcd_st.factors_data = data
                        _dcd_st.factors_nbins = np.asarray(nbins, dtype=np.int64)
                        _dcd_st.cols = list(cols)
                        if _dcd_st.pool_pruned_mask is not None and int(data.shape[1]) > int(_dcd_st.pool_pruned_mask.shape[0]):
                            _dcd_st.pool_pruned_mask = np.concatenate([
                                _dcd_st.pool_pruned_mask,
                                np.zeros(int(data.shape[1]) - int(_dcd_st.pool_pruned_mask.shape[0]), dtype=bool),
                            ])
                    # Swap conditioning set: the anchor + the OTHER selected RAW non-cluster columns
                    # only. The in-screen swap evaluates early when ``selected_vars`` is still small;
                    # post-selection the full ``selected_vars`` also holds engineered children of the
                    # SAME cluster latent (e.g. ``add(log(strong),prewarp(dup_c))``), and conditioning
                    # the aggregate-vs-anchor relevance comparison on those children removes the shared
                    # latent entirely -> both sides read ~0 residual -> the swap never fires. Restricting
                    # the conditioning set to selected raws outside the cluster restores the screen-time
                    # comparison (aggregate's denoised latent vs the single noisy anchor dup).
                    _cluster_idx_set = set(_members) | {int(_anchor)}
                    _swap_sel_vars = [
                        int(v)
                        for v in selected_vars
                        if int(v) == int(_anchor) or (int(v) not in _cluster_idx_set and 0 <= int(v) < _mask_w0 and cols[int(v)] in _raw_name_set_dcd)
                    ]
                    _dec = _post_dcd_eval_swap(
                        _dcd_st, int(_anchor), _swap_sel_vars,
                        target_y=target_indices,
                        factors_data=data,
                        factors_nbins=np.asarray(nbins, dtype=np.int64),
                        entropy_cache=None,
                        cached_MIs=None,
                        full_npermutations=int(getattr(self, "full_npermutations", 0) or 0),
                    )
                    if getattr(_dec, "accept", False):
                        _dref: dict = {}
                        _post_dcd_commit_swap(
                            _dcd_st, int(_anchor), _dec,
                            selected_vars=selected_vars,
                            data_ref=_dref,
                            engineered_recipes=engineered_recipes,
                            predictors_log=None,
                        )
                        data = _dref.get("data", data)
                        nbins = _dref.get("nbins", nbins)
                        cols = _dref.get("cols", cols)
                        _did_swap_dcd = True
            if _newly_pruned_dcd or _did_swap_dcd:
                selected_vars = [v for v in selected_vars if int(v) not in _newly_pruned_dcd]
                self.dcd_ = _post_dcd_summary(_dcd_st)
                if isinstance(self.dcd_, dict):
                    self.cluster_members_ = dict(self.dcd_.get("cluster_anchors_names", {}))
                if verbose:
                    logger.info(
                        "MRMR post-selection DCD: discovered a duplicate cluster the greedy screen never anchored on; " "pruned %d redundant member(s)%s.",
                        len(_newly_pruned_dcd),
                        " + committed an aggregate swap" if _did_swap_dcd else "",
                    )
        except Exception as _exc_post_dcd:
            logger.warning("MRMR post-selection DCD discovery failed: %s; keeping support as-is.", _exc_post_dcd)

    # PRODUCED-RECIPES AUDIT LEDGER: ``engineered_recipes`` at this point holds EVERY recipe the FE stages produced this fit, before the greedy CMI screen / accuracy gate / cross-stage dedup drop the
    # weaker candidates. ``self._engineered_recipes_`` (built just below) carries only the survivors -- it is intersected with support_ so the user-facing rosters stay a subset of get_feature_names_out()
    # (pinned by layer28). The audit / pickle-replay paths, however, need to recover WHICH mechanism produced each engineered column even when the screen dropped it, so snapshot the full produced set here
    # as a separate read-only ledger. fe_provenance_ reads this to emit one row per produced engineered column (survivors get their real greedy gain/rank, screened-out ones get NaN gain / rank -1).
    self._produced_recipes_ = list(engineered_recipes.values())

    # PSEUDO-CHILD MASKED-RAW RESCUE (2026-06-13). The default-ON conditional-gate / binned-
    # numeric-agg / row-argmax FE families append THRESHOLD/BINNING re-mixes of a raw operand
    # (``gate_mask__a__b`` / ``binagg_skew(c|qbin(a))`` / ``argmax__a__b``) into the screening pool
    # BEFORE the greedy screen. A re-mix of ``a`` can marginally OUT-SCORE raw ``a`` and is selected
    # first; raw ``a``'s conditional relevance given that re-mix then collapses (the re-mix is a lossy
    # function of ``a`` -- the data-processing-inequality trap), so the greedy screen drops ``a``
    # EVEN WHEN ``a`` carries a dominant private LINEAR term (``y += 10*a``) the re-mix only partially
    # tracks. Re-add such a masked raw -- one consumed by a selected pseudo-remix child but itself
    # dropped -- IFF it retains >= RAW_SELF_RETAIN_FRAC of its marginal debiased excess under the
    # keep-rule conditioned ONLY on its GENUINE (non-pseudo) selected children (an ``a**2/b`` ratio /
    # composite -- the real potential subsumers, with the masking pseudo re-mixes EXCLUDED from the
    # conditioning). A private LINEAR term keeps ~50% -> RESCUE; a fully-subsumed operand keeps ~0.6%
    # -> NOT rescued (so a raw genuinely subsumed by an elementary child is never resurrected). The
    # downstream raw-redundancy DROP sweep still runs after with the SAME pseudo-exclusion, so the two
    # passes agree. Byte-identical when no pseudo-remix child is selected (the candidate set is empty).
    # Off when the drop sweep is disabled (shares the ``fe_drop_redundant_raw_operands`` toggle).
    if getattr(self, "fe_drop_redundant_raw_operands", True) and len(selected_vars) >= 1:
        try:
            from .._fe_raw_redundancy_drop import (
                _is_pseudo_remix_child as _pcr_is_pseudo,
                _PSEUDO_SRC_SPLIT as _pcr_split,
                raw_retains_signal_given_genuine_children as _pcr_keep,
            )
            from .._mi_greedy_cmi_fe import _quantile_bin as _pcr_qbin
            _pcr_raw_set = set(self.feature_names_in_)
            _pcr_sel_set = set(selected_vars)
            _pcr_sel_names = {cols[i] for i in selected_vars}
            # Selected pseudo-remix children and the raw operands each re-mixes.
            _pcr_pseudo_sel = [i for i in selected_vars if _pcr_is_pseudo(cols[i])]
            if _pcr_pseudo_sel:
                # raw_name -> selected pseudo children consuming it.
                _pcr_consumed: dict = {}
                for _pi in _pcr_pseudo_sel:
                    _toks = {t for t in _pcr_split.split(cols[_pi]) if t}
                    for _t in _toks:
                        if _t in _pcr_raw_set:
                            _pcr_consumed.setdefault(_t, []).append(_pi)
                # A raw is also consumed by a GENUINE (non-pseudo) selected engineered child when its
                # name token appears there; such a raw is left to the DROP sweep (might be subsumed).
                _pcr_genuine_eng = [i for i in selected_vars if (cols[i] not in _pcr_raw_set) and not _pcr_is_pseudo(cols[i])]
                _pcr_y = np.ascontiguousarray(np.asarray(classes_y)).ravel().astype(np.int64)
                try:
                    _pcr_yv = y.values if hasattr(y, "values") else np.asarray(y)
                    _pcr_yv = np.asarray(_pcr_yv).reshape(-1)
                    if (_pcr_yv.shape[0] == int(data.shape[0]) and np.issubdtype(_pcr_yv.dtype, np.number)
                            and int(np.unique(_pcr_yv).size) > max(20, 2 * int(np.unique(_pcr_y).size))):
                        _pcr_nb = int(min(max(10, int(np.unique(_pcr_y).size)), max(2, int(data.shape[0]) // 50)))
                        _pcr_y = np.ascontiguousarray(_pcr_qbin(_pcr_yv.astype(np.float64), nbins=_pcr_nb)).astype(np.int64)
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in _fit_impl_core.py:7249: %s", e)
                    pass
                _pcr_eng_cont = _eng_continuous_snapshot
                from .._fe_raw_redundancy_drop import _TOKEN_SPLIT as _pcr_gtok
                # PERMUTATION-SIGNIFICANCE GATE on the masked-raw rescue (2026-06-19, I4 noise
                # admission). The keep-rule conditions on the GENUINE children only; when a raw's
                # ONLY consumer is a pseudo binagg/gate/argmax re-mix the conditioning set is empty
                # and the keep-rule returns True by construction (it cannot prove subsumption). A
                # PURE-NOISE raw (``e``, not in y, consumed only by ``binagg_std(e|qbin(a))``) thus
                # sails through and is re-added -- the I4 noise true-negative violation. Gate the
                # rescue on the SAME within-data marginal permutation-significance test the
                # raw-retention re-add uses: a raw must sit ABOVE its own permutation null against y
                # to be a genuine masked signal. ``e`` sits WITHIN its null (p>=alpha) -> NOT rescued;
                # a genuinely masked raw (``a`` carrying ``3*a``) clears it. Best-effort: a kernel
                # failure falls through to the permissive rescue (never drop on an estimator error).
                try:
                    from ..permutation import mi_direct as _pcr_mi_direct
                except Exception:
                    _pcr_mi_direct = None
                _pcr_signif_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
                _pcr_q_dtype = getattr(self, "quantization_dtype", np.int32)

                def _pcr_raw_is_significant(_idx):
                    if _pcr_mi_direct is None:
                        return True
                    try:
                        _sig = _pcr_mi_direct(
                            data, x=np.array([int(_idx)], dtype=np.int64), y=target_indices,
                            factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                            return_null_mean=True, parallelism="none", dtype=_pcr_q_dtype, prefer_gpu=False,
                        )
                        return float(_sig[3]) < _pcr_signif_alpha
                    except Exception:
                        return True
                _pcr_readd = []
                for _rn, _pchildren in _pcr_consumed.items():
                    if _rn in _pcr_sel_names:
                        continue  # already selected -> nothing to rescue
                    try:
                        _ridx = cols.index(_rn)
                    except ValueError:
                        continue
                    if _ridx in _pcr_sel_set:
                        continue
                    # KEEP-RULE conditioned ONLY on the raw's GENUINE (non-pseudo) selected children --
                    # the real potential subsumers (an ``a**2/b`` ratio/composite). The masking pseudo
                    # re-mixes are EXCLUDED from the conditioning so they cannot DPI-collapse the residual.
                    # A raw carrying a private term the genuine children do not span keeps a large residual
                    # (~50%) -> RESCUE; a raw fully subsumed by a genuine ratio child keeps ~0.6% -> NOT
                    # rescued (and would be dropped by the sweep anyway). When NO genuine child consumes the
                    # raw the conditioning set is empty and the keep-rule returns True (the drop was a pure
                    # pseudo-mask) -> RESCUE.
                    _child_bins = []
                    for _gi in _pcr_genuine_eng:
                        if _rn in {t for t in _pcr_gtok.split(cols[_gi]) if t}:
                            _cont = _pcr_eng_cont.get(cols[_gi])
                            if _cont is not None and np.asarray(_cont).shape[0] == int(data.shape[0]):
                                _child_bins.append(_pcr_qbin(np.asarray(_cont, dtype=np.float64), nbins=10))
                            else:
                                _child_bins.append(np.asarray(data[:, _gi]).astype(np.int64).ravel())
                    _rb = np.asarray(data[:, _ridx]).astype(np.int64).ravel()
                    if _pcr_keep(
                        raw_bin=_rb,
                        y_bin=_pcr_y,
                        genuine_child_bins=_child_bins,
                        allow_linear_usability=bool(getattr(self, "use_simple_mode", False)),
                        seed=int(getattr(self, "random_seed", 0) or 0),
                    ) and _pcr_raw_is_significant(_ridx):
                        _pcr_readd.append(_ridx)
                if _pcr_readd:
                    selected_vars = list(selected_vars) + [i for i in _pcr_readd if i not in _pcr_sel_set]
                    if verbose:
                        logger.info(
                            "MRMR pseudo-child masked-raw rescue: re-added %d raw operand(s) the greedy "
                            "screen dropped because a gate/binagg/argmax re-mix of them was selected first "
                            "and masked their conditional relevance (DPI trap), yet they retain a private "
                            "residual given that re-mix: %s",
                            len(_pcr_readd), [cols[i] for i in _pcr_readd],
                        )
        except Exception as _exc_pcr:
            logger.warning("MRMR pseudo-child masked-raw rescue failed: %s; keeping support as-is.", _exc_pcr)

    # RAW-VS-ENGINEERED CONDITIONAL-REDUNDANCY DROP (2026-06-08): the greedy MRMR order
    # selects a raw operand on its high MARGINAL relevance BEFORE the engineered child built
    # from it is in support, so the redundancy penalty never fires against it, and the
    # retention / augmentation passes above then re-add it. The result is a subsumed operand
    # admitted alongside the engineered feature that fully determines y from it (e.g. raw
    # ``a, b`` beside ``div(neg(a),sqrt(b))`` for ``y=(a**2)/b``, since
    # ``(a/sqrt(b))**2 = a**2/b``). This final sweep removes such operands using the SAME
    # debiased excess-CMI idea the engineered-vs-engineered S5 gate validated, so the verdict
    # is n-INVARIANT (identical at n=1000 and n=50000) and never drops a raw carrying genuine
    # independent signal (a private additive term keeps a large excess and is KEPT). On by
    # default; ``fe_drop_redundant_raw_operands=False`` restores the pre-fix behaviour.
    if getattr(self, "fe_drop_redundant_raw_operands", True) and getattr(self, "redundancy_policy", "emit_both") == "drop" and len(selected_vars) >= 2:
        try:
            from .._fe_raw_redundancy_drop import drop_redundant_raw_operands
            _raw_names_for_redund = set(self.feature_names_in_)
            # Only worth running when at least one engineered survivor and one raw operand
            # are both selected (otherwise the helper short-circuits anyway).
            _sel_names_redund = [cols[i] for i in selected_vars]
            _has_eng = any(nm not in _raw_names_for_redund for nm in _sel_names_redund)
            _has_raw = any(nm in _raw_names_for_redund for nm in _sel_names_redund)
            if _has_eng and _has_raw:
                # Continuous target for equi-frequency re-binning: the screening
                # ``classes_y`` is frequently HEAVILY imbalanced on a skewed regression
                # target (``y=(a**2)/b`` puts ~89% of rows in one bin), which crushes the
                # engineered anchor's MI and inflates a subsumed operand's apparent residual
                # fraction. Re-binning the continuous target equi-frequency restores a faithful
                # anchor. Falls back to ``classes_y`` for already-discrete targets.
                _y_cont_for_redund = None
                try:
                    _yv = y.values if hasattr(y, "values") else np.asarray(y)
                    _yv = np.asarray(_yv).reshape(-1)
                    if _yv.shape[0] == int(data.shape[0]) and np.issubdtype(np.asarray(_yv).dtype, np.number):
                        _y_cont_for_redund = _yv
                except Exception:
                    _y_cont_for_redund = None
                # Only engineered survivors with a replayable recipe (1-deep, in
                # ``engineered_recipes``) survive into transform output; a nested-
                # engineered child is dropped there. A raw must not be judged
                # redundant against a child that will not exist at predict time
                # (that empties the support -- see the guard in
                # drop_redundant_raw_operands), so anchor the verdict only on the
                # replayable survivors.
                _replayable_eng_names = set(engineered_recipes.keys())
                # NESTED-OPERAND CONSUMER DETECTION (BUG1, 2026-06-12): pass the
                # engineered RECIPES (name -> EngineeredRecipe) and the raw frame so
                # the redundancy verdict can walk each consuming composite's operand
                # tree, isolate the cleanest raw-containing sub-expression (e.g.
                # ``div(sqr(a),abs(b))`` = a**2/b inside a fused full-target composite),
                # and condition the raw on THAT clean sub-expression rather than the
                # fused whole -- so a fully-subsumed operand drops even when it is
                # selected alongside the composite (not only when the composite
                # collapsed the whole selection into the never-empty path).
                _kept_redund, _dropped_redund_names = drop_redundant_raw_operands(
                    data=data,
                    cols=cols,
                    selected_cols_idx=selected_vars,
                    raw_name_set=_raw_names_for_redund,
                    y_binned=classes_y,
                    y_continuous=_y_cont_for_redund,
                    engineered_continuous=_eng_continuous_snapshot,
                    replayable_eng_names=_replayable_eng_names,
                    recipes=engineered_recipes,
                    raw_X=X,
                    retain_frac=float(getattr(self, "fe_raw_redundancy_retain_frac", 0.15) or 0.15),
                    linear_usability_keep=bool(getattr(self, "use_simple_mode", False)),
                    tail_subsume_enable=bool(getattr(self, "fe_pair_usability_admission_enable", True)),
                    tail_subsume_min_corr=float(getattr(self, "fe_raw_tail_subsume_min_corr", 0.85)),
                    tail_subsume_rank_frac=float(getattr(self, "fe_pair_usability_admission_rank_frac", 0.7)),
                    seed=int(getattr(self, "random_seed", 0) or 0),
                    verbose=verbose,
                )
                if _dropped_redund_names:
                    selected_vars = _kept_redund
                    # Record the verdict so the downstream raw-signal-retention augmentation
                    # (which re-attaches a raw whose NAME tokenises a confirmed recipe by
                    # marginal MI) does NOT resurrect an operand this n-invariant conditional-
                    # redundancy sweep just dropped. The verdict is authoritative at every n.
                    self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | set(_dropped_redund_names)
                    # If the drop left NO raw survivor while engineered children survived, the
                    # engineered-only support is the INTENDED, complete outcome (every raw operand
                    # was conditionally subsumed). Flag it so the empty-RAW rescue ``else`` branch
                    # below does NOT mistake this for a "screen returned 0 raw" emergency and
                    # re-pollute the support with the dropped operands (or, worse, a pure-noise
                    # column ranked next by marginal MI).
                    _remaining_raw_after_drop = [v for v in selected_vars if cols[v] in set(self.feature_names_in_)]
                    if not _remaining_raw_after_drop:
                        # NEVER-EMPTY RAW FLOOR (2026-06-27 subsumption-aware fix). The drop is allowed to
                        # remove a raw subsumed by a surviving engineered child WHILE other raws remain (the
                        # I4b contract). When dropping the redundant raws would leave ZERO raw survivors, the
                        # PRIOR floor unconditionally re-added the single STRONGEST dropped raw by MARGINAL MI
                        # as a raw stand-in. That marginal-MI pick is exactly the trap the rest of this module
                        # warns about: a fully-subsumed DOMINANT operand (``a`` in ``a**2/b``, whose ratio is
                        # captured byte-for-byte by the surviving fused compound) has the LARGEST marginal MI
                        # yet ZERO conditional/private residual, so the floor resurrected the very operand the
                        # n-invariant CMI sweep had just correctly dropped (the scaled_1_5 / heavy_tailed F2
                        # failure: the compound ``add(div(sqr(a),b),mul(log(c),sin(d)))`` fully reconstructs y,
                        # the sweep dropped a/b/d, and this floor re-added raw ``a`` beside the compound that
                        # subsumes it). When the main sweep empties the raw support EVERY dropped raw was judged
                        # fully subsumed by a surviving multi-source child, so the engineered survivor(s) ARE the
                        # complete feature set -- the SAME engineered-only outcome the ``uniform`` profile and the
                        # never-empty re-attach block's all-operands-subsumed ``elif`` reach. Defer to that:
                        # re-add a dropped raw ONLY if it still carries a SIGNIFICANT PRIVATE LINEAR residual the
                        # engineered survivors do not linearly reproduce (a genuine partial-signal raw a downstream
                        # linear model needs); otherwise flag the intended engineered-only support so the
                        # downstream empty-raw rescue does not re-pollute it. The linear-usability re-add is a
                        # SIMPLE-mode concept ONLY: in full FE mode a subsumed MONOTONE operand (``a`` in ``a**2/b``
                        # on a positive domain, whose rank tracks ``y`` so a partial-rank-correlation reads a
                        # SPURIOUS private residual) is statistically indistinguishable from a genuine linear term
                        # and MUST still drop (I4b) -- this mirrors ``drop_redundant_raw_operands``'s own
                        # ``allow_linear_usability=False`` policy in full FE mode (see its docstring / keep-leg).
                        # So the floor re-adds a dropped raw ONLY in simple mode and ONLY when it clears the same
                        # permutation-floored partial-rank-correlation probe; in full FE mode the empty raw support
                        # is the intended engineered-only outcome (the ``uniform`` profile's result).
                        _floor_simple = bool(getattr(self, "use_simple_mode", False))
                        from .._fe_raw_redundancy_drop import raw_retains_linear_signal_given_children as _floor_lin
                        _best_floor_idx, _best_floor_rel = None, float("-inf")
                        _tgt_floor = np.asarray(target_indices, dtype=np.int64)
                        _fn_floor = np.asarray(nbins, dtype=np.int64)
                        # Continuous engineered survivor values (for the linear-usability child design).
                        _floor_child_vals = []
                        for _ei in _kept_redund:
                            _enm = cols[_ei]
                            if _enm in set(self.feature_names_in_):
                                continue
                            _cv = (_eng_continuous_snapshot or {}).get(_enm)
                            if _cv is not None and np.asarray(_cv).shape[0] == int(data.shape[0]):
                                _floor_child_vals.append(np.asarray(_cv, dtype=np.float64).ravel())
                            else:
                                _floor_child_vals.append(np.asarray(data[:, _ei], dtype=np.float64).ravel())
                        try:
                            _yv_floor = y.values if hasattr(y, "values") else np.asarray(y)
                            _yv_floor = np.asarray(_yv_floor, dtype=np.float64).reshape(-1)
                        except Exception:
                            _yv_floor = np.asarray(classes_y, dtype=np.float64).reshape(-1)
                        for _dn in _dropped_redund_names:
                            try:
                                _ci = cols.index(_dn)
                            except ValueError:
                                continue
                            # Only a dropped raw with genuine PRIVATE linear signal beyond the engineered
                            # survivors is eligible to be the raw representative; a fully-subsumed operand is not.
                            # Full FE mode: no operand is eligible (defer to engineered-only -- the I4b contract).
                            _eligible_floor = _floor_simple
                            if _floor_simple and _floor_child_vals:
                                try:
                                    _rawv = None
                                    if isinstance(X, pd.DataFrame) and _dn in X.columns:
                                        _rawv = np.asarray(X[_dn], dtype=np.float64).ravel()
                                    if _rawv is None:
                                        _rawv = np.asarray(data[:, _ci], dtype=np.float64).ravel()
                                    _eligible_floor = bool(_floor_lin(
                                        _rawv, _yv_floor, _floor_child_vals,
                                        seed=int(getattr(self, "random_seed", 0) or 0),
                                    ))
                                except Exception:
                                    _eligible_floor = False
                            if not _eligible_floor:
                                continue
                            try:
                                from ..info_theory import mi as _floor_mi
                                _rel = float(_floor_mi(data, np.array([int(_ci)], dtype=np.int64), _tgt_floor, _fn_floor))
                            except Exception:
                                _rel = 0.0
                            if _rel > _best_floor_rel:
                                _best_floor_rel, _best_floor_idx = _rel, int(_ci)
                        if _best_floor_idx is not None:
                            selected_vars = list(_kept_redund) + [_best_floor_idx]
                            _kept_name_floor = cols[_best_floor_idx]
                            # The re-kept raw is no longer "dropped": remove it from the verdict set so the
                            # downstream retention / rescue passes treat it as a genuine survivor.
                            self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) - {_kept_name_floor}
                            if verbose:
                                logger.info(
                                    "MRMR raw-redundancy never-empty floor: dropping all raw operands would "
                                    "empty support_; retained strongest LINEAR-USABLE raw %r (marginal MI %.4f) "
                                    "as the raw representative beside the surviving engineered child.",
                                    _kept_name_floor, _best_floor_rel,
                                )
                        else:
                            # Every dropped raw is fully subsumed by a surviving engineered child -- the
                            # engineered recipe(s) ARE the complete feature set (the uniform-profile outcome).
                            self._redundancy_emptied_raw_ = True
                    if verbose:
                        logger.info(
                            "MRMR raw-redundancy drop: removed %d raw operand(s) conditionally "
                            "redundant given their surviving engineered child (debiased excess "
                            "CMI below the relative bar): %s",
                            len(_dropped_redund_names), _dropped_redund_names,
                        )
        except Exception as _exc_redund:
            logger.warning(
                "MRMR raw-redundancy drop failed: %s; keeping the un-pruned support.",
                _exc_redund,
            )

    # RAW-vs-RAW MONOTONE-TWIN DROP (2026-06-16, F6). The cross-stage Spearman-0.99 dedup
    # (above, ~line 5343) collapses monotone-equivalent ENGINEERED columns, and the
    # raw-vs-engineered redundancy sweep (above) drops a raw subsumed by an engineered
    # CHILD. Neither catches a RAW DECOY that is a pure MONOTONE re-encoding of ANOTHER
    # selected RAW column (``a_exp = exp(a)`` when raw ``a`` is selected): both bin
    # byte-identically under the quantile / rank-invariant MI screen, so they carry the
    # SAME information about y, yet the greedy screen / floor-drop protection / retention
    # passes can admit BOTH (the redundancy penalty is computed on coarse bins and the
    # nonlinear twin slips a small residual past it). Mirror the engineered dedup at the
    # RAW level: among selected raw columns, when two are monotone twins (|Spearman rho|
    # >= the same 0.99 bar), drop the LOWER-relevance one (by screening marginal MI;
    # ties keep the earlier-selected). A genuine independent raw (rank-uncorrelated with
    # every other selected raw) is untouched, so this never over-drops. Byte-identical
    # when no two selected raws are monotone twins. Shares the
    # ``fe_drop_redundant_raw_operands`` toggle (off restores the prior behaviour).
    if getattr(self, "fe_drop_redundant_raw_operands", True) and isinstance(X, pd.DataFrame) and len(selected_vars) >= 2:
        try:
            _MONO_TWIN_RHO = 0.99
            _raw_set_mt = set(self.feature_names_in_)
            _raw_sel_mt = [v for v in selected_vars if cols[v] in _raw_set_mt and cols[v] in X.columns]
            if len(_raw_sel_mt) >= 2:
                _mt_n = int(data.shape[0])
                _mt_ranks: dict[int, np.ndarray] = {}
                for _v in _raw_sel_mt:
                    try:
                        _cv = np.asarray(X[cols[_v]].to_numpy(), dtype=np.float64).reshape(-1)
                    except (TypeError, ValueError):
                        continue
                    if _cv.shape[0] == _mt_n and np.all(np.isfinite(_cv)) and _cv.std() > 1e-12:
                        _mt_ranks[_v] = pd.Series(_cv).rank(method="average").to_numpy()
                # Relevance to break ties / pick the survivor: the screening marginal MI.
                def _mt_relevance(_v):
                    try:
                        return float(cached_MIs.get((_v,), 0.0))
                    except Exception:
                        return 0.0
                _mt_keep: list[int] = []
                _mt_drop: set[int] = set()
                # Keep order = selection order, so an earlier-selected twin is preferred on a tie.
                for _v in _raw_sel_mt:
                    if _v not in _mt_ranks:
                        _mt_keep.append(_v)
                        continue
                    _twin_of = None
                    for _k in _mt_keep:
                        _rk = _mt_ranks.get(_k)
                        if _rk is None:
                            continue
                        _rho = float(np.corrcoef(_mt_ranks[_v], _rk)[0, 1])
                        if np.isfinite(_rho) and abs(_rho) >= _MONO_TWIN_RHO:
                            _twin_of = _k
                            break
                    if _twin_of is None:
                        _mt_keep.append(_v)
                    else:
                        # Drop the LOWER-relevance twin; if the candidate out-scores the kept twin,
                        # displace the kept one instead.
                        if _mt_relevance(_v) > _mt_relevance(_twin_of) + 1e-12:
                            _mt_drop.add(_twin_of)
                            _mt_keep.remove(_twin_of)
                            _mt_keep.append(_v)
                        else:
                            _mt_drop.add(_v)
                if _mt_drop:
                    selected_vars = [v for v in selected_vars if v not in _mt_drop]
                    if verbose:
                        logger.info(
                            "MRMR raw monotone-twin drop: removed %d raw decoy(s) that are pure "
                            "monotone re-encodings of a higher-relevance selected raw (|Spearman rho|"
                            ">=%.2f, rank-redundant): %s",
                            len(_mt_drop), _MONO_TWIN_RHO, [cols[v] for v in _mt_drop],
                        )
        except Exception as _exc_mt:
            logger.warning(
                "MRMR raw monotone-twin drop failed: %s; keeping the un-pruned support.",
                _exc_mt,
            )

    # ---------------------------------------------------------------------------------------------------------------
    # selected_vars: cols-indices -> names -> original-frame indices (categorize_dataset may rearrange cat columns).
    # ---------------------------------------------------------------------------------------------------------------

    selected_vars_names = np.array(cols)[np.array(selected_vars, dtype=np.intp)]
    # BUG2 (2026-06-12): the cross-fold stability vote in ``_run_fe_step`` pops a
    # fold-unstable engineered recipe AND de-selects its column for that step, but the
    # materialised bin-code column stays in ``cols``/``data``, so the downstream greedy
    # screen (step>1 re-screen / final selection) re-admits it on marginal MI -- it then
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
    for col in selected_vars_names:
        if col in self.feature_names_in_:
            original_indices.append(self.feature_names_in_.index(col))
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

    # PSEUDO-REMIX OPERAND RE-ADD (2026-06-17). A surviving conditional-gate / binned-numeric-agg /
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
        for _pname in self._passthrough_features_:
            if _pname in self.feature_names_in_:
                _pidx = self.feature_names_in_.index(_pname)
                if _pidx not in _existing:
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
            # 2026-05-30 Wave 9.1 fix (loop iter 17): order-preserving set
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
            # the very redundancy the aggregation removed -- only features dropped for low marginal/joint relevance get reconsidered.
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
            # carries no independent signal, but CatBoost RFECV -- which scores raw
            # MARGINAL usefulness, blind to the engineered child's coverage -- would re-admit
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
                for feature in new_features:
                    selected_vars.append(self.feature_names_in_.index(feature))
            else:
                if verbose:
                    logger.info("RFECV selected no additional features.")

    # ---------------------------------------------------------------------------------------------------------------
    # Assign support
    # ---------------------------------------------------------------------------------------------------------------

    # ``selected_vars`` holds integer column indices. Force int64 dtype so an
    # EMPTY selection (all signal folded into engineered recipes under the
    # full-mode default -> zero raw survivors) stays an integer index array.
    # ``np.array([])`` defaults to float64, and the ndarray transform path
    # (``X[:, support_]`` in _mrmr_validate_transform) then raises
    # ``IndexError: arrays used as indices must be of integer (or boolean)
    # type`` because a float array can't index. Integer dtype makes the empty
    # slice a valid no-op on both the DataFrame and the ndarray paths.
    #
    # NEVER-EMPTY RAW REPRESENTATIVE (2026-06-12): when the ONLY confirmed feature(s) are engineered
    # recipes (their raw operands all judged redundant, so ``selected_vars`` is empty while
    # ``_engineered_recipes_`` is non-empty), the raw integer ``support_`` would be empty even though a
    # genuine signal-bearing feature WAS selected. That breaks any linear downstream that consumes the raw
    # ``support_`` (it sees zero columns) and the never-empty selection contract. Re-attach the single
    # highest-marginal-MI raw OPERAND of a surviving engineered feature as the cluster's raw stand-in --
    # mirrors ``reattach_raw_representative_after_aggregate_swap`` for the DCD aggregate case. One raw
    # column is added (the operand most relevant to y), never an unvalidated one, and the engineered
    # recipe still rides along via ``get_feature_names_out`` / ``transform``. Best-effort: any failure
    # leaves the empty support_ unchanged (no crash on a degenerate fit).
    # The conditional-redundancy sweep marks an INTENTIONALLY engineered-only support via
    # ``_redundancy_emptied_raw_``: every raw operand was FULLY subsumed by a surviving
    # engineered child (e.g. ``a`` + ``b`` both captured by ``div(neg(a),sqrt(b))`` for
    # ``y=a**2/b``), so re-attaching the "best" raw stand-in would resurrect exactly the
    # operand the n-invariant CMI verdict just dropped (observed at n=2000/5000: the sweep
    # dropped a+b, this block re-added a as the highest-marginal stand-in). When the empty
    # raw support is the redundancy sweep's deliberate outcome, the engineered recipes ARE
    # the complete feature set -- skip the never-empty re-attach and let ``support_`` stay
    # empty (transform still emits the engineered columns). The re-attach remains active for
    # the genuine degenerate case (engineered-only with no redundancy verdict).
    if (not selected_vars) and getattr(self, "_engineered_recipes_", None) and not getattr(self, "_redundancy_emptied_raw_", False):
        try:
            from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _NE_TOK_SPLIT
            from ..info_theory import mi as _ne_mi
            _raw_names_ne = set(self.feature_names_in_)
            # Recipe -> column NAME. ``self._engineered_recipes_`` holds EngineeredRecipe
            # OBJECTS whose ``str()``/``repr()`` is the full dataclass repr, NOT the column
            # name -- so ``str(r)`` neither matches ``cols`` nor is a clean token source.
            # Resolve the name from ``.name`` (the column the recipe materialises), falling
            # back to ``str(r)`` only for a legacy bare-string entry.
            def _ne_recipe_name(_r):
                _nm = getattr(_r, "name", None)
                return str(_nm) if _nm is not None else str(_r)
            _operand_idxs: set = set()
            for _r_obj in self._engineered_recipes_:
                for _tok in _NE_TOK_SPLIT.split(_ne_recipe_name(_r_obj)):
                    if not _tok:
                        continue
                    _base = _tok if _tok in _raw_names_ne else (_tok.split("__", 1)[0] if "__" in _tok else None)
                    if _base in _raw_names_ne:
                        try:
                            _operand_idxs.add(cols.index(_base))
                        except ValueError:
                            continue
            # CONDITIONAL-REDUNDANCY GUARD on the re-attach (2026-06-12, BUG1). The
            # operand picked below is the highest-MARGINAL-MI one, but a high marginal
            # does NOT mean it carries signal the engineered child lacks: a dominant
            # operand (``a`` in ``a**2/b``) has the largest marginal yet is FULLY
            # subsumed by the ``a**2/b`` ratio inside the surviving composite. Re-
            # attaching it re-introduces exactly the redundancy the campaign set out to
            # remove (observed at n=100k on the user fixture: the single full-target
            # composite ``add(mul(log(c),sin(d)),abs(div(sqr(a),abs(b))))`` rode as a
            # recipe -> ``selected_vars`` empty -> this block re-attached raw ``a``,
            # which the composite already captures). Restrict the candidate pool to
            # operands that carry a SIGNIFICANT INDEPENDENT RESIDUAL given the engineered
            # survivor(s), using the SAME n-invariant conditional-redundancy verdict as
            # the main drop. Only operands NOT judged subsumed are eligible; if every
            # operand is subsumed (the composite fully reconstructs y), leave the support
            # engineered-only -- the recipe IS the complete feature set.
            _subsumed_operand_names: set = set()
            try:
                # emit_both keeps engineered operands; skip the subsumption restriction so the never-empty re-attach is not narrowed.
                if getattr(self, "redundancy_policy", "emit_both") != "drop":
                    raise RuntimeError("redundancy_policy=emit_both: skip subsumption restriction")
                from .._fe_raw_redundancy_drop import drop_redundant_raw_operands as _ne_drop
                _recipe_names = [_ne_recipe_name(r) for r in self._engineered_recipes_]
                _eng_survivor_cols = [cols.index(_nm) for _nm in _recipe_names if _nm in cols and _nm not in _raw_names_ne]
                if _eng_survivor_cols and _operand_idxs:
                    _trial_sel = sorted(set(_operand_idxs) | set(_eng_survivor_cols))
                    # name -> EngineeredRecipe so the verdict can isolate clean nested
                    # sub-expressions here too (BUG1 nested-operand consumer detection).
                    _ne_recipes = {_ne_recipe_name(_r): _r for _r in self._engineered_recipes_ if _ne_recipe_name(_r) is not None}
                    _, _ne_dropped = _ne_drop(
                        data=data, cols=cols, selected_cols_idx=_trial_sel,
                        raw_name_set=_raw_names_ne, y_binned=classes_y,
                        y_continuous=(y.values if hasattr(y, "values") else np.asarray(y)),
                        engineered_continuous=_eng_continuous_snapshot,
                        replayable_eng_names=set(_recipe_names),
                        recipes=_ne_recipes,
                        raw_X=X,
                        linear_usability_keep=bool(getattr(self, "use_simple_mode", False)),
                        seed=int(getattr(self, "random_seed", 0) or 0), verbose=0,
                    )
                    _subsumed_operand_names = set(_ne_dropped or ())
            except Exception:
                _subsumed_operand_names = set()  # best-effort: fall back to MI-only pick
            # C2 ADDITIVE-FUSION EXCLUSION (2026-06-24): never re-attach a raw operand the
            # FE additive-fusion proposer already judged subsumed by the fused ``add(...)``
            # compound (``_raw_redundancy_dropped_``). The fused compound carries its additive
            # term, so resurrecting it as the never-empty stand-in re-injects the redundant
            # single-group fragment the fusion removed (the FUSION-blocked goal's leftover raw).
            _fused_dropped_ne = set(getattr(self, "_raw_redundancy_dropped_", None) or set())
            _eligible_idxs = [_oi for _oi in _operand_idxs if cols[_oi] not in _subsumed_operand_names and cols[_oi] not in _fused_dropped_ne]
            if _eligible_idxs:
                _tgt_ne = np.asarray(target_indices, dtype=np.int64)
                _fn_ne = np.asarray(nbins, dtype=np.int64)
                _best_idx_ne, _best_rel_ne = -1, float("-inf")
                for _oi in sorted(_eligible_idxs):
                    try:
                        _rel_ne = float(_ne_mi(data, np.array([int(_oi)], dtype=np.int64), _tgt_ne, _fn_ne))
                    except Exception:
                        _rel_ne = 0.0
                    if _rel_ne > _best_rel_ne:
                        _best_rel_ne, _best_idx_ne = _rel_ne, int(_oi)
                if _best_idx_ne >= 0:
                    # ``_best_idx_ne`` is a COLS-space index (the augmented, categorize_dataset-reordered matrix that carries the injected target +
                    # engineered columns). ``support_`` must index ``feature_names_in_`` (raw user columns only), so remap the chosen operand by NAME --
                    # the same translation the main selection does at the ``selected_vars_names`` split. Assigning the raw cols-space index directly let an
                    # out-of-range index (>= n_features_in_) reach ``support_`` and crashed ``transform`` with IndexError when feature_names_in_ was narrower.
                    _operand_name_ne = cols[_best_idx_ne]
                    selected_vars = [self.feature_names_in_.index(_operand_name_ne)]
                    if verbose:
                        logger.info(
                            "MRMR never-empty raw representative: support_ would be empty (only engineered "
                            "feature(s) selected); re-attached raw operand %r (marginal MI %.4f) as the raw "
                            "stand-in (carries residual signal beyond the engineered child).",
                            _operand_name_ne, _best_rel_ne,
                        )
            elif _operand_idxs and _subsumed_operand_names:
                # EVERY engineered operand is conditionally subsumed by a surviving
                # engineered child -- the engineered recipe(s) ARE the complete feature
                # set. Record the verdict so the DOWNSTREAM empty-raw rescue (the
                # ``else`` branch that tops up the support to ``min_features_fallback``
                # by marginal MI) does NOT resurrect a dropped operand. Without this the
                # rescue re-adds the highest-marginal operand (``a`` in the user's
                # ``a**2/b + log(c)sin(d)`` fixture, whose ``a**2/b`` is captured by the
                # composite), the BUG1 spurious-raw-kept regression -- because the raw
                # operands were dropped by the EARLIER raw-retention pass, not the main
                # ``drop_redundant_raw_operands`` sweep, so neither
                # ``_raw_redundancy_dropped_`` nor ``_redundancy_emptied_raw_`` was set.
                # The ``elif`` at the rescue site keys on ``_redundancy_emptied_raw_`` and
                # the rescue / RFECV / augmentation pools all exclude
                # ``_raw_redundancy_dropped_``; populate both here so the engineered-only
                # support stands.
                self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | set(_subsumed_operand_names)
                self._redundancy_emptied_raw_ = True
                if verbose:
                    logger.info(
                        "MRMR never-empty raw representative: ALL %d engineered operand(s) are "
                        "conditionally subsumed by the surviving engineered child; leaving support "
                        "engineered-only (no spurious raw stand-in re-attached): %s",
                        len(_operand_idxs), sorted(_subsumed_operand_names),
                    )
        except Exception as _ne_exc:
            logger.warning("MRMR never-empty raw representative re-attach failed (%r); leaving support_ empty.", _ne_exc)

    # CLUSTER-AGGREGATE 'replace' FINAL EXCLUSION (2026-06-16). Members folded into a denoised
    # MULTI-parent aggregate (``cluster_aggregate_mode='replace'`` -> ``_cluster_aggregate_removals_``,
    # or a DCD PC1/mean_z swap -> ``cluster_members_``) were dropped from ``selected_vars`` at the
    # replace step, but the many intervening raw-retention / masked-raw rescue / hinge / orth / pcr /
    # never-empty-representative / additional-RFECV passes can resurrect a removed member when it is an
    # OPERAND of a SURVIVING engineered child (e.g. ``add(refl0,sin(indep))`` keeps a private residual
    # given the aggregate). Several of those passes pre-date the cluster-aggregate feature and do not
    # consult ``_cluster_aggregate_removals_``, so rather than patch each call site we re-apply the
    # exclusion ONCE here -- the single chokepoint right before ``support_`` is frozen, in
    # feature_names_in_ index space -- guaranteeing a replaced member can never reach support_ /
    # get_feature_names_out regardless of which re-add path touched it. The denoised aggregate itself
    # (an engineered name in ``_engineered_recipes_``) is untouched.
    _ca_final_excl = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
    _cm_final = getattr(self, "cluster_members_", None)
    _raw_names_cmfinal = set(self.feature_names_in_)
    # Raw cluster representatives a DCD-aggregate-anchor swap would otherwise strip: force-kept / force-ADDED
    # below so every collapsed cluster retains >=1 raw column. Initialised at function-body level (NOT inside
    # the ``isinstance(_cm_final, dict)`` block) because it is referenced unconditionally further down -- a
    # fit whose ``cluster_members_`` is not a dict (e.g. dcd_enable=False) must not hit an UnboundLocalError.
    _ca_keep_raw: set = set()
    if isinstance(_cm_final, dict):
        # ``cluster_members_`` is populated by mechanisms with DIFFERENT final-exclusion semantics:
        #   * an ENGINEERED-anchor cluster (DCD PC1/mean_z swap whose anchor is the denoised aggregate, a
        #     name NOT in feature_names_in_): the aggregate survives, so its raw members are stripped from
        #     any raw support a downstream pass resurrected.
        #   * a pure RAW redundancy cluster (exact-duplicate / collinear / DCD decoy pair, ALL names raw):
        #     exactly ONE representative must survive. The cluster dict's anchor/member DIRECTION is NOT
        #     reliable for which to keep (e.g. ``{'collinear_b': ['good_b']}`` labels the genuine ``good_b``
        #     as a member), so keep the highest cached-MI(.,y) column of the cluster and strip the rest.
        #     This de-duplicates (RC2 exact-duplicate / realistic-mixed-degenerate -> keep ``good_a``,
        #     ``good_b``) AND prunes genuine decoys (layer6 DCD second-decoy -> keep the strong driver).
        # Mixed clusters (raw anchor + pseudo-remix/engineered member) fall to the pseudo-remix-protected
        # member strip below.
        _nm2col_cm = {c: i for i, c in enumerate(cols)}
        # Use the in-scope LOCAL cached_MIs (populated by the screen, same dict used at the other read
        # sites) -- self.cached_MIs is only assigned near the end of _fit_impl, so on a FRESH fit
        # hasattr(self,...) is False and this degraded to {} -> every rep tiebreak collapsed to 0.0.
        _cached_cm = cached_MIs if ("cached_MIs" in dir() and isinstance(cached_MIs, dict)) else {}
        _name2inidx_cm = {c: i for i, c in enumerate(self.feature_names_in_)}
        # Names ALREADY in selected_vars (raw, in feature_names_in_ index space). The greedy
        # screen / retention passes have already chosen these as the cluster's surviving
        # representative(s); the pure-raw-cluster strip below must KEEP one of them rather than
        # silently swapping in an unselected member.
        _sel_names_cm = {self.feature_names_in_[int(v)] for v in selected_vars if int(v) < len(self.feature_names_in_)}
        def _cm_mi(_nm):
            _ci = _nm2col_cm.get(_nm)
            return float(_cached_cm.get((_ci,), 0.0)) if _ci is not None else 0.0

        for _anchor, _members in _cm_final.items():
            _a = str(_anchor)
            _mlist = [str(_m) for _m in (_members or [])] if isinstance(_members, (list, tuple, set)) else []
            _group = [_a] + _mlist
            if all(_nm in _raw_names_cmfinal for _nm in _group):
                # pure raw cluster -- keep the single strongest representative, strip the rest.
                # KEEP-ONE-SELECTED-RAW (2026-06-18): the cached-MI lookup the rep tiebreak relies
                # on is often a miss for these members (``cached_MIs`` is keyed on the screening
                # cols-space and a cluster member may never have been scored there), collapsing every
                # member's relevance to 0.0 -> the rep degenerates to the LOWEST feature-index member.
                # When that lowest-index member is NOT the one the greedy screen actually selected,
                # the cluster's genuine selected representative (which IS in ``selected_vars``) gets
                # stripped and the whole latent block vanishes from support_ (embedding cross-terms
                # layer20: 12-member e1 cluster, only the high-MI anchor ``e1_17`` was selected, yet
                # the rep collapsed to ``e1_1`` and e1 dropped entirely). PRINCIPLE: a member already
                # chosen by the screen is the de-facto representative -- prefer it. Restrict the rep
                # candidate pool to the cluster members present in ``selected_vars`` when any are;
                # only fall back to the MI/index tiebreak over the whole group when none was selected.
                if len(_group) >= 2:
                    _rep_pool = [_nm for _nm in _group if _nm in _sel_names_cm] or _group
                    _rep = min(_rep_pool, key=lambda _nm: (-_cm_mi(_nm), _name2inidx_cm.get(_nm, 1 << 30)))
                    _ca_final_excl.update(_nm for _nm in _group if _nm != _rep)
                    # KEEP-ONE-RAW for pure-raw PRUNED clusters (no denoised aggregate): when a
                    # within-pack SU cluster is merely pool-pruned (size below the swap threshold, so
                    # no aggregate column is ever built) AND its screen-selected representative was
                    # later dropped (e.g. a second screen pass re-prunes the pack and the anchor falls
                    # out of selected_vars), NONE of the group survives -- the latent vanishes from
                    # support_ entirely and the RFECV rescue pool excludes every cluster member, so it
                    # is unrecoverable (scenario-A sensor mesh: L1 pack pruned, AUC -0.08). Force-keep
                    # the chosen representative exactly like the engineered-anchor branch below, so every
                    # collapsed cluster retains >=1 raw column. No support growth: this re-adds the SINGLE
                    # representative of a cluster that would otherwise contribute zero columns.
                    if _rep not in _sel_names_cm:
                        _ca_keep_raw.add(_rep)
            elif _a not in _raw_names_cmfinal:
                # engineered/aggregate anchor (DCD PC1/mean_z swap) -- strip its (raw) members; the
                # aggregate itself survives.
                #
                # KEEP-ONE-RAW-REPRESENTATIVE (2026-06-18): a DCD denoised-aggregate swap collapses an
                # entire raw cluster into a single engineered column and prunes every raw member. When
                # the aggregate is the cluster's ONLY survivor, the latent block has no RAW column in
                # ``support_`` at all -- any downstream consumer that reads the raw support names (a
                # linear model fed the raw matrix, a feature-importance report, the layer20 embedding
                # cross-terms contract) sees the whole block as dropped even though it was merely
                # denoised. PRINCIPLE: the engineered aggregate is a SUPPLEMENT, not a replacement for
                # the cluster's presence -- always leave at least one genuine raw representative of the
                # cluster alive. Keep the strongest raw member (highest cached MI, lowest-index
                # tiebreak) and strip the rest; the kept member is force-added to ``selected_vars``
                # below so it survives even if no raw member reached the support chokepoint.
                _raw_mem = [_m for _m in _mlist if _m in _raw_names_cmfinal]
                if _raw_mem:
                    _agg_rep = min(_raw_mem, key=lambda _nm: (-_cm_mi(_nm), _name2inidx_cm.get(_nm, 1 << 30)))
                    _ca_keep_raw.add(_agg_rep)
                    _ca_final_excl.update(_m for _m in _mlist if _m != _agg_rep)
                else:
                    _ca_final_excl.update(_mlist)
            else:
                # raw anchor + engineered/pseudo member(s) -- strip only the non-raw members (pseudo-remix
                # protection below keeps a raw operand the cluster pairs with a pseudo-remix built from it).
                _ca_final_excl.update(_m for _m in _mlist if _m not in _raw_names_cmfinal)
    # PSEUDO-REMIX SELF-SOURCE PROTECTION (2026-06-17). A conditional-gate / binned-numeric-agg /
    # row-argmax anchor (``gate_mask__a__b`` / ``binagg_mean(d|qbin(a))`` / ``argmax__a__b``) is a
    # LOSSY threshold/binning RE-MIX of its raw source(s): it cannot carry a raw operand's private
    # LINEAR term (a binary gate of ``a`` does not span ``10*a``). When the clustering folds a RAW
    # column into a cluster ANCHORED by such a pseudo-remix BUILT FROM that raw and strips the raw as
    # a "member", a genuine private term is lost (test_private_raw_a_kept: raw ``a`` with a dominant
    # ``10*a`` term clustered under ``gate_mask__a__b`` and dropped). Mirror the redundancy gate's
    # ``_is_pseudo_remix_child`` exclusion here: never strip a RAW column that the cluster pairs with
    # a pseudo-remix BUILT FROM that raw, in EITHER direction --
    #   (A) pseudo-remix ANCHOR + raw-source MEMBER  (``gate_mask__a__b`` anchors raw ``a``); or
    #   (B) raw ANCHOR + pseudo-remix MEMBER of it    (``x2`` anchors ``gate_mask__x2__x1``).
    # The lossy gate/binagg/argmax cannot carry the raw's continuous value a LINEAR downstream needs
    # (measured: a 5-class LogReg macro-F1 0.62 when x2 was stripped as such a cluster anchor vs >0.70
    # protected; and the test_private_raw_a_kept ``10*a`` case for direction A). Engineered members +
    # genuine (non-pseudo) aggregate members are untouched -> byte-identical when no such pairing exists.
    if _ca_final_excl and isinstance(_cm_final, dict):
        from .._fe_raw_redundancy_drop import _is_pseudo_remix_child, _PSEUDO_SRC_SPLIT
        _raw_names_ca = set(self.feature_names_in_)
        _protect_ca = set()
        for _anchor, _members in _cm_final.items():
            _a = str(_anchor)
            _mlist = [str(_m) for _m in (_members or [])]
            # (A) pseudo-remix anchor -> protect any raw member that is one of its sources.
            if _is_pseudo_remix_child(_a):
                _anchor_raw_srcs = {t for t in _PSEUDO_SRC_SPLIT.split(_a) if t in _raw_names_ca}
                for _m in _mlist:
                    if _m in _raw_names_ca and _m in _anchor_raw_srcs:
                        _protect_ca.add(_m)
            # (B) raw anchor -> protect it when a member is a pseudo-remix built from that raw.
            if _a in _raw_names_ca:
                for _m in _mlist:
                    if _is_pseudo_remix_child(_m) and _a in set(_PSEUDO_SRC_SPLIT.split(_m)):
                        _protect_ca.add(_a)
                        break
        if _protect_ca:
            _ca_final_excl -= _protect_ca
    # KEEP-ONE-RAW-REPRESENTATIVE force-keep: the designated raw representative of each
    # DCD-aggregate-collapsed cluster must never be stripped, even if another cluster's strip set or
    # a redundancy pass nominated it. Remove it from the exclusion set first.
    if _ca_keep_raw:
        _ca_final_excl -= _ca_keep_raw
    if _ca_final_excl and selected_vars:
        _fni = self.feature_names_in_
        _pre_n = len(selected_vars)
        selected_vars = [v for v in selected_vars if _fni[v] not in _ca_final_excl]
        if verbose and len(selected_vars) != _pre_n:
            logger.info(
                "MRMR cluster-aggregate 'replace': re-stripped %d cluster member(s) a downstream "
                "retention/rescue pass had resurrected; only the denoised aggregate survives.",
                _pre_n - len(selected_vars),
            )
    # Force-ADD the kept raw representative of each aggregate-collapsed cluster when no raw member of
    # that cluster reached the support chokepoint (the swap pruned them all). Guarantees every
    # denoised cluster keeps >=1 genuine raw column in ``support_`` alongside its engineered aggregate.
    if _ca_keep_raw:
        _name2inidx_add = {c: i for i, c in enumerate(self.feature_names_in_)}
        _sel_set = set(int(v) for v in selected_vars)
        for _kr in _ca_keep_raw:
            _ki = _name2inidx_add.get(_kr)
            if _ki is not None and _ki not in _sel_set:
                selected_vars.append(_ki)
                _sel_set.add(_ki)

    # SEARCH-SPACE RESTRICTION FINAL ENFORCEMENT (2026-06-16). When the caller pins the candidate
    # pool via ``factors_names_to_use`` / ``factors_to_use``, the SCREEN honours it, but the many
    # post-screen raw-retention / masked-raw rescue / hinge / orth / pcr / never-empty / count-floor
    # re-add passes do NOT all consult the restriction, so a forbidden raw column (e.g. ``good2`` when
    # the pool is pinned to ``["good1"]``) leaks into ``support_`` -- and because the in-object fit-skip
    # / _FIT_CACHE replay a stale selection unless every param change invalidates it, the bug also shows
    # as a stale-replay regression. Enforce the restriction ONCE at the support chokepoint (raw indices
    # into feature_names_in_): a raw column the user excluded can never reach support_ regardless of which
    # re-add path admitted it. Engineered survivors (in ``_engineered_recipes_``) are untouched -- they are
    # built only from allowed raws by the screen, which already respects the restriction.
    _allowed_raw_idx = None
    _fn_restrict = getattr(self, "factors_names_to_use", None)
    _fi_restrict = getattr(self, "factors_to_use", None)
    if _fn_restrict:
        _allowed_names = set(_fn_restrict)
        _allowed_raw_idx = {_j for _j, _nm in enumerate(self.feature_names_in_) if _nm in _allowed_names}
    elif _fi_restrict is not None:
        _allowed_raw_idx = set(int(_j) for _j in _fi_restrict)
    if _allowed_raw_idx is not None and selected_vars:
        _pre_r = len(selected_vars)
        selected_vars = [v for v in selected_vars if int(v) in _allowed_raw_idx]
        if verbose and len(selected_vars) != _pre_r:
            logger.info(
                "MRMR: dropped %d raw feature(s) outside the pinned factors_names_to_use / "
                "factors_to_use search space that a downstream re-add pass had admitted.",
                _pre_r - len(selected_vars),
            )

    # P>=N FP-CONTROL TOTAL CAP. In the p>>n regime some pure-noise column WILL correlate with y by chance, and the post-screen
    # retention / rescue passes can admit a few of them (measured: 51 raws at p=150, 103 at p=300 -- 1-3 over the multiple-comparison
    # ceiling). When p >= n, cap the total selected raw set at ``max(20, p//3)`` features chosen by descending relevance MI(X_j, y),
    # mirroring the RFECV ``p_ge_n_fp_control_cap``. Confined to p >= n so the well-powered p<n path is byte-unchanged. Engineered
    # survivors are counted toward the cap (they reach the output too) but never the dropped tail -- only raw ``selected_vars`` is trimmed.
    _pgn_n = int(data.shape[0]) if "data" in dir() else 0
    _pgn_p = int(getattr(self, "n_features_in_", 0) or 0)
    if _pgn_p > 0 and _pgn_n > 0 and _pgn_p >= _pgn_n and selected_vars:
        _pgn_ceiling = max(20, _pgn_p // 3)
        # ``n_engineered_out`` is not yet bound at this point (assigned further below near ``n_features_``); read the
        # engineered count straight off ``self._engineered_recipes_`` (populated by the main sweep above) so engineered
        # survivors are actually charged against the p>=n ceiling instead of silently degrading the count to 0.
        _pgn_eng = len(getattr(self, "_engineered_recipes_", None) or [])
        _pgn_budget = _pgn_raw_budget(_pgn_ceiling, _pgn_eng)
        if len(selected_vars) > _pgn_budget:
            # LOCAL cached_MIs (see the cluster-rep note above): self.cached_MIs is unset until the end of
            # _fit_impl, so on a fresh fit this read degraded to {} and the p>=n cap sort collapsed to index order.
            _pgn_cached = cached_MIs if ("cached_MIs" in dir() and isinstance(cached_MIs, dict)) else {}
            _pgn_n2ci = {c: i for i, c in enumerate(cols)} if "cols" in dir() else {}
            _fni_pgn = self.feature_names_in_

            def _pgn_rel(_v):
                _nm = _fni_pgn[_v] if _v < len(_fni_pgn) else None
                _ci = _pgn_n2ci.get(_nm)
                return float(_pgn_cached.get((_ci,), 0.0)) if _ci is not None else 0.0
            # Descending relevance, stable secondary key on the raw index so ties are column-order invariant.
            selected_vars = [v for v in sorted(selected_vars, key=lambda v: (-_pgn_rel(v), int(v)))][:_pgn_budget]
            if verbose:
                logger.info(
                    "MRMR p>=n FP-control: capped raw support to top-%d by relevance (p=%d >= n=%d, ceiling=%d, engineered=%d).",
                    _pgn_budget, _pgn_p, _pgn_n, _pgn_ceiling, _pgn_eng,
                )

    # EMIT-BOTH OPERAND RE-ATTACH. A feature selector must not destroy linearly-usable raw signal: for every SELECTED engineered feature, surface its raw operand
    # columns (parsed from the recipe ``src_names`` or name tokens). Re-attach only operands that themselves carry MARGINAL signal toward y (a within-data
    # permutation-significance test, p<alpha): a SIGNAL operand of a selected engineered feature is kept (the linear-usability win), but a NOISE operand fused into a
    # composite (e.g. ``noise_3`` inside ``sub(...,prewarp(noise_3))``) does NOT clear its null and is NOT re-attached -> FS still rejects noise. Bounded to operands
    # of SELECTED engineered features, in feature_names_in_, not already selected, inside the pinned search space (``_allowed_raw_idx``).
    if getattr(self, "redundancy_policy", "emit_both") != "drop" and selected_vars:
        try:
            from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _EB_TOK_SPLIT
            from ..permutation import mi_direct as _eb_mi_direct
            _eb_raw_names = set(self.feature_names_in_)
            _eb_sel_set = set(int(v) for v in selected_vars)
            _eb_name_to_in = {nm: i for i, nm in enumerate(self.feature_names_in_)}
            _eb_cols_idx = {nm: i for i, nm in enumerate(cols)}
            _eb_recipes = {getattr(_r, "name", None): _r for _r in (getattr(self, "_engineered_recipes_", None) or [])}
            _eb_alpha = float(os.environ.get("MLFRAME_MRMR_NULL_SIGNIF_ALPHA", "0.05"))
            _eb_qdtype = getattr(self, "quantization_dtype", np.int32)
            _eb_operands: list[str] = []
            for _enm, _erec in _eb_recipes.items():
                if _enm is None or _enm in _eb_raw_names or _enm not in cols:
                    continue
                if cols.index(_enm) not in _eb_sel_set:
                    continue  # only SELECTED engineered features
                _src = getattr(_erec, "src_names", None)
                _toks = list(_src) if _src else [t for t in _EB_TOK_SPLIT.split(str(_enm)) if t]
                for _t in _toks:
                    _base = _t if _t in _eb_raw_names else (_t.split("__", 1)[0] if "__" in _t else None)
                    if _base in _eb_raw_names and _base not in _eb_operands:
                        _eb_operands.append(_base)

            def _eb_operand_is_signal(_cols_i):
                try:
                    _r = _eb_mi_direct(data, x=np.array([int(_cols_i)], dtype=np.int64), y=target_indices,
                                       factors_nbins=nbins, npermutations=32, min_nonzero_confidence=0.0,
                                       return_null_mean=True, parallelism="none", dtype=_eb_qdtype, prefer_gpu=False)
                    return float(_r[3]) < _eb_alpha  # p-value below alpha -> genuine marginal signal
                except Exception:
                    return True  # estimator error -> do not silently drop a possibly-genuine operand
            _eb_added = []
            for _op in _eb_operands:
                _idx = _eb_name_to_in.get(_op)
                if _idx is None or int(_idx) in _eb_sel_set:
                    continue
                if _allowed_raw_idx is not None and int(_idx) not in _allowed_raw_idx:
                    continue
                _ci = _eb_cols_idx.get(_op)
                if _ci is None or not _eb_operand_is_signal(_ci):
                    continue  # noise operand of a composite -> FS keeps rejecting it
                selected_vars.append(int(_idx))
                _eb_sel_set.add(int(_idx))
                _eb_added.append(_op)
            if _eb_added and verbose:
                logger.info("MRMR emit_both operand re-attach: added %d signal raw operand(s) of selected engineered features: %s", len(_eb_added), _eb_added)
        except Exception as _eb_exc:
            if verbose:
                logger.info("MRMR emit_both operand re-attach skipped (%s: %s).", type(_eb_exc).__name__, _eb_exc)

    # C2 ADDITIVE-FUSION FINAL RAW STRIP (2026-06-24). Raw operands the FE additive-fusion
    # proposer verified the fused ``add(...)`` compound fully captures (``_fused_subsumed_raws_``,
    # set via the production keep-probe against the WHOLE compound) must not survive in the raw
    # support, no matter which downstream retention / rescue / re-attach pass re-added them: those
    # passes condition a raw on the CLEAN nested sub-expression, which on a corrupted a/b half does
    # NOT capture the raw and so KEEPS it, whereas the fused compound DOES -- this strip applies the
    # stronger whole-compound verdict. Only strips when the fused compound itself survives as a
    # recipe (so the additive term it carries is actually present); byte-identical (empty set) when
    # no fusion fired.
    _fused_subsumed = set(getattr(self, "_fused_subsumed_raws_", None) or set())
    if _fused_subsumed and getattr(self, "_engineered_recipes_", None):
        _surv_eng = {getattr(_r, "name", None) for _r in (self._engineered_recipes_ or [])}
        # Only strip a raw when a SURVIVING engineered compound actually references it (carries its
        # additive term) -- otherwise leave it (the fusion that subsumed it did not survive).
        import re as _re_fsr
        _fsr_tok = _re_fsr.compile(r"[^A-Za-z0-9_]+")
        _covered: set = set()
        for _en in _surv_eng:
            for _t in _fsr_tok.split(str(_en) or ""):
                if not _t:
                    continue
                _base = _t if _t in set(self.feature_names_in_) else (
                    _t.split("__", 1)[0] if "__" in _t and _t.split("__", 1)[0] in set(self.feature_names_in_) else None)
                if _base is not None:
                    _covered.add(_base)
        _strip = _fused_subsumed & _covered
        if _strip:
            selected_vars = [v for v in selected_vars if not (0 <= int(v) < len(self.feature_names_in_) and self.feature_names_in_[int(v)] in _strip)]
            if verbose:
                logger.info(
                    "MRMR C2 additive-fusion: stripped %d raw operand(s) the fused compound fully "
                    "captures from the final raw support: %s", len(_strip), sorted(_strip),
                )

    self.support_ = np.array(selected_vars, dtype=np.int64)

    # USABILITY-AWARE MULTI-LIST POST-PASS (2026-06-13). ``support_`` above is the pure-MI selection
    # (the nonlinear / tree list, byte-identical to today). When ``usability_aware_lists`` is on AND
    # a continuous target is available, ALSO produce a linear-downstream list (``support_linear_``)
    # and a blended universal list (``support_universal_``) -- each a replayable candidate list --
    # WITHOUT touching ``support_``. Fully guarded: a degenerate pool / non-numeric target / row
    # mismatch leaves the extra lists ``None`` and never breaks the fit. ``support_nonlinear_`` is
    # always set as the alias of ``support_`` so downstream routing has a stable name to read.
    try:
        if getattr(self, "usability_aware_lists", False):
            from .._usability_lists import build_usability_lists
            build_usability_lists(self, X=X, y_cont=getattr(self, "_fe_prewarp_y_continuous_", None))
        else:
            self.support_nonlinear_ = self.support_
            self.support_linear_ = None
            self.support_universal_ = None
    except Exception as _usability_exc:  # never let the optional second list break a fit
        self.support_nonlinear_ = getattr(self, "support_", None)
        self.support_linear_ = None
        self.support_universal_ = None
        if verbose:
            logger.info("Usability-aware multi-list post-pass skipped (%s: %s).", type(_usability_exc).__name__, _usability_exc)

    # SELECTION-STABILITY REPLAY STATE (backlog W3, 2026-06-11). Store a compact slice of the
    # already-discretised screening matrix ``data`` + the target codes + the per-column selection
    # outcome so ``MRMR.selection_stability_report(n_boot=K)`` can recompute per-feature selection-
    # frequency by REPLAY (K cheap marginal-MI sweeps over the frozen bins) without refitting MRMR --
    # the #15 "replay not refit" trick applied to a user-facing confidence readout. Subsample rows to
    # cap the stored footprint (8GB-shared box); the bins are frozen so resampling rows is leak-free.
    try:
        _build_stability_replay_state(
            self, data=data, cols=cols, nbins=nbins,
            target_indices=target_indices, selected_vars=selected_vars,
            engineered_recipes=engineered_recipes,
        )
    except Exception as _stab_exc:  # never let the diagnostic accessor break a fit
        self._stability_replay_state_ = None
        if verbose:
            logger.info("Stability replay-state capture skipped (%s: %s).", type(_stab_exc).__name__, _stab_exc)

    # ROSTER RECONCILIATION (2026-06-04): the per-stage engineered rosters (``hybrid_orth_features_``, ``_adaptive_fourier_features_``, the Layer-33/34/37/38/87+ family lists) are
    # populated as each FE stage APPENDS its columns, but the MRMR screen / accuracy gate / dedup then drop a subset before support is finalised. ``self._engineered_features_`` is the
    # authoritative set of engineered columns that actually survived into the output (reachable via ``get_feature_names_out``). Intersect every roster with it so a column the screen
    # dropped (and the adaptive-protection block did NOT re-add) no longer leaks into the user-facing roster. Runs AFTER the additional_rfecv rescue (which reads the FULL rosters to
    # exclude engineered columns from its raw-only rescue pool); the rescue never adds engineered columns, so ``_engineered_features_`` is final here. Order-preserving per roster.
    _surviving_eng = set(self._engineered_features_ or [])
    for _roster_attr in (
        "hybrid_orth_features_", "_adaptive_fourier_features_", "mi_greedy_features_",
        "kfold_te_features_", "count_encoding_features_", "frequency_encoding_features_",
        "cat_num_interaction_features_", "missingness_indicator_features_",
        "missingness_count_features_", "missingness_pattern_features_",
        "pairwise_ratio_features_", "pairwise_log_ratio_features_",
        "grouped_delta_features_", "lagged_diff_features_", "grouped_agg_features_",
        "composite_group_agg_features_", "grouped_quantile_features_",
        "cat_pair_features_", "cat_triple_features_", "numeric_decompose_features_",
        "modular_features_", "group_distance_features_", "rare_category_features_",
        "conditional_residual_features_", "conditional_dispersion_features_",
        "wavelet_features_",
        "rankgauss_features_", "temporal_agg_features_",
    ):
        _roster = getattr(self, _roster_attr, None)
        if _roster:
            setattr(self, _roster_attr, [c for c in _roster if c in _surviving_eng])

    # Always store ``cached_MIs`` -- the empty-support fallback at the bottom
    # of this function reads ``self.cached_MIs`` to rank by raw MI(X_j, y), so
    # the attribute should exist regardless of ``retain_artifacts``. Cheap (a
    # dict of tuple->float; bounded by the screen's candidate pool).
    self.cached_MIs = cached_MIs

    # iter66: artifact retention for cross-selector reuse (off by default).
    # Captured at the cols-space stage so ``data`` / ``cols`` / ``nbins`` are
    # the active matrices the screen actually consumed; the export dict is
    # axis-aligned to the original ``feature_names_in_`` for the downstream
    # consumer's convenience.
    if getattr(self, "retain_artifacts", False):
        try:
            from .._mrmr_artifacts import compute_mrmr_artifacts
            self._artifacts_ = compute_mrmr_artifacts(
                data=data,
                cols=list(cols),
                nbins=nbins,
                target_indices=target_indices,
                cached_MIs=cached_MIs,
                feature_names_in=list(self.feature_names_in_),
                support_original=self.support_,
                retain_bins=bool(getattr(self, "retain_bins", True)),
                dtype=self.quantization_dtype,
            )
        except Exception as _exc:
            logger.warning(
                "MRMR.retain_artifacts: capture failed (%s); export_artifacts() will raise. " "Cause: %s",
                type(_exc).__name__,
                _exc,
            )
            self._artifacts_ = None
    # 2026-05-30 Wave 9.1 fix (loop iter 30): populate ``mrmr_gains_``
    # so the documented ``uaed_auto_size=True`` post-fit elbow trim at
    # line 1020+ actually fires. Pre-fix the comment claimed
    # "Wave-7 audit landed this trace" but no code ever assigned the
    # attribute - ``getattr(self, "mrmr_gains_", [])`` defaulted to
    # empty, ``gains.size >= 3`` was False, the UAED block was
    # guaranteed dead code. ``MRMR(uaed_auto_size=True)`` silently
    # returned the full screen output regardless. Restore the
    # advertised behaviour: store per-selection-round gains in
    # screening order, aligned with the predictor log.
    try:
        self.mrmr_gains_ = np.asarray(
            [float(p.get("gain", 0.0)) for p in (predictors or [])],
            dtype=np.float64,
        )
    except Exception:
        self.mrmr_gains_ = np.array([], dtype=np.float64)
    # Layer 54: stash the greedy predictor log on ``self`` so the FE
    # provenance helper can map engineered feature names back to their
    # support_rank / mrmr_gain entries. ``predictors`` is a list of
    # ``{"name", "indices", "gain", ...}`` dicts in selection order.
    # Light copy (per-entry shallow) to dodge accidental downstream
    # mutation of the screen's working list; ``indices`` is captured as
    # a tuple to keep the entry pickle-safe across processes.
    try:
        self._predictors_log_ = tuple(
            {
                "name": p.get("name"),
                "gain": float(p.get("gain", 0.0)),
                "indices": tuple(p.get("indices", ()) or ()),
            }
            for p in (predictors or [])
        )
    except Exception:
        self._predictors_log_ = ()
    self.fallback_used_ = False
    self.fallback_metadata_ = None

    # USABILITY-AWARE PURE-FORM RETENTION (2026-06-17). On an ADDITIVE target whose terms share
    # operands, the MI greedy keeps a high-MI CROSS-MIX feature and drops the pure single-pair forms
    # (``a**2/b`` / ``log(c)*sin(d)``) as conditionally redundant -- the right call for a TREE model but
    # not for the LINEAR/additive downstream, which needs the clean pure form the lossy cross-mix cannot
    # replace. Re-attach a pure single-pair engineered form whenever a CROSS-VALIDATED linear wrapper
    # confirms it lowers the linear CV-MAE on top of the current selection AND the pair is not already
    # represented by a pure (<=2-operand) selected feature. Purely ADDITIVE (nothing MI-selected is
    # removed; support_ untouched) and no-op when the pure form adds no linear value -> byte-identical
    # there. Only when FE is enabled (fe_max_steps>0); skipped on the fe-disabled raw-only path.
    # Names of engineered survivors RE-ATTACHED by the retention passes below (AFTER the main raw-vs-
    # engineered redundancy sweep). The post-retention drop only re-litigates raws against THESE -- the
    # main sweep already vetted raws against survivors it could see (so a genuine pair-interaction operand
    # the main sweep kept is not re-dropped by the stricter post-retention margin).
    _retention_added_eng_names: set = set()
    if fe_max_steps > 0:
        try:
            from .._fe_pure_form_retention import retain_usable_pure_forms

            _retain_extra = retain_usable_pure_forms(
                self, X, getattr(self, "_fe_prewarp_y_continuous_", None),
                seed=int(getattr(self, "random_seed", 0) or 0), verbose=verbose,
            )
            # ENGINEERED-SUBSUMPTION GUARD (2026-06-20). The pure-form retention runs AFTER the post-FE
            # engineered-vs-engineered CMI redundancy gate, so a re-attached pure form is never tested
            # against the engineered survivors admitted BEFORE retention. When an incumbent survivor is a
            # FUSED compound that already carries BOTH additive halves of the target (the canonical
            # ``add(neg(mul(sqr(a),reciproc(b))),neg(mul(log(c),sin(d))))`` for y=a**2/b+log(c)*sin(d)),
            # a re-attached pure half (``mul(log(c),sin(d))`` / ``div(sqr(a),sin(b))``) is FULLY redundant
            # given it -- the fragmentation regression (one compound PLUS several sub-fragments). Re-run
            # the SAME n-invariant debiased-excess CMI subsumption check the S5 gate validated, conditioning
            # each retention candidate on the INCUMBENT (pre-retention) engineered survivors, and skip any
            # whose information collapses given them. A genuinely COMPLEMENTARY pure form (one the incumbents
            # do not span -- the case this retention pass exists to rescue) keeps a large conditional excess
            # and is admitted; only sub-fragments of an incumbent compound are dropped. No-op (byte-identical)
            # when there is no incumbent engineered survivor to condition on.
            if _retain_extra:
                try:
                    from .._fe_retention_subsumption import retention_form_is_subsumed
                    from ..engineered_recipes._recipe_dispatch import apply_recipe as _ret_apply
                    _inc_names = [str(_n) for _n in (self._engineered_features_ or [])]
                    _inc_cont = []
                    for _in in _inc_names:
                        _iv = _eng_continuous_snapshot.get(_in)
                        if _iv is not None and np.asarray(_iv).shape[0] == int(data.shape[0]):
                            _inc_cont.append(np.asarray(_iv, dtype=np.float64).ravel())
                    if _inc_cont:
                        _ret_y = np.ascontiguousarray(np.asarray(classes_y)).ravel()
                        _ret_y_cont = getattr(self, "_fe_prewarp_y_continuous_", None)
                        if _ret_y_cont is None:
                            try:
                                _yv = y.values if hasattr(y, "values") else np.asarray(y)
                                _yv = np.asarray(_yv).reshape(-1)
                                if _yv.shape[0] == int(data.shape[0]) and np.issubdtype(np.asarray(_yv).dtype, np.number):
                                    _ret_y_cont = _yv
                            except Exception:
                                _ret_y_cont = None
                        _ret_seed = int(getattr(self, "random_seed", 0) or 0)
                        _kept_extra = []
                        for _r_recipe, _r_name in _retain_extra:
                            try:
                                _cv = np.asarray(_ret_apply(_r_recipe, X), dtype=np.float64).ravel()
                                _cv = np.nan_to_num(_cv, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            except Exception:
                                _kept_extra.append((_r_recipe, _r_name))  # cannot replay -> retain (conservative)
                                continue
                            if _cv.shape[0] == int(data.shape[0]) and retention_form_is_subsumed(
                                cand_continuous=_cv, incumbent_continuous=_inc_cont,
                                y_binned=_ret_y, y_continuous=_ret_y_cont, seed=_ret_seed,
                            ):
                                if verbose:
                                    logger.info(
                                        "MRMR usability-aware retention: SKIP re-attaching pure form %r -- "
                                        "fully subsumed by an incumbent engineered compound (conditional CMI "
                                        "collapses given the pre-retention survivors).", _r_name,
                                    )
                                continue
                            _kept_extra.append((_r_recipe, _r_name))
                        _retain_extra = _kept_extra
                except Exception as _subsume_exc:
                    if verbose:
                        logger.info("MRMR retention engineered-subsumption guard skipped (%s: %s).", type(_subsume_exc).__name__, _subsume_exc)
            for _r_recipe, _r_name in _retain_extra:
                self._engineered_recipes_.append(_r_recipe)
                self._engineered_features_.append(_r_name)
                _retention_added_eng_names.add(str(_r_name))
            if _retain_extra and verbose:
                logger.info(
                    "MRMR usability-aware retention: re-attached %d linearly-usable pure pair form(s) " "the MI greedy dropped for a higher-MI cross-mix: %s",
                    len(_retain_extra),
                    [n for _, n in _retain_extra],
                )
        except Exception as _retain_exc:  # never let the optional retention break a fit
            if verbose:
                logger.info("MRMR usability-aware pure-form retention skipped (%s: %s).", type(_retain_exc).__name__, _retain_exc)

        # USABILITY-AWARE RAW RETENTION (2026-06-18). The companion to the pure-form retention above for the
        # case where the genuinely useful structure is a RAW the MI greedy under-ranked, not a pair form.
        # MRMR ranks raws by binned MI, which under-values a linearly-usable raw whose marginal-MI estimate is
        # small -- e.g. operands g/k of a WEAK additive ratio term ``+ g/k`` in y = w*a**2/b + g/k +
        # log(c)*sin(d): binned MI ~0.01-0.02 (below the relevance floor) yet linear corr ~0.15-0.24 and a tree
        # recovers the ratio. Both are dropped from support_, the pure-form retention cannot rescue the pair
        # (the clean g/k engineered form is a pool-generation lottery), and the marginal-MI re-attach skips
        # them (MI below floor, not a recipe operand) -> the FE space loses the g/k signal and a downstream
        # model scores BELOW raw-only (BUG3 "FE harmful"; the I5 ratio_plus_trig case). The CV-MAE linear
        # wrapper (the same one the pure-form retention trusts) run over the RAW passthroughs surfaces these
        # under-ranked raws and -- crucially -- rejects pure-noise raws (they do not lower the average CV-MAE).
        # Re-attaches only raws NOT already in support_; purely additive (no engineered recipe touched).
        try:
            from .._fe_pure_form_retention import retain_usable_raw_columns

            _raw_extra = retain_usable_raw_columns(
                self, X, getattr(self, "_fe_prewarp_y_continuous_", None),
                seed=int(getattr(self, "random_seed", 0) or 0), verbose=verbose,
            )
            # CLUSTER-COLLAPSE EXCLUSION (2026-06-18). ``retain_usable_raw_columns`` ranks raws by
            # linear usability and is OBLIVIOUS to the cluster-aggregate / DCD redundancy collapse that
            # the support chokepoint above already applied. A perfectly-collinear duplicate (``z=2a+3``)
            # is maximally linearly-usable, so this pass happily re-attaches the very cluster member the
            # chokepoint stripped -- re-injecting the redundancy and selecting BOTH members of a
            # collinear pair (test_duplicate_collinear_handled_and_recorded). Mirror the same exclusion
            # the raw-signal augmentation below applies: never re-attach a raw the cluster collapse
            # already folded into another representative / a denoised aggregate.
            if _raw_extra:
                # Exclude the NON-REPRESENTATIVE members of every redundancy cluster: per cluster
                # exactly ONE representative (the strongest member) stays eligible for re-attachment,
                # mirroring the chokepoint's keep-one-strip-rest. ``_cluster_aggregate_removals_`` (the
                # explicit 'replace'-mode removals) are excluded outright -- they are folded into a
                # denoised aggregate that already represents the cluster.
                # Exclude at most all-but-one member of every cluster, so the pass can never select
                # BOTH members of a redundant pair, while still re-attaching ONE representative when the
                # whole cluster was dropped. For each cluster, of the members ``retain_usable_raw_columns``
                # surfaced, keep the first (it is the strongest by the pass's own usability ranking) and
                # exclude the rest; if a cluster member is ALREADY in ``selected_vars`` that member is the
                # representative, so exclude every cluster member from ``_raw_extra`` (no second copy).
                _rr_excl_names = set(str(_n) for _n in (getattr(self, "_cluster_aggregate_removals_", None) or []))
                _cm_rr = getattr(self, "cluster_members_", None)
                if isinstance(_cm_rr, dict):
                    _rr_raw = set(self.feature_names_in_)
                    _rr_sel_names = {self.feature_names_in_[int(v)] for v in selected_vars if int(v) < len(self.feature_names_in_)}
                    _rr_order = {str(_nm): _i for _i, _nm in enumerate(_raw_extra)}
                    for _rr_anchor, _rr_members in _cm_rr.items():
                        _a = str(_rr_anchor)
                        _ms = [str(_m) for _m in _rr_members] if isinstance(_rr_members, (list, tuple, set)) else []
                        if _a not in _rr_raw:
                            # aggregate/engineered anchor: every raw member is folded into the aggregate.
                            _rr_excl_names.update(_n for _n in _ms if _n in _rr_raw)
                            continue
                        _grp = [_n for _n in ([_a] + _ms) if _n in _rr_raw]
                        if len(_grp) < 2:
                            continue
                        if any(_n in _rr_sel_names for _n in _grp):
                            # a representative already survived -> drop every cluster member from re-attach.
                            _rr_excl_names.update(_grp)
                        else:
                            # whole cluster dropped -> keep the single member the retention ranked highest.
                            _cands = [_n for _n in _grp if _n in _rr_order]
                            if _cands:
                                _keep = min(_cands, key=lambda _n: _rr_order[_n])
                                _rr_excl_names.update(_n for _n in _grp if _n != _keep)
                            else:
                                _rr_excl_names.update(_grp)
                # SUBSUMED-OPERAND EXCLUSION (signal-aware, variant-3). A raw that is an operand of a
                # SURVIVING engineered feature MAY be fully represented by that feature (re-attaching it then
                # re-injects the raw-redundancy I4b forbids) -- but it may instead carry a large PRIVATE signal
                # the engineered child only partially tracks (e.g. a dominant linear term ``y += 2*a`` that a
                # nonlinear nesting ``sub(log(a),...)`` cannot capture). A blanket name-token exclusion drops
                # BOTH cases and silently destroys genuine raw signal (fs_robustness: a linear ``y`` whose raws
                # are all folded into nonlinear engineered survivors loses every raw -> empty support). Decide
                # PER RAW with the same conditional-redundancy discriminator the rescue/drop passes use: exclude
                # ONLY raws truly subsumed by the engineered survivors consuming them (no private signal given
                # those children); KEEP raws that retain >= RAW_SELF_RETAIN_FRAC of their marginal excess.
                from .._confirm_predictor_engineered import _PARENT_TOKEN_SPLIT as _RR_TOK_SPLIT2
                _rr_raw_set = set(self.feature_names_in_)
                # raw name -> surviving engineered recipe names that consume it as an operand.
                _rr_consumers: dict = {}
                for _en in getattr(self, "_engineered_recipes_", {}) or {}:
                    _en_name = getattr(_en, "name", _en)
                    for _tok in _RR_TOK_SPLIT2.split(str(_en_name)):
                        if not _tok:
                            continue
                        _base = _tok if _tok in _rr_raw_set else (_tok.split("__", 1)[0] if "__" in _tok else None)
                        if _base in _rr_raw_set:
                            _rr_consumers.setdefault(_base, set()).add(str(_en_name))
                # Only the raws actually up for re-attachment need a verdict.
                _rr_cand_subsumed = {str(_n) for _n in _raw_extra if str(_n) in _rr_consumers}
                if _rr_cand_subsumed and not bool(getattr(self, "use_simple_mode", False)):
                    # FULL FE mode: the caller opted into replacing subsumed raws with engineered
                    # survivors, so exclude EVERY engineered operand from re-attachment unconditionally
                    # (the I4b subsumed-raw contract). The signal-aware verdict below is only for SIMPLE
                    # mode, where a linearly-usable raw must survive even when an engineered child encodes
                    # it nonlinearly.
                    _rr_excl_names.update(_rr_cand_subsumed)
                elif _rr_cand_subsumed:
                    try:
                        from .._fe_raw_redundancy_drop import raw_retains_signal_given_genuine_children as _rr_keep
                        from .._mi_greedy_cmi_fe import _quantile_bin as _rr_qbin
                        _rr_cols_idx = {nm: i for i, nm in enumerate(cols)}
                        _rr_y = np.ascontiguousarray(np.asarray(classes_y)).ravel().astype(np.int64)
                        _rr_eng_cont = _eng_continuous_snapshot or {}
                        _rr_seed = int(getattr(self, "random_seed", 0) or 0)
                        for _base in _rr_cand_subsumed:
                            _ci = _rr_cols_idx.get(_base)
                            if _ci is None:
                                _rr_excl_names.add(_base)  # cannot test -> keep the conservative exclusion
                                continue
                            _raw_b = np.asarray(data[:, _ci]).astype(np.int64).ravel()
                            _child_bins = []
                            for _en_name in _rr_consumers.get(_base, ()):  # genuine engineered survivors
                                _cci = _rr_cols_idx.get(_en_name)
                                if _cci is not None:
                                    _child_bins.append(np.asarray(data[:, _cci]).astype(np.int64).ravel())
                                elif _en_name in _rr_eng_cont:
                                    _child_bins.append(
                                        np.asarray(_rr_qbin(np.asarray(_rr_eng_cont[_en_name], dtype=np.float64), nbins=10)).astype(np.int64).ravel()
                                    )
                            if not _child_bins:
                                continue  # no usable child to condition on -> not provably subsumed -> KEEP
                            try:
                                _retains = _rr_keep(raw_bin=_raw_b, y_bin=_rr_y,
                                                    genuine_child_bins=_child_bins,
                                                    allow_linear_usability=bool(getattr(self, "use_simple_mode", False)),
                                                    seed=_rr_seed)
                            except Exception:
                                _retains = True  # estimator error -> never drop genuine signal
                            if not _retains:
                                _rr_excl_names.add(_base)  # truly subsumed -> exclude from re-attach
                    except Exception:
                        # discriminator unavailable -> fall back to the conservative blanket exclusion.
                        _rr_excl_names.update(_rr_cand_subsumed)
                if _rr_excl_names:
                    _raw_extra = [_nm for _nm in _raw_extra if str(_nm) not in _rr_excl_names]
            if _raw_extra:
                _name_to_in_idx = {nm: i for i, nm in enumerate(getattr(self, "feature_names_in_", []) or [])}
                # Append to the local ``selected_vars`` (the canonical raw-support list every downstream
                # step -- n_features_, the marginal-MI augmentation, the elbow trim, and the final
                # ``self.support_ = np.array(selected_vars)`` -- reads), NOT directly to ``self.support_``:
                # a later block re-derives ``support_`` from ``selected_vars`` and would clobber a direct
                # ``support_`` edit. Keeps every consumer consistent.
                _cur_set = set(int(v) for v in selected_vars)
                _added_idx = []
                for _nm in _raw_extra:
                    _idx = _name_to_in_idx.get(_nm)
                    if _idx is not None and int(_idx) not in _cur_set:
                        # Honour the caller's pinned search space: the usability-aware retention runs AFTER the
                        # factors_names_to_use / factors_to_use chokepoint (above), so a re-attached raw outside
                        # the pinned pool would re-leak a forbidden column into support_.
                        if _allowed_raw_idx is not None and int(_idx) not in _allowed_raw_idx:
                            continue
                        selected_vars.append(int(_idx))
                        _cur_set.add(int(_idx))
                        _added_idx.append(int(_idx))
                if _added_idx:
                    self.support_ = np.array(selected_vars, dtype=np.int64)
                    if verbose:
                        logger.info(
                            "MRMR usability-aware raw retention: re-attached %d linearly-usable raw(s) the "
                            "MI greedy under-ranked: %s", len(_added_idx), _raw_extra,
                        )
        except Exception as _raw_retain_exc:  # never let the optional retention break a fit
            if verbose:
                logger.info("MRMR usability-aware raw retention skipped (%s: %s).", type(_raw_retain_exc).__name__, _raw_retain_exc)

    # POST-RETENTION RAW-REDUNDANCY DROP (BUG1, 2026-06-19). The main raw-vs-engineered
    # redundancy sweep (above, ~line 7915) runs on the screen-stage ``selected_vars`` BEFORE
    # the usability-aware pure-form retention re-attaches an engineered survivor. When that
    # retention adds a MULTI-OPERAND composite (e.g. ``div(qubed(a),sin(b))``) AFTER the
    # sweep, the raw operands it subsumes (``a``, ``b``) are still in ``selected_vars`` and no
    # later pass conditions them on the freshly-attached child -- so a fully-subsumed raw rides
    # into ``support_`` beside the composite that captures it (the I4b end-to-end violation).
    # Re-run the SAME n-invariant conditional-redundancy verdict on the FINAL selection, with
    # the now-complete engineered survivor set (incl. the retained pure forms) as the anchor.
    # Only DROPS raws fully subsumed by a surviving MULTI-SOURCE child; a genuine private raw
    # (large independent residual) and a raw consumed by no surviving engineered feature are
    # KEPT (the DPI-trap filter + self-retention leg inside the helper enforce this). Off when
    # the drop sweep is disabled (shares ``fe_drop_redundant_raw_operands``).
    if (getattr(self, "fe_drop_redundant_raw_operands", True)
            and getattr(self, "redundancy_policy", "emit_both") == "drop"
            and selected_vars and getattr(self, "_engineered_recipes_", None)):
        try:
            from .._fe_raw_redundancy_drop import drop_redundant_raw_operands as _post_drop
            from ..engineered_recipes._recipe_dispatch import apply_recipe as _post_apply
            from .._mi_greedy_cmi_fe import _quantile_bin as _post_qbin

            _post_raw_set = set(self.feature_names_in_)
            # Final engineered survivor recipes (name -> EngineeredRecipe); these are the
            # columns that actually reach transform() output.
            # Anchor ONLY on engineered survivors RE-ATTACHED by the retention passes (after the main
            # sweep). The main raw-vs-engineered sweep already vetted every raw against the survivors it
            # could see, so re-litigating those raws here -- with the stricter post-retention margin --
            # wrongly drops a genuine pair-interaction operand the main sweep KEPT (TestPairInteraction:
            # x_a/x_b in y=x_a+x_b+2*x_a*x_b, main-sweep cmi 1.21x floor -> KEEP, post 1.5x -> DROP). The
            # post-retention sweep exists ONLY for composites retention attached after the sweep ran.
            _post_recipes: dict = {}
            for _r in self._engineered_recipes_ or []:
                _nm = getattr(_r, "name", None)
                if _nm is not None and _nm not in _post_raw_set and str(_nm) in _retention_added_eng_names:
                    _post_recipes[str(_nm)] = _r
            # Selected raw operand cols-indices (selected_vars is in feature_names_in_ space here;
            # map each surviving raw back to its cols-space index by name).
            _post_sel_raw_names = [self.feature_names_in_[int(v)] for v in selected_vars if 0 <= int(v) < len(self.feature_names_in_)]
            if _post_recipes and _post_sel_raw_names:
                _post_cols = list(cols)
                _post_data = data
                _post_eng_cont = dict(_eng_continuous_snapshot or {})
                _post_extra_cols: list = []
                # Ensure each engineered survivor has a cols-space column + continuous snapshot;
                # replay any retained pure form that the FE-step matrix does not already carry.
                _n_rows_post = int(data.shape[0])
                for _enm, _erec in _post_recipes.items():
                    if _enm not in _post_eng_cont:
                        try:
                            _vals = np.asarray(_post_apply(_erec, X), dtype=np.float64).ravel()
                            _vals = np.nan_to_num(_vals, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                            if _vals.shape[0] == _n_rows_post:
                                _post_eng_cont[_enm] = _vals
                        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                            logger.debug("suppressed in _fit_impl_core.py:8754: %s", e)
                            continue
                    if _enm not in _post_cols and _enm in _post_eng_cont:
                        _post_extra_cols.append(_post_qbin(np.asarray(_post_eng_cont[_enm], dtype=np.float64), nbins=10))
                        _post_cols.append(_enm)
                if _post_extra_cols:
                    _post_data = np.column_stack([data] + [np.asarray(c).reshape(-1, 1) for c in _post_extra_cols])
                _post_name_to_idx = {nm: i for i, nm in enumerate(_post_cols)}
                _post_sel_idx = []
                for _rn in _post_sel_raw_names:
                    _ci = _post_name_to_idx.get(_rn)
                    if _ci is not None:
                        _post_sel_idx.append(_ci)
                for _enm in _post_recipes:
                    _ci = _post_name_to_idx.get(_enm)
                    if _ci is not None and _ci not in _post_sel_idx:
                        _post_sel_idx.append(_ci)
                _has_eng_post = any(_post_cols[i] not in _post_raw_set for i in _post_sel_idx)
                _has_raw_post = any(_post_cols[i] in _post_raw_set for i in _post_sel_idx)
                if _has_eng_post and _has_raw_post:
                    _y_cont_post = None
                    try:
                        _yv = y.values if hasattr(y, "values") else np.asarray(y)
                        _yv = np.asarray(_yv).reshape(-1)
                        if _yv.shape[0] == _n_rows_post and np.issubdtype(np.asarray(_yv).dtype, np.number):
                            _y_cont_post = _yv
                    except Exception:
                        _y_cont_post = None
                    _, _post_dropped = _post_drop(
                        data=_post_data, cols=_post_cols, selected_cols_idx=_post_sel_idx,
                        raw_name_set=_post_raw_set, y_binned=classes_y, y_continuous=_y_cont_post,
                        engineered_continuous=_post_eng_cont,
                        replayable_eng_names=set(_post_recipes.keys()), recipes=_post_recipes,
                        raw_X=X, floor_margin_mult=1.5,
                        linear_usability_keep=bool(getattr(self, "use_simple_mode", False)),
                        seed=int(getattr(self, "random_seed", 0) or 0), verbose=verbose,
                    )
                    if _post_dropped:
                        _post_drop_set = set(_post_dropped)
                        self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) | _post_drop_set
                        selected_vars = [
                            v for v in selected_vars if not (0 <= int(v) < len(self.feature_names_in_) and self.feature_names_in_[int(v)] in _post_drop_set)
                        ]
                        # NEVER-EMPTY RAW FLOOR (mirrors the main sweep): the post-retention drop may not
                        # empty the raw support. If no raw survives, re-add the strongest dropped raw (by
                        # marginal MI) as the representative; the engineered survivor still rides along.
                        if not selected_vars:
                            _bf_idx, _bf_rel = None, float("-inf")
                            _tgt_pf = np.asarray(target_indices, dtype=np.int64)
                            _fn_pf = np.asarray(nbins, dtype=np.int64)
                            for _dn in _post_drop_set:
                                try:
                                    _ci = cols.index(_dn)
                                except ValueError:
                                    continue
                                try:
                                    from ..info_theory import mi as _pf_mi
                                    _rel = float(_pf_mi(data, np.array([int(_ci)], dtype=np.int64), _tgt_pf, _fn_pf))
                                except Exception:
                                    _rel = 0.0
                                if _rel > _bf_rel:
                                    _bf_rel, _bf_idx = _rel, _dn
                            if _bf_idx is not None and _bf_idx in self.feature_names_in_:
                                selected_vars = [self.feature_names_in_.index(_bf_idx)]
                                self._raw_redundancy_dropped_ = set(getattr(self, "_raw_redundancy_dropped_", None) or set()) - {_bf_idx}
                        self.support_ = np.array(selected_vars, dtype=np.int64)
                        if verbose:
                            logger.info(
                                "MRMR post-retention raw-redundancy drop: removed %d raw operand(s) "
                                "subsumed by an engineered survivor re-attached AFTER the main sweep: %s",
                                len(_post_dropped), sorted(_post_drop_set),
                            )
        except Exception as _post_exc:
            logger.warning(
                "MRMR post-retention raw-redundancy drop failed: %s; keeping the support.",
                _post_exc,
            )

    # n_features_ reports the column count produced by transform() = raw selected + engineered (replayable via _engineered_recipes_). Higher-order
    # engineered features without a replayable recipe were already warned about above and are NOT counted (they don't appear in transform output).
    n_engineered_out = len(self._engineered_recipes_)
    if selected_vars:
        self.n_features_ = len(selected_vars) + n_engineered_out
        # RAW-SIGNAL-RETENTION augmentation (Fix B). On a wide composite-FE pool the screen often confirms an ENGINEERED derivative of a strong raw signal (e.g.
        # ``x1__resid_by__cat_a`` whose cat-residual MI exceeds raw ``x1``), which then conditionally redundifies the raw column so raw ``x1`` is dropped from
        # ``support_`` even though it is genuine, generalising signal. The empirical-null debiasing makes the per-feature ``cached_MIs`` an honest relevance ranking
        # (cardinality / heavy-tail / monotone in-sample inflation removed), so a raw feature that clears the relevance floor AND is the SOURCE of a confirmed
        # engineered child is genuine signal that the greedy step merely shadowed behind its derivative -- we re-attach it. The augmentation is deliberately narrow:
        # it rescues ONLY columns whose name appears as a source token in some engineered recipe name, so it can never re-inflate a redundant block of near-duplicate
        # raw columns (those have no engineered child) and never overrides DCD / cluster-aggregate redundancy collapse. ``min_features_fallback==0`` opts out.
        _min_fb_aug = int(getattr(self, "min_features_fallback", 0) or 0)
        if _min_fb_aug >= 1 and self.n_features_in_ > 0 and hasattr(self, "cached_MIs") and n_engineered_out > 0:
            try:
                # Source tokens referenced by any confirmed engineered recipe (split on the engineered-name separators ``__``, ``(``, ``|``, ``)``, ``,``).
                import re as _re_aug
                _eng_names = []
                for _r in self._engineered_recipes_ or []:
                    _nm = getattr(_r, "output_name", None) or getattr(_r, "name", None) or (_r.get("name") if isinstance(_r, dict) else None)
                    if _nm:
                        _eng_names.append(str(_nm))
                _eng_tokens = set()
                for _nm in _eng_names:
                    for _tok in _re_aug.split(r"[^0-9A-Za-z_]+", _nm.replace("__", " ")):
                        if _tok:
                            _eng_tokens.add(_tok)
                # Members folded into a denoised aggregate -- cluster_aggregate 'replace' mode (``_cluster_aggregate_removals_``) or a DCD PC1/mean_z swap (``cluster_members_``) -- are
                # ALREADY represented by that aggregate and were deliberately removed from the support. The token scan above matches them anyway because the member NAME survives as a
                # token inside OTHER engineered recipe names (e.g. ``add(refl0,sin(indep))``), so without this exclusion the augmentation resurrects the very members 'replace' mode and
                # DCD just collapsed, re-injecting the redundancy. Mirror the same exclusion the raw-retention block and the additional-RFECV rescue pool apply.
                _aug_excluded_names = set(getattr(self, "_cluster_aggregate_removals_", None) or [])
                _cm_for_aug = getattr(self, "cluster_members_", None)
                if isinstance(_cm_for_aug, dict):
                    for _anchor, _members in _cm_for_aug.items():
                        _aug_excluded_names.add(_anchor)
                        if isinstance(_members, (list, tuple, set)):
                            _aug_excluded_names.update(_members)
                _name_to_cols_idx_aug = {c: i for i, c in enumerate(cols)}
                _abs_floor_aug = float(getattr(self, "min_relevance_gain", 0.0) or 0.0)
                _rel_frac_aug = float(getattr(self, "min_relevance_gain_relative_to_first", 0.0) or 0.0)
                _raw_mi_aug = []
                for _i in range(self.n_features_in_):
                    _name = self.feature_names_in_[_i] if _i < len(self.feature_names_in_) else None
                    _ci = _name_to_cols_idx_aug.get(_name)
                    _mi = self.cached_MIs.get((_ci,), 0.0) if _ci is not None else 0.0
                    _raw_mi_aug.append((_i, _name, float(_mi)))
                _max_mi_aug = max((m for _, _, m in _raw_mi_aug), default=0.0)
                _floor_aug = max(_abs_floor_aug, _max_mi_aug * _rel_frac_aug)
                _selected_set = set(int(v) for v in selected_vars)
                # LARGE-N SCOPE (2026-06-08 regression fix): this augmentation re-attaches a raw
                # column whose NAME is a source token of a confirmed engineered recipe and whose
                # MARGINAL MI clears the relevance floor. Marginal MI cannot tell a FULLY-ABSORBED
                # operand (``a`` in ``div(sqr(a),abs(b))`` for ``y=a**2/b`` -- high marginal MI, ZERO
                # conditional signal beyond the ratio) from a genuine independent term, so on the
                # canonical composite fixtures it resurrected exactly the redundant raw operands the
                # post-FE re-selection had correctly dropped (support_rank -1, no gain). At large n the
                # re-selection's conditional-MI redundancy verdict is reliable, so we DEFER to it: skip
                # the token-based re-attach for any raw column that is an operand of a SURVIVING
                # engineered feature. The small-n regime (where the augmentation was validated) keeps
                # the marginal-MI re-attach. Threshold shared with the raw-retention pass above.
                # ``selected_vars`` here holds RAW indices into ``feature_names_in_`` (the surviving
                # engineered columns live in ``self._engineered_recipes_`` / ``_engineered_features_``,
                # not in ``selected_vars``). Derive the surviving engineered OPERANDS from the recipe
                # source tokens (``_eng_tokens`` already = every source token of every confirmed recipe,
                # restricted here to raw names), since every confirmed engineered child contributes its
                # operands to that set. A raw column that is such an operand was dropped by the
                # re-selection IN FAVOUR of its engineered child -> at large n, do not resurrect it.
                _aug_max_n = int(getattr(self, "fe_raw_retention_max_n", 20000) or 0)
                _aug_large_n = int(data.shape[0]) > _aug_max_n
                _raw_names_for_aug = set(self.feature_names_in_)
                _surviving_eng_operands = {t for t in _eng_tokens if t in _raw_names_for_aug} if _aug_large_n else set()
                # Operands the n-invariant conditional-redundancy sweep dropped are authoritative
                # at EVERY n -- never re-attach them here (the marginal-MI token match cannot tell
                # a fully-subsumed operand from a genuine independent term; the excess-CMI sweep can).
                _redund_dropped_names = set(getattr(self, "_raw_redundancy_dropped_", None) or ())
                _to_add = [i for i, _name, m in sorted(_raw_mi_aug, key=lambda kv: (-kv[2], kv[0]))
                           if m > _floor_aug and i not in _selected_set and _name in _eng_tokens
                           and _name not in _aug_excluded_names
                           and _name not in _redund_dropped_names
                           and not (_aug_large_n and _name in _surviving_eng_operands)]
                if _to_add:
                    selected_vars.extend(_to_add)
                    self.support_ = np.array(selected_vars, dtype=np.int64)
                    self.n_features_ = len(selected_vars) + n_engineered_out
            except Exception as _exc_aug:
                logger.warning("MRMR raw-signal-retention augmentation failed: %s; keeping greedy support.", _exc_aug)
    elif getattr(self, "_redundancy_emptied_raw_", False):
        # The raw support is empty because the n-invariant conditional-redundancy sweep
        # deliberately dropped every raw operand (each fully subsumed by a surviving
        # engineered child) -- an INTENDED, complete engineered-only support, NOT a
        # "screen returned 0 raw" emergency. SKIP the empty-raw rescue entirely; firing
        # it would resurrect the dropped operands or pull in the next pure-noise column
        # ranked by marginal MI (measured ws1: ``e`` rescued at n=1000, ``a`` re-added at
        # n=25000). n_features_ is the engineered-only count.
        self.n_features_ = n_engineered_out
    else:
        # Empty-RAW-support fallback rescue carved into _finalise.py (Tier E partial split).
        # Threads the instance + fit-body locals explicitly; mutates self.support_ / n_features_ /
        # fallback_used_ / fallback_metadata_ in place. Behaviour byte-for-byte identical to the
        # former inlined branch.
        from ._finalise import _finalise_empty_support_fallback
        _finalise_empty_support_fallback(self, n_engineered_out, cols, data, nbins, target_indices)

    # ---------------------------------------------------------------------------------------------------------------
    # Report FS results
    # ---------------------------------------------------------------------------------------------------------------

    if verbose:
        predictors_str = ", ".join([f"{el['name']}: {el['gain']:.4f}" for el in predictors[:50]])
        predictors_str = textwrap.shorten(predictors_str, width=300)
        logger.info("MRMR+ selected %d out of %d features: %s", self.n_features_, self.n_features_in_, predictors_str)

    # Refresh the params slot with POST-fit values before storing: should fit ever resolve/normalise a
    # param in place (RFECV does this with ``scoring``), the entry-time params fingerprint would never
    # match the NEXT fit's ``get_params`` and identical refits would never skip. The data slots
    # (shapes/hashes/columns) stay as computed at fit entry.
    try:
        signature = signature[:-1] + (_hashable_params_signature(self.get_params(deep=True)),)
    except Exception:
        signature = signature[:-1] + (object(),)  # unique token => next identical fit refits (conservative)
    self.signature = signature
    # ran_out_of_time was set only by the outer FE-loop deadline (line ~6714). screen_predictors honours
    # self.max_runtime_mins on its OWN and can return a truncated selection without the FE loop ever tripping, so a
    # screen-level timeout was reported as ran_out_of_time_=False -- misleading a caller inspecting why selection was
    # thin. OR-in a total-elapsed-vs-budget check so any stage that pushed the fit past its budget is reflected.
    if self.max_runtime_mins is not None and (timer() - start_time) / 60.0 >= self.max_runtime_mins:
        ran_out_of_time = True
    self.ran_out_of_time_ = ran_out_of_time

    # Store self in process-wide cache so cloned MRMR instances fit on the same (X, y) arrays can replay
    # this fitted state instead of re-running cat-FE + permutation. Bound the LRU by ``fit_cache_max``;
    # the default (4) covers a typical model suite without thrashing and long-lived workers no longer leak.
    if _cache_key is not None:
        # Whole store + LRU/byte-cap eviction held under the cache lock so a concurrent fit cannot interleave its
        # own ``__setitem__``/``popitem``/``move_to_end`` (KeyError, wrong-entry eviction) or iterate ``.values()``
        # via ``_mrmr_cache_bytes_total`` while another thread mutates the dict.
        with _MRMR_FIT_CACHE_LOCK:
            MRMR._FIT_CACHE[_cache_key] = self
            MRMR._FIT_CACHE.move_to_end(_cache_key)
            # ``fit_cache_max=0`` is the operator-explicit "disable LRU" sentinel
            # (e.g. for memory-constrained suites where the 4-entry cache pins
            # too much state). The previous ``or 4`` form silently restored the
            # default cap, so cache-off was a no-op. ``None`` (unset attr) still
            # folds to 4.
            _cap_raw = getattr(self, "fit_cache_max", 4)
            _cap = int(4 if _cap_raw is None else _cap_raw)
            if _cap <= 0:
                MRMR._FIT_CACHE.clear()
            else:
                while len(MRMR._FIT_CACHE) > _cap:
                    MRMR._FIT_CACHE.popitem(last=False)
            # Byte-size cap on top of entry count: a 1k-feature suite carrying 4 cached MRMR instances each
            # holding _selectors_ / _engineered_features_ state can exceed 1 GB of process RSS.
            # ``fit_cache_max_mb`` (default 1024 MB; env override ``MLFRAME_MRMR_FIT_CACHE_MAX_MB``) bounds the
            # aggregate cache footprint.
            _mb_cap_raw = getattr(self, "fit_cache_max_mb", None)
            if _mb_cap_raw is None:
                _env_mb = os.environ.get("MLFRAME_MRMR_FIT_CACHE_MAX_MB", "1024")
                try:
                    _mb_cap = float(_env_mb)
                except ValueError:
                    _mb_cap = 1024.0
            else:
                try:
                    _mb_cap = float(_mb_cap_raw)
                except (TypeError, ValueError):
                    _mb_cap = 1024.0
            if _mb_cap > 0 and _cap > 0 and len(MRMR._FIT_CACHE) > 0:
                _byte_cap = _mb_cap * (1024**2)
                while len(MRMR._FIT_CACHE) > 1 and _mrmr_cache_bytes_total() > _byte_cap:
                    MRMR._FIT_CACHE.popitem(last=False)
    # 2026-05-30 Wave 8 — post-fit UAED auto-size. When enabled, replaces the
    # configured ``min_features_fallback`` floor with an automatic elbow on
    # the per-feature MI gain curve. Relevance trace is taken from the
    # ``mrmr_gains_`` attribute (Wave-7 audit landed this trace in the
    # standard fit output); if missing, this step no-ops.
    if getattr(self, "uaed_auto_size", False):
        try:
            from .._cmi_perm_stop import uaed_elbow
            gains = np.asarray(getattr(self, "mrmr_gains_", []), dtype=np.float64)
            # UAED runs BEFORE the mrmr_gains_ length-alignment below, so at this point ``gains`` is the
            # raw GREEDY log (one entry per confirmed greedy round) -- often SHORTER than n_features_ when
            # FE/retention appended features the greedy never scored. The public ``mrmr_gains_`` the caller
            # sees is the n_features_-aligned (zero-padded) trace, so the elbow must be computed on that SAME
            # trace; otherwise a frame whose greedy log has <3 rounds but >=3 final features silently skips
            # the elbow (uaed_elbow_ never set, support never trimmed). Zero-extend to n_features_ to match.
            _nf_uaed = int(getattr(self, "n_features_", gains.size) or gains.size)
            if 0 < gains.size < _nf_uaed:
                gains = np.concatenate([gains, np.zeros(_nf_uaed - gains.size, dtype=np.float64)])
            if gains.size >= 3:
                elbow = int(uaed_elbow(gains))
                if 0 < elbow < gains.size and hasattr(self, "support_"):
                    # ``gains`` is the COMBINED trace (raw greedy gains + zero-padded engineered tail), matching the
                    # transform-time feature order [support_ ..., engineered recipes ...]. The elbow index therefore
                    # lives in COMBINED space, but ``support_`` holds RAW indices only. Slicing raw support by a
                    # combined elbow (and setting n_features_ = support_.size) dropped the engineered count while the
                    # recipes still fired in transform -- transform emitted MORE columns than n_features_/mrmr_gains_
                    # claimed (a hard support/output desync). Trim raw support AND engineered recipes in LOCKSTEP so
                    # the retained feature count is exactly elbow+1 in both the state and the transform output.
                    _sup = np.asarray(self.support_)
                    _recipes = list(getattr(self, "_engineered_recipes_", []) or [])
                    _keep = elbow + 1  # combined features to retain
                    _raw_keep = min(_keep, _sup.size)
                    _eng_keep = max(0, _keep - _sup.size)  # <= len(_recipes): gains was zero-extended to n_features_
                    self.support_ = _sup[:_raw_keep]
                    if _recipes and _eng_keep < len(_recipes):
                        self._engineered_recipes_ = _recipes[:_eng_keep]
                    self.n_features_ = int(self.support_.size) + min(_eng_keep, len(_recipes))
                    self.uaed_elbow_ = int(elbow)
        except Exception:  # nosec B110 - non-trivial body
            # UAED is best-effort post-fit; don't break fit() on internal hiccup.
            pass
    # Transient FE-escalation fitting target: full-n array, fit-time only.
    self._fe_escalation_y_rank_ = None
    # Transient prewarp ALS reconstruction target: full-n continuous y, fit-time only.
    self._fe_prewarp_y_continuous_ = None

    # MRMR_GAINS LENGTH ALIGNMENT (2026-06-17, FINAL). ``mrmr_gains_`` is the GREEDY selection log;
    # the FINAL feature count diverges from it -- SHORTER on a degenerate-frame collapse / redundancy /
    # cluster-aggregate exclusion / p>=n cap / UAED elbow trim, LONGER when FE / retention / pseudo-
    # remix re-add appended features the greedy log never scored. The public contract + downstream
    # expect ``len(mrmr_gains_) == n_features_`` (TestSupportGainsAlignment). Reconcile HERE, after every
    # support/n_features_ mutation above is final: keep the top screening gains (descending -- what the
    # UAED elbow already consumed) and pad any FE tail with 0.0. Byte-identical when already aligned.
    try:
        _g = getattr(self, "mrmr_gains_", None)
        _nf_final = int(getattr(self, "n_features_", 0) or 0)
        if _g is not None and _nf_final >= 0 and _g.shape[0] != _nf_final:
            if _g.shape[0] > _nf_final:
                self.mrmr_gains_ = _g[:_nf_final]
            else:
                self.mrmr_gains_ = np.concatenate([_g, np.zeros(_nf_final - _g.shape[0], dtype=np.float64)])
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logger.debug("suppressed in _fit_impl_core.py:9065: %s", e)
        pass

    # SUPPORT_NONLINEAR_ ALIAS RE-SYNC. ``support_nonlinear_`` is set right after the FIRST support_
    # assignment as an alias of the pure-MI support_, but several later passes (usability-aware RAW
    # retention, count-floor rescue, UAED elbow trim) REASSIGN self.support_ to a NEW array, leaving the
    # alias pointing at the stale pre-mutation array. By contract support_nonlinear_ IS the final pure-MI
    # support_, so re-point it here after every support_ mutation (the separate linear/universal lists,
    # when present, are untouched).
    if hasattr(self, "support_nonlinear_"):
        self.support_nonlinear_ = self.support_
    return self
