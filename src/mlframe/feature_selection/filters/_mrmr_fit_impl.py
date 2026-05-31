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

import copy
import gc
import hashlib
import logging
import math
import os
import textwrap
import time
import warnings
from collections import OrderedDict, defaultdict
from itertools import combinations, islice
from timeit import default_timer as timer
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import make_scorer

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _mrmr_instance_state_size_bytes(instance: Any) -> int:
    """Best-effort byte estimate for a single fitted MRMR instance's selector + engineered-features state.

    Used by the LRU eviction byte gate. Walks the small set of large state attributes (``mi_scores_``, ``_selectors_``, ``_engineered_features_``, ``_y_full_hash`` retained y) so the estimate reflects the dominant footprint without paying ``pickle.dumps`` cost on every eviction probe.
    """
    total = 0
    for _attr in ("mi_scores_", "_selectors_", "_engineered_features_", "ranking_", "support_", "selected_features_"):
        try:
            _v = getattr(instance, _attr, None)
            if _v is None:
                continue
            _nb = getattr(_v, "nbytes", None)
            if isinstance(_nb, int):
                total += _nb
                continue
            if isinstance(_v, dict):
                for _vv in _v.values():
                    _vvnb = getattr(_vv, "nbytes", None)
                    if isinstance(_vvnb, int):
                        total += _vvnb
                    else:
                        try:
                            total += int(np.asarray(_vv).nbytes)
                        except Exception:
                            pass
            elif isinstance(_v, (list, tuple)):
                for _item in _v:
                    _inb = getattr(_item, "nbytes", None)
                    if isinstance(_inb, int):
                        total += _inb
        except Exception:
            continue
    return total


def _mrmr_cache_bytes_total() -> int:
    """Sum of state bytes across every cached MRMR instance in MRMR._FIT_CACHE."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    return sum(_mrmr_instance_state_size_bytes(_v) for _v in MRMR._FIT_CACHE.values())


def _fit_impl(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | pd.Series | np.ndarray, groups: pd.Series | np.ndarray = None, **fit_params):
    """We run N selections on data subsets, and pick only features that appear in all selections"""
    # Lazy import: ``.mrmr`` re-imports this module at its module bottom for
    # method binding -> any top-level ``from .mrmr import ...`` here would
    # create a hard import cycle that ``tests/test_meta/test_no_import_cycles.py``
    # flags. Python's module cache makes repeat imports cheap.
    from .mrmr import (
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
    X = self._validate_inputs(X, y)

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
    signature = (X.shape, y.shape, _y_hash_for_sig, _x_hash_for_sig, _x_cols_sig)
    if self.skip_retraining_on_same_shape:
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
        if not _y_full_hash or not _x_full_hash:
            _cache_key = None
        else:
            _cache_key = (_x_sig, _y_sig, _y_name, _y_full_hash, _x_full_hash, _params_sig)
    except Exception:
        _cache_key = None
    if _cache_key is not None and _cache_key in MRMR._FIT_CACHE:
        _cached = MRMR._FIT_CACHE[_cache_key]
        MRMR._FIT_CACHE.move_to_end(_cache_key)
        _replayed = _replay_fitted_state(self, _cached)
        if self.verbose:
            logger.info(
                "MRMR.fit: _FIT_CACHE hit -- replayed %d fitted attrs "
                "from prior fit, skipping cat-FE + permutation.",
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
        X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self._feature_names_in_synthesized_ = True
    else:
        self._feature_names_in_synthesized_ = False

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
    _hybrid_orth_pre_recipes: dict = {}
    if bool(getattr(self, "fe_hybrid_orth_enable", False)):
        # Polars frames: skip with a warning -- hybrid FE pipeline operates on
        # pandas. Native polars support would require a separate code path;
        # not in Layer 23 MVP scope.
        _is_pandas_for_hybrid = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_hybrid:
            warnings.warn(
                "MRMR: fe_hybrid_orth_enable=True but X is not a pandas "
                "DataFrame; hybrid orthogonal-polynomial FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want hybrid FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_univariate_fe import (
                    hybrid_orth_mi_fe_with_recipes,
                    hybrid_orth_mi_pair_fe_with_recipes,
                )
                _y_for_hybrid = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _h_pair_enable = bool(self.fe_hybrid_orth_pair_enable)
                _h_pair_max_degree = int(self.fe_hybrid_orth_pair_max_degree)
                # Restrict the source pool to numeric columns the caller passed
                # via factors_names_to_use (when set); otherwise the hybrid
                # pipeline auto-routes to all numeric columns of X.
                _h_cols = None
                if getattr(self, "factors_names_to_use", None):
                    _h_cols = [
                        c for c in self.factors_names_to_use if c in X.columns
                    ]
                _X_before_hybrid_cols = list(X.columns)
                if _h_pair_enable:
                    X_h, _uni_sc, _cross_sc, _recipes = hybrid_orth_mi_pair_fe_with_recipes(
                        X, _y_for_hybrid,
                        cols=_h_cols,
                        degrees=_h_degrees,
                        basis=_h_basis,
                        top_k=_h_top_k,
                        top_pair_count=_h_top_k,
                        pair_max_degree=_h_pair_max_degree,
                    )
                else:
                    X_h, _uni_sc, _recipes = hybrid_orth_mi_fe_with_recipes(
                        X, _y_for_hybrid,
                        cols=_h_cols,
                        degrees=_h_degrees,
                        basis=_h_basis,
                        top_k=_h_top_k,
                    )
                # Identify appended columns vs the pre-hybrid X.
                _appended = [
                    c for c in X_h.columns if c not in _X_before_hybrid_cols
                ]
                if _appended:
                    X = X_h
                    self.hybrid_orth_features_ = list(_appended)
                    for _r in _recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth: appended %d engineered "
                            "column(s) (univariate + pair): %s",
                            len(_appended), _appended[:8],
                        )
            except Exception as _h_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth FE raised %s: %s; continuing "
                    "without hybrid-FE columns.",
                    type(_h_exc).__name__, _h_exc,
                )
            # 2026-05-31 Layer 32 — extra-basis (B-spline / Fourier) FE stage.
            # Runs only when the master hybrid switch is on AND the user
            # opted in via a non-empty ``fe_hybrid_orth_extra_bases`` tuple.
            # Complementary to the polynomial path: spline catches threshold
            # rules, Fourier catches periodic patterns. Recipes are
            # closed-form (no y), replay safe.
            _extra_bases_cfg = tuple(
                getattr(self, "fe_hybrid_orth_extra_bases", ()) or ()
            )
            # Defensive guard: the polynomial-stage ``try:`` may have raised
            # before defining ``_y_for_hybrid`` / ``_h_top_k``. Bind safe
            # defaults so the extra-basis stage can still run.
            try:
                _y_for_extra = _y_for_hybrid
            except NameError:
                _y_for_extra = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
            if _is_pandas_for_hybrid and _extra_bases_cfg:
                try:
                    from ._orthogonal_univariate_fe import (
                        hybrid_orth_extra_basis_fe_with_recipes,
                    )
                    _fourier_freqs = tuple(
                        float(f) for f in
                        getattr(self, "fe_hybrid_orth_fourier_freqs", (1.0, 2.0))
                    )
                    _spline_knots = int(
                        getattr(self, "fe_hybrid_orth_spline_knots", 5)
                    )
                    _X_before_extra_cols = list(X.columns)
                    # Use the SAME source-column scope as the polynomial stage
                    # (factors_names_to_use restriction).
                    _e_cols = None
                    if getattr(self, "factors_names_to_use", None):
                        _e_cols = [
                            c for c in self.factors_names_to_use if c in X.columns
                        ]
                    X_e, _e_scores, _e_recipes = hybrid_orth_extra_basis_fe_with_recipes(
                        X, _y_for_extra,
                        cols=_e_cols,
                        extra_bases=_extra_bases_cfg,
                        fourier_freqs=_fourier_freqs,
                        spline_knots=_spline_knots,
                        top_k=_top_k_for_extra,
                    )
                    _e_appended = [
                        c for c in X_e.columns if c not in _X_before_extra_cols
                    ]
                    if _e_appended:
                        X = X_e
                        # Extend hybrid_orth_features_ with the extra-basis winners
                        # so the downstream remap / transform pipeline handles them
                        # exactly like the polynomial winners.
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_e_appended)
                        )
                        for _r in _e_recipes:
                            _hybrid_orth_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit hybrid_orth extra-basis: appended %d "
                                "engineered column(s) (spline/fourier): %s",
                                len(_e_appended), _e_appended[:8],
                            )
                except Exception as _e_exc:
                    logger.warning(
                        "MRMR.fit hybrid_orth extra-basis FE raised %s: %s; "
                        "continuing without extra-basis columns.",
                        type(_e_exc).__name__, _e_exc,
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
        _is_pandas_for_mi_greedy = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_mi_greedy:
            warnings.warn(
                "MRMR: fe_mi_greedy_enable=True but X is not a pandas "
                "DataFrame; MI-greedy FE is skipped. Convert to pandas via "
                "X.to_pandas() before fit() if you want MI-greedy FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._mi_greedy_fe import greedy_mi_fe_construct_with_recipes
                _y_for_mig = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                _mig_cols = None
                if getattr(self, "factors_names_to_use", None):
                    _mig_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _mig_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _X_before_mig_cols = list(X.columns)
                X_mg, _mig_scores, _mig_recipes = greedy_mi_fe_construct_with_recipes(
                    X, _y_for_mig,
                    cols=_mig_cols,
                    seed_cols_count=int(self.fe_mi_greedy_seed_cols_count),
                    top_k=int(self.fe_mi_greedy_top_k),
                    include_unary=bool(self.fe_mi_greedy_include_unary),
                    include_binary=bool(self.fe_mi_greedy_include_binary),
                )
                _mig_appended = [
                    c for c in X_mg.columns if c not in _X_before_mig_cols
                ]
                if _mig_appended:
                    X = X_mg
                    self.mi_greedy_features_ = list(_mig_appended)
                    for _r in _mig_recipes:
                        _mi_greedy_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit mi_greedy: appended %d engineered "
                            "column(s): %s",
                            len(_mig_appended), _mig_appended[:8],
                        )
            except Exception as _mig_exc:
                logger.warning(
                    "MRMR.fit mi_greedy FE raised %s: %s; continuing "
                    "without MI-greedy columns.",
                    type(_mig_exc).__name__, _mig_exc,
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
    if bool(getattr(self, "fe_kfold_te_enable", False)):
        _is_pandas_for_te = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_te:
            warnings.warn(
                "MRMR: fe_kfold_te_enable=True but X is not a pandas "
                "DataFrame; K-fold target encoding is skipped. Convert "
                "to pandas via X.to_pandas() before fit() if you want "
                "K-fold TE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._target_encoding_fe import (
                    kfold_target_encode_with_recipes,
                )
                _te_cols_cfg = tuple(
                    getattr(self, "fe_kfold_te_cols", ()) or ()
                )
                # Explicit empty tuple -> auto-detect; explicit names -> use
                # exactly those (after intersecting with X.columns).
                _te_cols = list(_te_cols_cfg) if _te_cols_cfg else None
                if _te_cols is not None:
                    _hybrid_appended = set(self.hybrid_orth_features_ or [])
                    _mig_appended = set(self.mi_greedy_features_ or [])
                    _te_cols = [
                        c for c in _te_cols
                        if c in X.columns
                        and c not in _hybrid_appended
                        and c not in _mig_appended
                    ]
                _y_for_te = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                # TE works for both binary classification and regression as-
                # is (mean of {0,1} = P(y=1); mean of continuous = mean).
                # Cast bool / object to float to avoid type errors inside
                # the mean computation.
                _y_for_te = np.asarray(_y_for_te, dtype=np.float64).ravel()
                _X_before_te_cols = list(X.columns)
                X_te, _te_appended, _te_recipes = kfold_target_encode_with_recipes(
                    X, _y_for_te,
                    cat_cols=_te_cols,
                    n_folds=int(getattr(self, "fe_kfold_te_folds", 5)),
                    smoothing=float(
                        getattr(self, "fe_kfold_te_smoothing", 10.0)
                    ),
                    random_state=int(
                        getattr(self, "random_seed", 0) or 0
                    ),
                )
                # Guard against silent overlap with prior stages: the
                # ``{col}__te`` suffix is dedicated to this stage so the
                # collision pre-condition would require a user-supplied
                # source column literally named ``{src}__te``. Drop any
                # accidental name collision rather than overwrite.
                _te_appended = [
                    c for c in _te_appended if c not in _X_before_te_cols
                ]
                if _te_appended:
                    X = X_te
                    self.kfold_te_features_ = list(_te_appended)
                    # Route through hybrid_orth_features_ so the end-of-fit
                    # remap routes by-name selected items into
                    # _engineered_recipes_ (Layer 23 routing path).
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_te_appended)
                    )
                    for _r in _te_recipes:
                        if _r.name in _te_appended:
                            _kfold_te_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit kfold_te: appended %d engineered "
                            "column(s): %s",
                            len(_te_appended), _te_appended[:8],
                        )
            except Exception as _te_exc:
                logger.warning(
                    "MRMR.fit kfold_te FE raised %s: %s; continuing "
                    "without target-encoded columns.",
                    type(_te_exc).__name__, _te_exc,
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
        _is_pandas_l34 = isinstance(X, pd.DataFrame)
        if not _is_pandas_l34:
            warnings.warn(
                "MRMR: Layer 34 FE (count/frequency/cat_num) enabled but X "
                "is not a pandas DataFrame; the encodings are skipped. "
                "Convert to pandas via X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            from ._count_freq_interaction_fe import (
                count_encode_with_recipes,
                frequency_encode_with_recipes,
                cat_num_interaction_with_recipes,
            )
            from ._target_encoding_fe import auto_detect_te_cols

            _hybrid_appended_l34 = set(self.hybrid_orth_features_ or [])
            _mig_appended_l34 = set(self.mi_greedy_features_ or [])
            _te_appended_l34 = set(self.kfold_te_features_ or [])
            _engineered_seen_l34 = (
                _hybrid_appended_l34 | _mig_appended_l34 | _te_appended_l34
            )

            # ----- Count encoding ----------------------------------------
            if bool(getattr(self, "fe_count_encoding_enable", False)):
                try:
                    _cnt_cfg = tuple(
                        getattr(self, "fe_count_encoding_cols", ()) or ()
                    )
                    if _cnt_cfg:
                        _cnt_cols = [
                            c for c in _cnt_cfg
                            if c in X.columns and c not in _engineered_seen_l34
                        ]
                    else:
                        _cnt_cols = auto_detect_te_cols(
                            X, min_card=5, max_card=500,
                        )
                    _X_before_cnt_cols = list(X.columns)
                    X_c, _cnt_appended, _cnt_recipes = count_encode_with_recipes(
                        X, cat_cols=_cnt_cols,
                    )
                    _cnt_appended = [
                        c for c in _cnt_appended if c not in _X_before_cnt_cols
                    ]
                    if _cnt_appended:
                        X = X_c
                        self.count_encoding_features_ = list(_cnt_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_cnt_appended)
                        )
                        for _r in _cnt_recipes:
                            if _r.name in _cnt_appended:
                                _count_enc_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit count_encoding: appended %d "
                                "engineered column(s): %s",
                                len(_cnt_appended), _cnt_appended[:8],
                            )
                except Exception as _cnt_exc:
                    logger.warning(
                        "MRMR.fit count_encoding FE raised %s: %s; "
                        "continuing without count-encoded columns.",
                        type(_cnt_exc).__name__, _cnt_exc,
                    )

            # ----- Frequency encoding ------------------------------------
            if bool(getattr(self, "fe_frequency_encoding_enable", False)):
                try:
                    _freq_cfg = tuple(
                        getattr(self, "fe_frequency_encoding_cols", ()) or ()
                    )
                    if _freq_cfg:
                        _freq_cols = [
                            c for c in _freq_cfg
                            if c in X.columns and c not in _engineered_seen_l34
                        ]
                    else:
                        _freq_cols = auto_detect_te_cols(
                            X, min_card=5, max_card=500,
                        )
                    _X_before_freq_cols = list(X.columns)
                    X_f, _freq_appended, _freq_recipes = frequency_encode_with_recipes(
                        X, cat_cols=_freq_cols,
                    )
                    _freq_appended = [
                        c for c in _freq_appended if c not in _X_before_freq_cols
                    ]
                    if _freq_appended:
                        X = X_f
                        self.frequency_encoding_features_ = list(_freq_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_freq_appended)
                        )
                        for _r in _freq_recipes:
                            if _r.name in _freq_appended:
                                _freq_enc_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit frequency_encoding: appended %d "
                                "engineered column(s): %s",
                                len(_freq_appended), _freq_appended[:8],
                            )
                except Exception as _freq_exc:
                    logger.warning(
                        "MRMR.fit frequency_encoding FE raised %s: %s; "
                        "continuing without frequency-encoded columns.",
                        type(_freq_exc).__name__, _freq_exc,
                    )

            # ----- Cat x Num interaction (OOF residual) ------------------
            if bool(getattr(self, "fe_cat_num_interaction_enable", False)):
                try:
                    _cn_cats = tuple(
                        getattr(self, "fe_cat_num_interaction_cat_cols", ()) or ()
                    )
                    _cn_nums = tuple(
                        getattr(self, "fe_cat_num_interaction_num_cols", ()) or ()
                    )
                    _cn_cats = [
                        c for c in _cn_cats if c in X.columns
                    ]
                    _cn_nums = [
                        c for c in _cn_nums if c in X.columns
                    ]
                    if _cn_cats and _cn_nums:
                        _y_for_cn = (
                            y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                        )
                        _y_for_cn = np.asarray(_y_for_cn, dtype=np.float64).ravel()
                        _X_before_cn_cols = list(X.columns)
                        X_cn, _cn_appended, _cn_recipes = (
                            cat_num_interaction_with_recipes(
                                X, _y_for_cn,
                                cat_cols=_cn_cats,
                                num_cols=_cn_nums,
                                n_folds=int(
                                    getattr(self, "fe_cat_num_interaction_folds", 5)
                                ),
                                smoothing=float(
                                    getattr(self, "fe_cat_num_interaction_smoothing", 10.0)
                                ),
                                random_state=int(
                                    getattr(self, "random_seed", 0) or 0
                                ),
                            )
                        )
                        _cn_appended = [
                            c for c in _cn_appended if c not in _X_before_cn_cols
                        ]
                        if _cn_appended:
                            X = X_cn
                            self.cat_num_interaction_features_ = list(_cn_appended)
                            self.hybrid_orth_features_ = (
                                list(self.hybrid_orth_features_ or []) + list(_cn_appended)
                            )
                            for _r in _cn_recipes:
                                if _r.name in _cn_appended:
                                    _cat_num_pre_recipes[_r.name] = _r
                            if verbose:
                                logger.info(
                                    "MRMR.fit cat_num_interaction: appended %d "
                                    "engineered column(s): %s",
                                    len(_cn_appended), _cn_appended[:8],
                                )
                except Exception as _cn_exc:
                    logger.warning(
                        "MRMR.fit cat_num_interaction FE raised %s: %s; "
                        "continuing without cat x num residual columns.",
                        type(_cn_exc).__name__, _cn_exc,
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
        _is_pandas_l37 = isinstance(X, pd.DataFrame)
        if not _is_pandas_l37:
            warnings.warn(
                "MRMR: Layer 37 FE (missingness indicator/count/pattern) enabled "
                "but X is not a pandas DataFrame; the missingness encodings are "
                "skipped. Convert to pandas via X.to_pandas() before fit() to "
                "apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            from ._missingness_fe import (
                auto_detect_missing_cols,
                missing_indicator_with_recipes,
                missingness_count_with_recipes,
                missingness_pattern_with_recipes,
            )

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
                    return [
                        c for c in _cfg
                        if c in X.columns and c not in _engineered_seen_l37
                    ]
                # Auto-detect candidate cols with NaN rate in [1%, 99%].
                return [
                    c for c in auto_detect_missing_cols(X)
                    if c not in _engineered_seen_l37
                ]

            # ----- Per-column indicator ------------------------------------
            if bool(getattr(self, "fe_missingness_indicator_enable", False)):
                try:
                    _ind_cols = _resolve_missing_cols(
                        getattr(self, "fe_missingness_indicator_cols", ())
                    )
                    _X_before_ind_cols = list(X.columns)
                    X_i, _ind_appended, _ind_recipes = missing_indicator_with_recipes(
                        X, cols=_ind_cols,
                    )
                    _ind_appended = [
                        c for c in _ind_appended if c not in _X_before_ind_cols
                    ]
                    if _ind_appended:
                        X = X_i
                        self.missingness_indicator_features_ = list(_ind_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_ind_appended)
                        )
                        for _r in _ind_recipes:
                            if _r.name in _ind_appended:
                                _miss_ind_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_indicator: appended %d "
                                "engineered column(s): %s",
                                len(_ind_appended), _ind_appended[:8],
                            )
                except Exception as _ind_exc:
                    logger.warning(
                        "MRMR.fit missingness_indicator FE raised %s: %s; "
                        "continuing without missingness indicator columns.",
                        type(_ind_exc).__name__, _ind_exc,
                    )

            # ----- Per-row missingness count -------------------------------
            if bool(getattr(self, "fe_missingness_count_enable", False)):
                try:
                    _cnt_cols = _resolve_missing_cols(
                        getattr(self, "fe_missingness_indicator_cols", ())
                    )
                    _X_before_mc_cols = list(X.columns)
                    X_c, _mc_appended, _mc_recipes = missingness_count_with_recipes(
                        X, cols=_cnt_cols,
                    )
                    _mc_appended = [
                        c for c in _mc_appended if c not in _X_before_mc_cols
                    ]
                    if _mc_appended:
                        X = X_c
                        self.missingness_count_features_ = list(_mc_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_mc_appended)
                        )
                        for _r in _mc_recipes:
                            if _r.name in _mc_appended:
                                _miss_cnt_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_count: appended %d "
                                "engineered column(s): %s",
                                len(_mc_appended), _mc_appended[:8],
                            )
                except Exception as _mc_exc:
                    logger.warning(
                        "MRMR.fit missingness_count FE raised %s: %s; "
                        "continuing without missingness count column.",
                        type(_mc_exc).__name__, _mc_exc,
                    )

            # ----- Per-row top-K pattern -----------------------------------
            if bool(getattr(self, "fe_missingness_pattern_enable", False)):
                try:
                    _pat_cols = _resolve_missing_cols(
                        getattr(self, "fe_missingness_indicator_cols", ())
                    )
                    _top_k = int(getattr(self, "fe_missingness_pattern_top_k", 5))
                    _X_before_pat_cols = list(X.columns)
                    X_p, _pat_appended, _pat_recipes = missingness_pattern_with_recipes(
                        X, cols=_pat_cols, top_k=_top_k,
                    )
                    _pat_appended = [
                        c for c in _pat_appended if c not in _X_before_pat_cols
                    ]
                    if _pat_appended:
                        X = X_p
                        self.missingness_pattern_features_ = list(_pat_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_pat_appended)
                        )
                        for _r in _pat_recipes:
                            if _r.name in _pat_appended:
                                _miss_pat_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit missingness_pattern: appended %d "
                                "engineered column(s): %s",
                                len(_pat_appended), _pat_appended[:8],
                            )
                except Exception as _pat_exc:
                    logger.warning(
                        "MRMR.fit missingness_pattern FE raised %s: %s; "
                        "continuing without missingness pattern column.",
                        type(_pat_exc).__name__, _pat_exc,
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
    _eng_cols_appended = list(self.hybrid_orth_features_ or []) + list(
        self.mi_greedy_features_ or []
    )
    if len(_eng_cols_appended) >= 2 and isinstance(X, pd.DataFrame):
        _eng_keep: list[str] = []
        _eng_drop: set[str] = set()
        _eng_arrs: dict[str, np.ndarray] = {}
        for _c in _eng_cols_appended:
            if _c in _eng_drop:
                continue
            _arr_c = np.asarray(X[_c].to_numpy(), dtype=np.float64)
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
            _ranks_c = pd.Series(_arr_c).rank(method="average").to_numpy()
            _is_dup = False
            for _kept in _eng_keep:
                _arr_k = _eng_arrs[_kept]
                _mask = _fin_c & np.isfinite(_arr_k)
                if _mask.sum() < 8:
                    continue
                _a, _b = _arr_c[_mask], _arr_k[_mask]
                if _a.std() <= 1e-12 or _b.std() <= 1e-12:
                    continue
                _ranks_a = pd.Series(_a).rank(method="average").to_numpy()
                _ranks_b = pd.Series(_b).rank(method="average").to_numpy()
                if _ranks_a.std() <= 1e-12 or _ranks_b.std() <= 1e-12:
                    continue
                _rank_corr = abs(float(np.corrcoef(_ranks_a, _ranks_b)[0, 1]))
                if np.isfinite(_rank_corr) and _rank_corr >= 0.99:
                    _is_dup = True
                    break
            if _is_dup:
                _eng_drop.add(_c)
            else:
                _eng_keep.append(_c)
                _eng_arrs[_c] = _arr_c
        if _eng_drop:
            X = X.drop(columns=list(_eng_drop))
            self.hybrid_orth_features_ = [
                c for c in (self.hybrid_orth_features_ or []) if c not in _eng_drop
            ]
            self.mi_greedy_features_ = [
                c for c in (self.mi_greedy_features_ or []) if c not in _eng_drop
            ]
            # Layer 33: mirror the same cleanup for TE-encoded columns.
            self.kfold_te_features_ = [
                c for c in (getattr(self, "kfold_te_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 34: mirror cleanup for count / freq / cat_num residual.
            self.count_encoding_features_ = [
                c for c in (getattr(self, "count_encoding_features_", []) or [])
                if c not in _eng_drop
            ]
            self.frequency_encoding_features_ = [
                c for c in (getattr(self, "frequency_encoding_features_", []) or [])
                if c not in _eng_drop
            ]
            self.cat_num_interaction_features_ = [
                c for c in (getattr(self, "cat_num_interaction_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 37: mirror cleanup for missingness indicator / count / pattern.
            self.missingness_indicator_features_ = [
                c for c in (getattr(self, "missingness_indicator_features_", []) or [])
                if c not in _eng_drop
            ]
            self.missingness_count_features_ = [
                c for c in (getattr(self, "missingness_count_features_", []) or [])
                if c not in _eng_drop
            ]
            self.missingness_pattern_features_ = [
                c for c in (getattr(self, "missingness_pattern_features_", []) or [])
                if c not in _eng_drop
            ]
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
            if verbose:
                logger.info(
                    "MRMR.fit engineered-FE dedup: pruned %d near-duplicate "
                    "engineered column(s) at Spearman |rho| >= 0.99: %s",
                    len(_eng_drop), sorted(_eng_drop),
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
    self.feature_names_in_ = [c for c in _all_cols if c not in _engineered_names_set]
    self.n_features_in_ = len(self.feature_names_in_)

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
        missing_strategy=_strategy_for_categorize,
        nbins_strategy=_nbins_strategy,
        nbins_strategy_kwargs=_nbins_strategy_kwargs,
        y_for_strategy=_y_for_strategy,
    )
    logger.info("categorized.")

    target_indices = np.array([cols.index(col) for col in target_names], dtype=np.int64)

    # ---------------------------------------------------------------------------------------------------------------
    # Core
    # ---------------------------------------------------------------------------------------------------------------

    if _is_polars_input:
        # Polars schema-driven detection; mirrors categorize_dataset's _is_pl_cat.
        import polars as _pl
        _CAT_DTYPES_FOR_VARS = {_pl.Utf8, _pl.String, _pl.Categorical, _pl.Boolean}
        categorical_vars_names = [
            name for name, dt in X.schema.items()
            if dt in _CAT_DTYPES_FOR_VARS
            or (hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum))
        ]
    else:
        categorical_vars_names = X.head().select_dtypes(include=("category", "object", "string", "bool")).columns.values.tolist()
    categorical_vars = [cols.index(col) for col in categorical_vars_names]

    if fe_max_steps > 0:
        unary_transformations = create_unary_transformations(preset=fe_unary_preset)
        binary_transformations = create_binary_transformations(preset=fe_binary_preset)
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
        from .cat_fe_state import CatFEConfig as _CatFEConfig
        cat_fe_cfg = _CatFEConfig()
    if cat_fe_cfg.enable and len(categorical_vars) >= 2:
        from .cat_interactions import run_cat_interaction_step
        from .info_theory import merge_vars as _merge_vars_for_cat_fe

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
            dtype=dtype, verbose=verbose,
        )
        self._cat_fe_state_ = cat_fe_state
        # Register engineered cat features as categorical_vars so the downstream numeric-FE step excludes them
        # from numeric_vars_to_consider; without this, k-way cat engineered cols enter prospective_pairs and
        # check_prospective_fe_pairs hits KeyError reading them from X (which lacks engineered cols).
        # Engineered cat cols are appended at the end of data/cols at positions [_n_cols_before_cat_fe..].
        _n_cat_fe_added = data.shape[1] - _n_cols_before_cat_fe
        if _n_cat_fe_added > 0:
            categorical_vars = list(categorical_vars) + list(
                range(_n_cols_before_cat_fe, data.shape[1])
            )
        # Persist cache for next fit() call
        if cat_fe_state.streaming_cache_out:
            self._cat_fe_cache_ = cat_fe_state.streaming_cache_out
        # Cat-FE recipes feed the same engineered_recipes dict numeric FE uses; the fit-end splitter copies
        # any recipe whose engineered name appears in selected_vars_names into ``self._engineered_recipes_``.
        for r in cat_fe_state.recipes:
            engineered_recipes[r.name] = r
        if verbose and cat_fe_state.recipes:
            logger.info(
                "MRMR cat-FE produced %d engineered feature(s); "
                "data extended from %d to %d cols.",
                len(cat_fe_state.recipes),
                data.shape[1] - len(cat_fe_state.recipes),
                data.shape[1],
            )

    # Resolve effective ``min_relevance_gain`` against the target entropy. ``'relative_to_entropy'`` mode uses ``min_relevance_gain_frac * H(y)`` so the stop floor scales with how much information the target actually carries; ``'absolute'`` mode retains the legacy verbatim value. The target is already discretized into bins (``data[:, target_indices[0]]`` with bin count ``nbins[target_indices[0]]``); ``np.bincount`` + Shannon entropy in nats matches the screen_predictors estimator family.
    if self.min_relevance_gain_mode not in ("absolute", "relative_to_entropy"):
        raise ValueError(
            f"MRMR.min_relevance_gain_mode={self.min_relevance_gain_mode!r} must be 'absolute' or 'relative_to_entropy'."
        )
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
    while True:
        n_recommended_features = 0
        times_spent = defaultdict(float)
        selected_vars, predictors, any_influencing, entropy_cache, cached_MIs, cached_confident_MIs, cached_cond_MIs, classes_y, classes_y_safe, freqs_y, _dcd_state = (
            screen_predictors(
                factors_data=data,
                y=target_indices,
                factors_nbins=nbins,
                factors_names=cols,
                # Layer 23: when hybrid orth FE appended columns, extend the
                # candidate pool to include them so they reach the screening
                # gates. When the caller did not pin factors_names_to_use,
                # screen_predictors uses every column from ``cols`` so the
                # hybrid cols are naturally included.
                factors_names_to_use=(
                    list(self.factors_names_to_use)
                    + list(self.hybrid_orth_features_ or [])
                    + list(getattr(self, "mi_greedy_features_", None) or [])
                    if (
                        self.factors_names_to_use
                        and (
                            self.hybrid_orth_features_
                            or getattr(self, "mi_greedy_features_", None)
                        )
                    )
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
                    self.max_confirmation_cand_nbins
                    if self.max_confirmation_cand_nbins is not None
                    else self.quantization_nbins ** self.interactions_max_order * 2
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
                engineered_lineage=(
                    self._cat_fe_state_.lineage
                    if getattr(self, "_cat_fe_state_", None) is not None
                    and self._cat_fe_state_.lineage
                    else None
                ),
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
                        X_raw=X,
                        quantization_method=self.quantization_method,
                        quantization_nbins=self.quantization_nbins,
                        quantization_dtype=self.quantization_dtype,
                    )
                    if getattr(self, "dcd_enable", False) else None
                ),
            )
        )
        # 2026-05-30 Wave 9 — stash DCD summary on the estimator for the
        # public ``dcd_`` attribute (None when DCD was disabled).
        try:
            from ._dynamic_cluster_discovery import dcd_summary as _dcd_summary
            self.dcd_ = _dcd_summary(_dcd_state)
        except Exception:
            self.dcd_ = None
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
            except Exception:
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

        # Feature engineering iteration delegated to ``_run_fe_step`` (testable / experiment-friendly outside
        # the screening loop). Returns updated state + n_recommended_features; zero breaks the outer loop.
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
        if (
            n_recommended_features == 0
            and _adaptive
            and fe_max_steps > 0
            and num_fs_steps == 0   # only on the very first FE step
        ):
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
            break  # uncomment to avoid recheck of single-rounded FE

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
        X.drop(columns=target_names, inplace=True)  # restores caller's original schema

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
    if getattr(self, "build_friend_graph", False) and len(selected_vars) > 0:
        try:
            from .friend_graph import build_friend_graph as _build_fg, prune_by_friend_graph as _prune_fg

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
            )
            if self.friend_graph_prune:
                # Protect cluster-aggregate columns from pruning: they are correlated with all their
                # members by construction, so the sink classifier could mis-flag them.
                _ca_protect = [
                    v for v in selected_vars
                    if getattr(engineered_recipes.get(cols[v]), "kind", None) == "cluster_aggregate"
                ]
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

    # ---------------------------------------------------------------------------------------------------------------
    # selected_vars: cols-indices -> names -> original-frame indices (categorize_dataset may rearrange cat columns).
    # ---------------------------------------------------------------------------------------------------------------

    selected_vars_names = np.array(cols)[np.array(selected_vars, dtype=np.intp)]
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
            _sel_names = set(X.columns[selected_vars].tolist())
            temp_columns = [c for c in X.columns if c not in _sel_names]

            if _is_classification:
                cb_num_rfecv = RFECV(
                    estimator=CatBoostClassifier(**configs.CB_CLASSIF),
                    fit_params=dict(plot=False),
                    cat_features=categorical_vars_names,
                    scoring=make_scorer(
                        score_func=compute_probabilistic_multiclass_error, response_method='predict_proba', greater_is_better=False
                    ),
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

    self.support_ = np.array(selected_vars)
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
    self.fallback_used_ = False
    # n_features_ reports the column count produced by transform() = raw selected + engineered (replayable via _engineered_recipes_). Higher-order
    # engineered features without a replayable recipe were already warned about above and are NOT counted (they don't appear in transform output).
    n_engineered_out = len(self._engineered_recipes_)
    if selected_vars or n_engineered_out:
        self.n_features_ = len(selected_vars) + n_engineered_out
    else:
        self.n_features_ = 0
        # Empty support_ fallback: rank by raw MI(X_j, y) so downstream pipelines don't crash on 0-feature
        # transform output. Only triggers when min_features_fallback >= 1 (off by default).
        _min_fb = int(getattr(self, "min_features_fallback", 0) or 0)
        # 2026-05-30 Wave 9.1 fix (loop iter 39): hoist the
        # ``warnings.warn`` OUT of the try block. Pre-fix the warning
        # was inside ``try:`` and the surrounding ``except Exception``
        # caught it under ``simplefilter('error', UserWarning)`` -
        # making the user-facing warning indistinguishable from a real
        # fallback failure (and silently dropping it). Now the
        # try/except scopes only the MI computation; the warning fires
        # afterwards on the successful path.
        _fallback_msg = None
        if _min_fb >= 1 and self.n_features_in_ > 0:
            try:
                # Rank by cached confident MI with the target; take top-K. cached_MIs may not be populated;
                # re-compute from the original frame as a last resort.
                _raw_mi = []
                for _i in range(self.n_features_in_):
                    # ``cached_MIs`` is keyed by the candidate variable-index
                    # tuple (see evaluation.py: ``cached_MIs[X] = direct_gain``,
                    # where ``X`` is the tuple of variable indices). For
                    # single-variable candidates the key is ``(_i,)``. The prior
                    # ``(_i, -1)`` lookup never matched any key, so every
                    # entry resolved to 0.0 and the fallback's top-K ranking
                    # was a no-op (picked index 0 regardless of signal).
                    _key = (_i,)
                    _mi = self.cached_MIs.get(_key, 0.0) if hasattr(self, "cached_MIs") else 0.0
                    _raw_mi.append((_i, float(_mi)))
                # Sort by MI desc; pick top-K.
                # Wave 57 (2026-05-20): secondary key on feature index so
                # tied MI doesn't make the empty-support fallback drift.
                _raw_mi.sort(key=lambda kv: (-kv[1], kv[0]))
                _topk = [i for i, _ in _raw_mi[:_min_fb]]
                if _topk:
                    self.support_ = np.array(_topk)
                    self.n_features_ = len(_topk)
                    self.fallback_used_ = True
                    _top_mi = float(_raw_mi[0][1]) if _raw_mi else 0.0
                    _uninformative = _top_mi <= 0.0
                    _fallback_msg = (
                        f"MRMR: screening returned 0 features; falling "
                        f"back to top-{self.n_features_} by raw "
                        f"MI(X_j, y). Set min_features_fallback=0 to "
                        f"disable. fallback_used_=True is set on the "
                        f"estimator."
                    )
                    if _uninformative:
                        _fallback_msg = (
                            f"{_fallback_msg} All candidates have MI <= 0 "
                            f"(e.g. constant X columns or empty "
                            f"cached_MIs); the returned support_ carries "
                            f"NO signal."
                        )
            except Exception as _exc:
                logger.warning(
                    "MRMR fallback to top-K MI failed: %s. Returning empty support_.",
                    _exc,
                )
        if _fallback_msg is not None:
            # logger.warning for log-grepping back-compat AND
            # warnings.warn so simplefilter('error', UserWarning) / test
            # suites can intercept programmatically.
            logger.warning(_fallback_msg)
            import warnings as _w_iter39
            _w_iter39.warn(_fallback_msg, UserWarning, stacklevel=2)

    # ---------------------------------------------------------------------------------------------------------------
    # Report FS results
    # ---------------------------------------------------------------------------------------------------------------

    if verbose:
        predictors_str = ", ".join([f"{el['name']}: {el['gain']:.4f}" for el in predictors[:50]])
        predictors_str = textwrap.shorten(predictors_str, width=300)
        logger.info("MRMR+ selected %d out of %d features: %s", self.n_features_, self.n_features_in_, predictors_str)

    self.signature = signature
    self.ran_out_of_time_ = ran_out_of_time

    # Store self in process-wide cache so cloned MRMR instances fit on the same (X, y) arrays can replay
    # this fitted state instead of re-running cat-FE + permutation. Bound the LRU by ``fit_cache_max``;
    # the default (4) covers a typical model suite without thrashing and long-lived workers no longer leak.
    if _cache_key is not None:
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
        # Byte-size cap on top of entry count (audit A5 P1 #6): a 1k-feature suite
        # carrying 4 cached MRMR instances each holding _selectors_ / _engineered_features_
        # state can exceed 1 GB of process RSS. ``fit_cache_max_mb`` (default 1024 MB; env
        # override ``MLFRAME_MRMR_FIT_CACHE_MAX_MB``) bounds the aggregate cache footprint.
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
            _byte_cap = _mb_cap * (1024 ** 2)
            while len(MRMR._FIT_CACHE) > 1 and _mrmr_cache_bytes_total() > _byte_cap:
                MRMR._FIT_CACHE.popitem(last=False)
    # 2026-05-30 Wave 8 — post-fit UAED auto-size. When enabled, replaces the
    # configured ``min_features_fallback`` floor with an automatic elbow on
    # the per-feature MI gain curve. Relevance trace is taken from the
    # ``mrmr_gains_`` attribute (Wave-7 audit landed this trace in the
    # standard fit output); if missing, this step no-ops.
    if getattr(self, "uaed_auto_size", False):
        try:
            from ._cmi_perm_stop import uaed_elbow
            gains = np.asarray(getattr(self, "mrmr_gains_", []), dtype=np.float64)
            if gains.size >= 3:
                elbow = int(uaed_elbow(gains))
                if 0 < elbow < gains.size and hasattr(self, "support_"):
                    self.support_ = np.asarray(self.support_)[:elbow + 1]
                    self.n_features_ = int(self.support_.size)
                    self.uaed_elbow_ = int(elbow)
        except Exception:
            # UAED is best-effort post-fit; don't break fit() on internal hiccup.
            pass
    return self
