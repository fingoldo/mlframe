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


# 2026-06-01 Layer 85 — default-scorer dispatcher for the Layer 21 hybrid
# orth-poly univariate basis-selection stage. Imports the alternate scorer
# module lazily so callers paying for "plug_in" (the default) do not pay
# the import cost. Every alternate scorer's ``*_with_recipes`` returns the
# same ``(X_aug, scores_df, recipes_list)`` 3-tuple as Layer 21's plain
# univariate path -- so the caller plumbing in ``_fit_impl`` is unchanged.
def _dispatch_default_scorer(
    scorer: str,
    *,
    X: pd.DataFrame,
    y: np.ndarray,
    cols,
    degrees,
    basis: str,
    top_k: int,
):
    """Route a non-``plug_in`` ``fe_hybrid_orth_default_scorer`` value to
    the matching ``*_with_recipes`` univariate-stage builder.

    Returns the same ``(X_aug, scores_df, recipes_list)`` tuple every
    alternate scorer's ``with_recipes`` variant exposes. Raises
    ``ValueError`` on an unrecognised scorer string (defensive: the public
    validation entry point catches this in ``_validate_string_params``).
    """
    if scorer == "cmim":
        from ._orthogonal_cmim_fe import (
            hybrid_orth_mi_cmim_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "jmim":
        from ._orthogonal_jmim_fe import (
            hybrid_orth_mi_jmim_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "tc":
        from ._orthogonal_total_correlation_fe import (
            hybrid_orth_mi_tc_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "ksg":
        from ._orthogonal_ksg_mi_fe import (
            hybrid_orth_mi_ksg_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "copula":
        from ._orthogonal_copula_mi_fe import (
            hybrid_orth_mi_copula_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "dcor":
        from ._orthogonal_dcor_fe import (
            hybrid_orth_mi_dcor_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "hsic":
        from ._orthogonal_hsic_fe import (
            hybrid_orth_mi_hsic_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "lasso":
        from ._orthogonal_lasso_fe import (
            hybrid_orth_mi_lasso_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "elasticnet":
        from ._orthogonal_elasticnet_fe import (
            hybrid_orth_mi_elasticnet_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "auto":
        from ._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_auto_scorer_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "ensemble":
        from ._orthogonal_scorer_auto_fe import (
            hybrid_orth_mi_ensemble_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "meta":
        from ._orthogonal_meta_scorer_fe import (
            hybrid_orth_mi_meta_fe_with_recipes as _fn,
        )
        return _fn(X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k)
    if scorer == "auto_oracle":
        # 2026-06-01 Layer 100 — UNIFIED scorer-selection. The
        # OracleScorerSelector resolves a concrete scorer for this
        # dataset's fingerprint (learned-best when the oracle has confident
        # history, else the L76 cold-start cascade), then we delegate to
        # THAT scorer's univariate builder. The bake-off that populates the
        # oracle (``benchmark_all_scorers``) runs out-of-band, so fit-time
        # cost is one recommend + one scorer run, not the full L68 sweep.
        from ._oracle_scorer_select import OracleScorerSelector
        _selector = OracleScorerSelector()
        _resolved = _selector.recommend_scorer(X, y)
        # ``auto_oracle`` never resolves to itself / "plug_in"-by-default is
        # routed through the plug_in builders below; any other resolved
        # scorer recurses into its own branch of this dispatcher.
        if _resolved == "plug_in":
            from ._orthogonal_univariate_fe import (
                hybrid_orth_mi_fe_with_recipes as _fn,
            )
            return _fn(
                X, y, cols=cols, degrees=degrees, basis=basis, top_k=top_k,
            )
        return _dispatch_default_scorer(
            _resolved, X=X, y=y, cols=cols, degrees=degrees,
            basis=basis, top_k=top_k,
        )
    raise ValueError(
        f"_dispatch_default_scorer: unrecognised scorer={scorer!r}. "
        f"This is a defensive bug -- ``_validate_string_params`` should "
        f"have caught the bad value before fit reached the dispatcher."
    )


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
    # ADAPTIVE-FREQUENCY Fourier (2026-06-03): names of the held-out-validated
    # adaptive sin/cos columns the extra-basis stage emitted. Used by the
    # support-finalisation ADAPTIVE-PROTECTION block to re-add any the MRMR
    # screen dropped. Always present (empty when no adaptive freq detected) so
    # transform / pickle / clone never trip on a missing attribute.
    self._adaptive_fourier_features_ = []
    _hybrid_orth_pre_recipes: dict = {}
    # Snapshot the raw input columns BEFORE any FE stage appends engineered
    # intermediates. The cat_pair / cat_triple auto-detect paths restrict their
    # candidate members to this set so a cross is never built on an engineered
    # column (which cannot be replayed at transform time -> KeyError).
    _raw_input_cols_pre_fe = (
        list(X.columns) if isinstance(X, pd.DataFrame) else []
    )
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
    if _hybrid_on or _univ_basis_on:
        # Polars frames: skip with a warning -- hybrid FE pipeline operates on
        # pandas. Native polars support would require a separate code path;
        # not in Layer 23 MVP scope.
        _is_pandas_for_hybrid = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_hybrid:
            # The block now opens whenever fe_univariate_basis_enable (default
            # True) OR fe_hybrid_orth_enable is on, so the message must not
            # blame fe_hybrid_orth_enable -- on the default polars path the user
            # never set it. Name the actual triggering flag(s).
            warnings.warn(
                "MRMR: univariate/hybrid orthogonal-polynomial FE is enabled "
                "(fe_univariate_basis_enable / fe_hybrid_orth_enable) but X is "
                "not a pandas DataFrame; the FE stage is skipped. Convert via "
                "X.to_pandas() before fit() if you want it applied.",
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
                    _h_cols = [
                        c for c in self.factors_names_to_use if c in X.columns
                    ]
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
                else:
                    X_h, _uni_sc, _recipes = _dispatch_default_scorer(
                        _default_scorer,
                        X=X,
                        y=_y_for_hybrid,
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
            if _univ_fourier_on and _univ_basis_on and "fourier" not in _eff_extra_bases:
                _eff_extra_bases = _eff_extra_bases + ("fourier",)
            if _is_pandas_for_hybrid and _eff_extra_bases:
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
                    _fourier_powers = tuple(
                        int(p) for p in
                        getattr(self, "fe_hybrid_orth_fourier_powers", (1, 2))
                    )
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
                        _e_cols = [
                            c for c in self.factors_names_to_use
                            if c in X.columns and c not in _already_eng_for_extra
                        ]
                    else:
                        _e_cols = [
                            c for c in X.columns if c not in _already_eng_for_extra
                        ]
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
                    _fourier_adaptive = bool(
                        getattr(self, "fe_univariate_fourier_adaptive", True)
                    )
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
                    _fourier_chirp = bool(
                        getattr(self, "fe_univariate_fourier_chirp", True)
                    )
                    _fourier_chirp_mvc = float(
                        getattr(
                            self,
                            "fe_univariate_fourier_chirp_min_val_corr",
                            0.15,
                        )
                    )
                    X_e, _e_scores, _e_recipes = hybrid_orth_extra_basis_fe_with_recipes(
                        X, _y_for_extra,
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
                            _prev_adaptive = list(
                                getattr(self, "_adaptive_fourier_features_", None) or []
                            )
                            self._adaptive_fourier_features_ = (
                                _prev_adaptive + _adaptive_names
                            )
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
    # 2026-05-31 Layer 56 — TRI-PRODUCT cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable): captures
    # genuine 3-way interactions like 3-way XOR and price*quantity*count
    # that the pair stage cannot. O(seed_k^3 * deg^3) candidate count is
    # bounded by seed_k=4 default. Recipes (``orth_triplet_cross``) replay
    # from X only, no y, leakage-free by construction.
    if bool(getattr(self, "fe_hybrid_orth_triplet_enable", False)):
        _is_pandas_for_triplet = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_triplet:
            warnings.warn(
                "MRMR: fe_hybrid_orth_triplet_enable=True but X is not a pandas "
                "DataFrame; triplet cross-basis FE is skipped. Convert to "
                "pandas via X.to_pandas() before fit() if you want triplet "
                "FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_triplet_fe import (
                    hybrid_orth_mi_triplet_fe_with_recipes,
                )
                _y_for_triplet = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                _t_cols: list | None = None
                if getattr(self, "factors_names_to_use", None):
                    _t_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _t_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _t_max_degree = int(
                    getattr(self, "fe_hybrid_orth_triplet_max_degree", 1)
                )
                _t_seed_k = int(
                    getattr(self, "fe_hybrid_orth_triplet_seed_k", 4)
                )
                _t_top_count = int(
                    getattr(self, "fe_hybrid_orth_triplet_top_count", 2)
                )
                _t_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
                _t_degrees = tuple(
                    int(d) for d in getattr(self, "fe_hybrid_orth_degrees", (2, 3))
                )
                _t_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
                _X_before_triplet_cols = list(X.columns)
                X_t, _t_uni_sc, _t_triplet_sc, _t_recipes = (
                    hybrid_orth_mi_triplet_fe_with_recipes(
                        X, _y_for_triplet,
                        cols=_t_cols,
                        degrees=_t_degrees,
                        basis=_t_basis,
                        top_k=_t_top_k,
                        triplet_max_degree=_t_max_degree,
                        top_triplet_seed_k=_t_seed_k,
                        top_triplet_count=_t_top_count,
                    )
                )
                _t_appended = [
                    c for c in X_t.columns if c not in _X_before_triplet_cols
                ]
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
                    X = pd.concat([X, X_t[_t_triplet_only]], axis=1)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_t_triplet_only)
                    )
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
                            "MRMR.fit hybrid_orth triplet: appended %d "
                            "engineered column(s): %s",
                            len(_t_triplet_only), _t_triplet_only[:8],
                        )
            except Exception as _t_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth triplet FE raised %s: %s; "
                    "continuing without triplet-FE columns.",
                    type(_t_exc).__name__, _t_exc,
                )
    # 2026-06-01 Layer 77 — QUADRUPLET (4-way) cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable): captures
    # genuine 4-way interactions like 4-way XOR (every triplet marginal MI
    # is zero by symmetry, only the He_1^4 cell carries signal) and
    # revenue = price*qty*count*discount. O(seed_k^4 * deg^4) candidate
    # count is bounded by seed_k=4 default. Recipes
    # (``orth_quadruplet_cross``) replay from X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_quadruplet_enable", False)):
        _is_pandas_for_quad = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_quad:
            warnings.warn(
                "MRMR: fe_hybrid_orth_quadruplet_enable=True but X is not a "
                "pandas DataFrame; quadruplet cross-basis FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want quadruplet FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_quadruplet_fe import (
                    hybrid_orth_mi_quadruplet_fe_with_recipes,
                )
                _y_for_quad = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                _q_cols: list | None = None
                if getattr(self, "factors_names_to_use", None):
                    _q_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _q_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _q_max_degree = int(
                    getattr(self, "fe_hybrid_orth_quadruplet_max_degree", 1)
                )
                _q_seed_k = int(
                    getattr(self, "fe_hybrid_orth_quadruplet_seed_k", 4)
                )
                _q_top_count = int(
                    getattr(self, "fe_hybrid_orth_quadruplet_top_count", 2)
                )
                _q_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
                _q_degrees = tuple(
                    int(d) for d in getattr(self, "fe_hybrid_orth_degrees", (2, 3))
                )
                _q_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
                _X_before_quad_cols = list(X.columns)
                X_q, _q_uni_sc, _q_quad_sc, _q_recipes = (
                    hybrid_orth_mi_quadruplet_fe_with_recipes(
                        X, _y_for_quad,
                        cols=_q_cols,
                        degrees=_q_degrees,
                        basis=_q_basis,
                        top_k=_q_top_k,
                        quadruplet_max_degree=_q_max_degree,
                        top_quadruplet_seed_k=_q_seed_k,
                        top_quadruplet_count=_q_top_count,
                    )
                )
                _q_appended = [
                    c for c in X_q.columns if c not in _X_before_quad_cols
                ]
                # Only keep TRUE quadruplet columns (4 legs joined by '*');
                # the wrapper may also pass univariate winners through which
                # the master hybrid stage already handles when enabled.
                _q_quad_only = [
                    c for c in _q_appended if c.split("__", 1)[0].count("*") == 3
                ]
                if _q_quad_only:
                    X = pd.concat([X, X_q[_q_quad_only]], axis=1)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_q_quad_only)
                    )
                    _kept = set(_q_quad_only)
                    for _r in _q_recipes:
                        if _r.name in _kept:
                            _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth quadruplet: appended %d "
                            "engineered column(s): %s",
                            len(_q_quad_only), _q_quad_only[:8],
                        )
            except Exception as _q_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth quadruplet FE raised %s: %s; "
                    "continuing without quadruplet-FE columns.",
                    type(_q_exc).__name__, _q_exc,
                )
    # 2026-06-01 Layer 78 — ADAPTIVE-ARITY cross-basis FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, the stage enumerates arity 2..max_arity per seed tuple and
    # keeps ONLY the winning arity per maximal signal set (a higher arity
    # is emitted iff its MI strictly beats every lower-arity prefix).
    # Recipes route to the per-arity Layer 22 / 56 / 77 builders.
    if bool(getattr(self, "fe_hybrid_orth_adaptive_arity_enable", False)):
        _is_pandas_for_aa = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_aa:
            warnings.warn(
                "MRMR: fe_hybrid_orth_adaptive_arity_enable=True but X is "
                "not a pandas DataFrame; adaptive-arity FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want adaptive-arity FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_adaptive_arity_fe import (
                    hybrid_orth_mi_adaptive_arity_fe_with_recipes,
                )
                _y_for_aa = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                _aa_cols: list | None = None
                if getattr(self, "factors_names_to_use", None):
                    _aa_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _aa_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _aa_max_arity = int(
                    getattr(self, "fe_hybrid_orth_adaptive_arity_max_arity", 3)
                )
                _aa_max_degree = int(
                    getattr(self, "fe_hybrid_orth_adaptive_arity_max_degree", 1)
                )
                _aa_seed_k = int(
                    getattr(self, "fe_hybrid_orth_adaptive_arity_seed_k", 4)
                )
                _aa_top_count = int(
                    getattr(self, "fe_hybrid_orth_adaptive_arity_top_count", 3)
                )
                _aa_basis = str(getattr(self, "fe_hybrid_orth_basis", "auto"))
                _aa_degrees = tuple(
                    int(d) for d in getattr(self, "fe_hybrid_orth_degrees", (2, 3))
                )
                _aa_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
                _X_before_aa_cols = list(X.columns)
                X_aa, _aa_uni_sc, _aa_adapt_sc, _aa_recipes = (
                    hybrid_orth_mi_adaptive_arity_fe_with_recipes(
                        X, _y_for_aa,
                        cols=_aa_cols,
                        degrees=_aa_degrees,
                        basis=_aa_basis,
                        top_k=_aa_top_k,
                        seed_k=_aa_seed_k,
                        max_arity=_aa_max_arity,
                        max_degree=_aa_max_degree,
                        top_count=_aa_top_count,
                    )
                )
                _aa_appended = [
                    c for c in X_aa.columns if c not in _X_before_aa_cols
                ]
                # Only keep TRUE cross columns (arity >= 2 -- one or more '*').
                _aa_cross_only = [
                    c for c in _aa_appended
                    if c.split("__", 1)[0].count("*") >= 1
                ]
                if _aa_cross_only:
                    X = pd.concat([X, X_aa[_aa_cross_only]], axis=1)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_aa_cross_only)
                    )
                    _kept_aa = set(_aa_cross_only)
                    for _r in _aa_recipes:
                        if _r.name in _kept_aa:
                            _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth adaptive-arity: appended %d "
                            "engineered column(s): %s",
                            len(_aa_cross_only), _aa_cross_only[:8],
                        )
            except Exception as _aa_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth adaptive-arity FE raised %s: %s; "
                    "continuing without adaptive-arity-FE columns.",
                    type(_aa_exc).__name__, _aa_exc,
                )
    # 2026-05-31 Layer 57 — ADAPTIVE PER-COLUMN DEGREE FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, for each source column we evaluate every degree in
    # ``fe_hybrid_orth_adaptive_degree_range`` and emit ONLY the argmax-MI
    # degree (if it clears the per-col uplift gate). Recipe kind reuses
    # ``orth_univariate`` -- replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_adaptive_degree_enable", False)):
        _is_pandas_for_adaptive = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_adaptive:
            warnings.warn(
                "MRMR: fe_hybrid_orth_adaptive_degree_enable=True but X is "
                "not a pandas DataFrame; adaptive-degree FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want adaptive-degree FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_adaptive_degree_fe import (
                    hybrid_orth_mi_adaptive_degree_fe_with_recipes,
                )
                _y_for_adapt = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                _ad_cols: list | None = None
                if getattr(self, "factors_names_to_use", None):
                    _ad_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
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
                X_ad, _ad_scores, _ad_recipes = (
                    hybrid_orth_mi_adaptive_degree_fe_with_recipes(
                        X, _y_for_adapt,
                        cols=_ad_cols,
                        degree_range=_ad_range,
                        basis=_ad_basis,
                        min_uplift=_ad_min_uplift,
                    )
                )
                _ad_appended = [
                    c for c in X_ad.columns if c not in _X_before_adaptive_cols
                ]
                if _ad_appended:
                    X = X_ad
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_ad_appended)
                    )
                    # Merge into the same recipe dict used by the master
                    # hybrid stage so the end-of-fit remap into
                    # ``_engineered_recipes_`` picks it up.
                    for _r in _ad_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth adaptive-degree: appended "
                            "%d engineered column(s): %s",
                            len(_ad_appended), _ad_appended[:8],
                        )
            except Exception as _ad_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth adaptive-degree FE raised %s: %s; "
                    "continuing without adaptive-degree columns.",
                    type(_ad_exc).__name__, _ad_exc,
                )
    # 2026-05-31 Layer 58 — CONDITIONAL BASIS ROUTING FE stage.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, we try every (pre_transform, basis, degree) cell per source
    # column and keep the MI-uplift winner; global top-K appended. Recipe
    # kind reuses ``orth_univariate`` (extra carries ``pre_transform``);
    # replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_conditional_routing_enable", False)):
        _is_pandas_for_routing = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_routing:
            warnings.warn(
                "MRMR: fe_hybrid_orth_conditional_routing_enable=True but X is "
                "not a pandas DataFrame; conditional routing FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want conditional routing FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_routing_fe import (
                    hybrid_orth_mi_conditional_routing_fe_with_recipes,
                )
                _y_for_route = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                _rt_cols: list | None = None
                if getattr(self, "factors_names_to_use", None):
                    _rt_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _rt_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _rt_top_k = int(getattr(
                    self, "fe_hybrid_orth_conditional_routing_top_k", 5,
                ))
                _rt_min_uplift = float(getattr(
                    self, "fe_hybrid_orth_conditional_routing_min_uplift", 1.10,
                ))
                _rt_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_conditional_routing_degrees", (2, 3),
                ))
                _X_before_routing_cols = list(X.columns)
                X_rt, _rt_scores, _rt_recipes = (
                    hybrid_orth_mi_conditional_routing_fe_with_recipes(
                        X, _y_for_route,
                        cols=_rt_cols,
                        degrees=_rt_degrees,
                        top_k=_rt_top_k,
                        min_uplift=_rt_min_uplift,
                    )
                )
                _rt_appended = [
                    c for c in X_rt.columns if c not in _X_before_routing_cols
                ]
                if _rt_appended:
                    X = X_rt
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_rt_appended)
                    )
                    for _r in _rt_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth conditional-routing: appended "
                            "%d engineered column(s): %s",
                            len(_rt_appended), _rt_appended[:8],
                        )
            except Exception as _rt_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth conditional-routing FE raised %s: %s; "
                    "continuing without conditional-routing columns.",
                    type(_rt_exc).__name__, _rt_exc,
                )
    # 2026-05-31 Layer 59 — DIFF-BASIS FE for highly-correlated source pairs.
    # Independent opt-in (does NOT require fe_hybrid_orth_enable). When
    # active, the auto-pair detector flags every pair with |Pearson corr| >=
    # threshold, computes the residual diff, and evaluates a basis expansion
    # per requested degree; top-K winners appended. Recipe kind
    # ``orth_diff_basis``; replay reads X only, no y.
    if bool(getattr(self, "fe_hybrid_orth_diff_basis_enable", False)):
        _is_pandas_for_diff = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_diff:
            warnings.warn(
                "MRMR: fe_hybrid_orth_diff_basis_enable=True but X is not a "
                "pandas DataFrame; diff-basis FE is skipped. Convert to "
                "pandas via X.to_pandas() before fit() if you want diff-basis "
                "FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_diff_basis_fe import (
                    hybrid_orth_mi_diff_basis_fe_with_recipes,
                )
                _y_for_diff = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _df_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _df_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _df_corr = float(getattr(
                    self, "fe_hybrid_orth_diff_basis_corr_threshold", 0.7,
                ))
                _df_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_diff_basis_degrees", (1, 2, 3),
                ))
                _df_top_k = int(getattr(
                    self, "fe_hybrid_orth_diff_basis_top_k", 3,
                ))
                _X_before_diff_cols = list(X.columns)
                X_df, _df_scores, _df_recipes = (
                    hybrid_orth_mi_diff_basis_fe_with_recipes(
                        X, _y_for_diff,
                        cols=_df_cols,
                        degrees=_df_degrees,
                        pair_corr_threshold=_df_corr,
                        top_k=_df_top_k,
                    )
                )
                _df_appended = [
                    c for c in X_df.columns if c not in _X_before_diff_cols
                ]
                if _df_appended:
                    X = X_df
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_df_appended)
                    )
                    for _r in _df_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth diff-basis: appended %d "
                            "engineered column(s): %s",
                            len(_df_appended), _df_appended[:8],
                        )
            except Exception as _df_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth diff-basis FE raised %s: %s; "
                    "continuing without diff-basis columns.",
                    type(_df_exc).__name__, _df_exc,
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
        _is_pandas_for_cb = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_cb:
            warnings.warn(
                "MRMR: fe_hybrid_orth_cluster_basis_enable=True but X is not "
                "a pandas DataFrame; cluster-basis FE is skipped. Convert to "
                "pandas via X.to_pandas() before fit() if you want cluster-"
                "basis FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_cluster_basis_fe import (
                    hybrid_orth_mi_cluster_basis_fe_with_recipes,
                )
                _y_for_cb = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _cb_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _cb_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _cb_aggregator = str(getattr(
                    self, "fe_hybrid_orth_cluster_basis_aggregator", "mean_z",
                ))
                _cb_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_cluster_basis_degrees", (2, 3),
                ))
                _cb_top_k = int(getattr(
                    self, "fe_hybrid_orth_cluster_basis_top_k", 3,
                ))
                # Cluster detection reuses the diff-basis corr threshold as a
                # sensible default (same calibration: 0.7 is the reflection-
                # cluster floor). We deliberately do NOT share the same
                # constructor argument so callers can tune diff-basis and
                # cluster-basis independently.
                _cb_corr = float(getattr(
                    self, "fe_hybrid_orth_diff_basis_corr_threshold", 0.7,
                ))
                _X_before_cb_cols = list(X.columns)
                X_cb, _cb_scores, _cb_recipes = (
                    hybrid_orth_mi_cluster_basis_fe_with_recipes(
                        X, _y_for_cb,
                        cols=_cb_cols,
                        aggregator=_cb_aggregator,
                        degrees=_cb_degrees,
                        corr_threshold=_cb_corr,
                        top_k=_cb_top_k,
                    )
                )
                _cb_appended = [
                    c for c in X_cb.columns if c not in _X_before_cb_cols
                ]
                if _cb_appended:
                    X = X_cb
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_cb_appended)
                    )
                    for _r in _cb_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth cluster-basis: appended %d "
                            "engineered column(s): %s",
                            len(_cb_appended), _cb_appended[:8],
                        )
            except Exception as _cb_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth cluster-basis FE raised %s: %s; "
                    "continuing without cluster-basis columns.",
                    type(_cb_exc).__name__, _cb_exc,
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
        _is_pandas_for_boot = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_boot:
            warnings.warn(
                "MRMR: fe_hybrid_orth_bootstrap_enable=True but X is not a "
                "pandas DataFrame; bootstrap-stable hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the bootstrap-stable selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_bootstrap_mi_fe import (
                    hybrid_orth_mi_bootstrap_fe_with_recipes,
                )
                _y_for_boot = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _boot_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _boot_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
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
                X_boot, _boot_scores, _boot_recipes = (
                    hybrid_orth_mi_bootstrap_fe_with_recipes(
                        X, _y_for_boot,
                        cols=_boot_cols,
                        degrees=_boot_degrees,
                        basis=_boot_basis,
                        top_k=_boot_top_k,
                        n_boot=_boot_n,
                        sample_fraction=_boot_frac,
                        seed=_boot_seed,
                    )
                )
                _boot_appended = [
                    c for c in X_boot.columns if c not in _X_before_boot_cols
                ]
                if _boot_appended:
                    X = X_boot
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_boot_appended)
                    )
                    for _r in _boot_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth bootstrap-stable: appended "
                            "%d engineered column(s): %s",
                            len(_boot_appended), _boot_appended[:8],
                        )
            except Exception as _boot_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth bootstrap-stable FE raised %s: %s; "
                    "continuing without bootstrap-stable columns.",
                    type(_boot_exc).__name__, _boot_exc,
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
        _is_pandas_for_tg = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_tg:
            warnings.warn(
                "MRMR: fe_hybrid_orth_three_gate_enable=True but X is not a "
                "pandas DataFrame; three-gate hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the three-gate selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_three_gate_mi_fe import (
                    hybrid_orth_mi_three_gate_fe_with_recipes,
                )
                _y_for_tg = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _tg_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _tg_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
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
                _tg_support_cols = [
                    c for c in _hybrid_already_appended if c in X.columns
                ]
                _tg_current_support = (
                    X[_tg_support_cols].copy()
                    if _tg_support_cols
                    else None
                )
                _X_before_tg_cols = list(X.columns)
                X_tg, _tg_scores, _tg_recipes = (
                    hybrid_orth_mi_three_gate_fe_with_recipes(
                        X, _y_for_tg, _tg_current_support,
                        cols=_tg_cols,
                        degrees=_tg_degrees,
                        basis=_tg_basis,
                        top_k=_tg_top_k,
                        cmi_min=_tg_cmi_min,
                        n_folds=_tg_n_folds,
                        seed=_tg_seed,
                    )
                )
                _tg_appended = [
                    c for c in X_tg.columns if c not in _X_before_tg_cols
                ]
                if _tg_appended:
                    X = X_tg
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_tg_appended)
                    )
                    for _r in _tg_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth three-gate: appended "
                            "%d engineered column(s): %s",
                            len(_tg_appended), _tg_appended[:8],
                        )
            except Exception as _tg_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth three-gate FE raised %s: %s; "
                    "continuing without three-gate columns.",
                    type(_tg_exc).__name__, _tg_exc,
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
        _is_pandas_for_ksg = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_ksg:
            warnings.warn(
                "MRMR: fe_hybrid_orth_ksg_enable=True but X is not a "
                "pandas DataFrame; KSG hybrid FE is skipped. Convert to "
                "pandas via X.to_pandas() before fit() if you want the "
                "KSG-MI selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_ksg_mi_fe import (
                    hybrid_orth_mi_ksg_fe_with_recipes,
                )
                _y_for_ksg = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _ksg_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _ksg_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _ksg_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
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
                X_ksg, _ksg_scores, _ksg_recipes = (
                    hybrid_orth_mi_ksg_fe_with_recipes(
                        X, _y_for_ksg,
                        cols=_ksg_cols,
                        degrees=_ksg_degrees,
                        basis=_ksg_basis,
                        top_k=_ksg_top_k,
                        min_uplift=_ksg_min_uplift,
                        min_abs_mi_frac=_ksg_min_abs_mi_frac,
                        n_neighbors=_ksg_n_neighbors,
                        random_state=_ksg_seed,
                    )
                )
                _ksg_appended = [
                    c for c in X_ksg.columns if c not in _X_before_ksg_cols
                ]
                if _ksg_appended:
                    X = X_ksg
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_ksg_appended)
                    )
                    for _r in _ksg_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth KSG-MI: appended "
                            "%d engineered column(s): %s",
                            len(_ksg_appended), _ksg_appended[:8],
                        )
            except Exception as _ksg_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth KSG-MI FE raised %s: %s; "
                    "continuing without KSG-MI columns.",
                    type(_ksg_exc).__name__, _ksg_exc,
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
        _is_pandas_for_copula = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_copula:
            warnings.warn(
                "MRMR: fe_hybrid_orth_copula_enable=True but X is not a "
                "pandas DataFrame; copula-MI hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the copula-MI selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_copula_mi_fe import (
                    hybrid_orth_mi_copula_fe_with_recipes,
                )
                _y_for_copula = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _copula_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _copula_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _copula_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _copula_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
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
                X_copula, _copula_scores, _copula_recipes = (
                    hybrid_orth_mi_copula_fe_with_recipes(
                        X, _y_for_copula,
                        cols=_copula_cols,
                        degrees=_copula_degrees,
                        basis=_copula_basis,
                        top_k=_copula_top_k,
                        min_uplift=_copula_min_uplift,
                        min_abs_mi_frac=_copula_min_abs_mi_frac,
                        n_bins=_copula_n_bins,
                    )
                )
                _copula_appended = [
                    c for c in X_copula.columns
                    if c not in _X_before_copula_cols
                ]
                if _copula_appended:
                    X = X_copula
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_copula_appended)
                    )
                    for _r in _copula_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth copula-MI: appended "
                            "%d engineered column(s): %s",
                            len(_copula_appended), _copula_appended[:8],
                        )
            except Exception as _copula_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth copula-MI FE raised %s: %s; "
                    "continuing without copula-MI columns.",
                    type(_copula_exc).__name__, _copula_exc,
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
        _is_pandas_for_dcor = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_dcor:
            warnings.warn(
                "MRMR: fe_hybrid_orth_dcor_enable=True but X is not a "
                "pandas DataFrame; dCor hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the dCor selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_dcor_fe import (
                    hybrid_orth_mi_dcor_fe_with_recipes,
                )
                _y_for_dcor = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _dcor_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _dcor_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _dcor_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _dcor_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
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
                X_dcor, _dcor_scores, _dcor_recipes = (
                    hybrid_orth_mi_dcor_fe_with_recipes(
                        X, _y_for_dcor,
                        cols=_dcor_cols,
                        degrees=_dcor_degrees,
                        basis=_dcor_basis,
                        top_k=_dcor_top_k,
                        min_uplift=_dcor_min_uplift,
                        min_abs_mi_frac=_dcor_min_abs_mi_frac,
                        n_sample=_dcor_n_sample,
                        random_state=int(getattr(self, "random_seed", 0) or 0),
                    )
                )
                _dcor_appended = [
                    c for c in X_dcor.columns
                    if c not in _X_before_dcor_cols
                ]
                if _dcor_appended:
                    X = X_dcor
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_dcor_appended)
                    )
                    for _r in _dcor_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth dCor: appended %d "
                            "engineered column(s): %s",
                            len(_dcor_appended), _dcor_appended[:8],
                        )
            except Exception as _dcor_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth dCor FE raised %s: %s; "
                    "continuing without dCor columns.",
                    type(_dcor_exc).__name__, _dcor_exc,
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
        _is_pandas_for_hsic = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_hsic:
            warnings.warn(
                "MRMR: fe_hybrid_orth_hsic_enable=True but X is not a "
                "pandas DataFrame; HSIC hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the HSIC selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_hsic_fe import (
                    hybrid_orth_mi_hsic_fe_with_recipes,
                )
                _y_for_hsic = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _hsic_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _hsic_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _hsic_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _hsic_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
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
                X_hsic, _hsic_scores, _hsic_recipes = (
                    hybrid_orth_mi_hsic_fe_with_recipes(
                        X, _y_for_hsic,
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
                )
                _hsic_appended = [
                    c for c in X_hsic.columns
                    if c not in _X_before_hsic_cols
                ]
                if _hsic_appended:
                    X = X_hsic
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_hsic_appended)
                    )
                    for _r in _hsic_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth HSIC: appended %d "
                            "engineered column(s): %s",
                            len(_hsic_appended), _hsic_appended[:8],
                        )
            except Exception as _hsic_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth HSIC FE raised %s: %s; "
                    "continuing without HSIC columns.",
                    type(_hsic_exc).__name__, _hsic_exc,
                )
    # 2026-06-01 Layer 72 — JMIM (Bennasar 2015) redundancy-aware ranking
    # for hybrid orth-poly FE (independent opt-in; does NOT require
    # fe_hybrid_orth_enable). Each engineered candidate is scored by
    # ``min over X_j in S of I((X_cand, X_j); Y)`` where S is the raw
    # source column pool. Selection: same two-gate rule as Layers 65 /
    # 66 / 67 / 71. Engineered VALUES bit-equal to Layer 21 -> recipes
    # reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_jmim_enable", False)):
        _is_pandas_for_jmim = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_jmim:
            warnings.warn(
                "MRMR: fe_hybrid_orth_jmim_enable=True but X is not a "
                "pandas DataFrame; JMIM hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the JMIM selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_jmim_fe import (
                    hybrid_orth_mi_jmim_fe_with_recipes,
                )
                _y_for_jmim = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _jmim_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _jmim_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _jmim_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _jmim_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
                _jmim_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
                _jmim_n_bins = int(getattr(
                    self, "fe_hybrid_orth_jmim_n_bins", 10,
                ))
                # Same calibration as Layers 65 / 66 / 67 / 71: 0.95 /
                # 0.05 floor keeps genuine borderline wins.
                _jmim_min_uplift = 0.95
                _jmim_min_abs_mi_frac = 0.05
                _X_before_jmim_cols = list(X.columns)
                X_jmim, _jmim_scores, _jmim_recipes = (
                    hybrid_orth_mi_jmim_fe_with_recipes(
                        X, _y_for_jmim,
                        cols=_jmim_cols,
                        degrees=_jmim_degrees,
                        basis=_jmim_basis,
                        top_k=_jmim_top_k,
                        min_uplift=_jmim_min_uplift,
                        min_abs_mi_frac=_jmim_min_abs_mi_frac,
                        n_bins=_jmim_n_bins,
                    )
                )
                _jmim_appended = [
                    c for c in X_jmim.columns
                    if c not in _X_before_jmim_cols
                ]
                if _jmim_appended:
                    X = X_jmim
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_jmim_appended)
                    )
                    for _r in _jmim_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth JMIM: appended %d "
                            "engineered column(s): %s",
                            len(_jmim_appended), _jmim_appended[:8],
                        )
            except Exception as _jmim_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth JMIM FE raised %s: %s; "
                    "continuing without JMIM columns.",
                    type(_jmim_exc).__name__, _jmim_exc,
                )
    # 2026-06-01 Layer 73 — Total Correlation (Watanabe 1960) multivariate-
    # redundancy ranking for hybrid orth-poly FE (independent opt-in; does
    # NOT require fe_hybrid_orth_enable). Each engineered candidate is
    # scored by the FULL-ORDER joint shared information delta against the
    # current support union with y. Selection: same absolute floor as
    # Layers 65 / 66 / 67 / 71 / 72. Engineered VALUES bit-equal to Layer
    # 21 -> recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_tc_enable", False)):
        _is_pandas_for_tc = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_tc:
            warnings.warn(
                "MRMR: fe_hybrid_orth_tc_enable=True but X is not a "
                "pandas DataFrame; Total-Correlation hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the TC selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_total_correlation_fe import (
                    hybrid_orth_mi_tc_fe_with_recipes,
                )
                _y_for_tc = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _tc_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _tc_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _tc_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _tc_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
                _tc_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
                _tc_n_bins = int(getattr(
                    self, "fe_hybrid_orth_tc_n_bins", 10,
                ))
                # Same calibration as Layers 65 / 66 / 67 / 71 / 72: 0.95 /
                # 0.05 floor keeps genuine borderline wins.
                _tc_min_uplift = 0.95
                _tc_min_abs_mi_frac = 0.05
                _X_before_tc_cols = list(X.columns)
                X_tc, _tc_scores, _tc_recipes = (
                    hybrid_orth_mi_tc_fe_with_recipes(
                        X, _y_for_tc,
                        cols=_tc_cols,
                        degrees=_tc_degrees,
                        basis=_tc_basis,
                        top_k=_tc_top_k,
                        min_uplift=_tc_min_uplift,
                        min_abs_mi_frac=_tc_min_abs_mi_frac,
                        n_bins=_tc_n_bins,
                    )
                )
                _tc_appended = [
                    c for c in X_tc.columns
                    if c not in _X_before_tc_cols
                ]
                if _tc_appended:
                    X = X_tc
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_tc_appended)
                    )
                    for _r in _tc_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth TC: appended %d "
                            "engineered column(s): %s",
                            len(_tc_appended), _tc_appended[:8],
                        )
            except Exception as _tc_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth TC FE raised %s: %s; "
                    "continuing without TC columns.",
                    type(_tc_exc).__name__, _tc_exc,
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
        _is_pandas_for_cmim = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_cmim:
            warnings.warn(
                "MRMR: fe_hybrid_orth_cmim_enable=True but X is not a "
                "pandas DataFrame; CMIM hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the CMIM selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_cmim_fe import (
                    hybrid_orth_mi_cmim_fe_with_recipes,
                )
                _y_for_cmim = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _cmim_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _cmim_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _cmim_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _cmim_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
                _cmim_top_k = int(getattr(self, "fe_hybrid_orth_top_k", 5))
                _cmim_n_bins = int(getattr(
                    self, "fe_hybrid_orth_cmim_n_bins", 10,
                ))
                # Same calibration as Layers 65 / 66 / 67 / 71 / 72 / 73:
                # 0.95 / 0.05 floor keeps genuine borderline wins.
                _cmim_min_uplift = 0.95
                _cmim_min_abs_mi_frac = 0.05
                _X_before_cmim_cols = list(X.columns)
                X_cmim, _cmim_scores, _cmim_recipes = (
                    hybrid_orth_mi_cmim_fe_with_recipes(
                        X, _y_for_cmim,
                        cols=_cmim_cols,
                        degrees=_cmim_degrees,
                        basis=_cmim_basis,
                        top_k=_cmim_top_k,
                        min_uplift=_cmim_min_uplift,
                        min_abs_mi_frac=_cmim_min_abs_mi_frac,
                        n_bins=_cmim_n_bins,
                    )
                )
                _cmim_appended = [
                    c for c in X_cmim.columns
                    if c not in _X_before_cmim_cols
                ]
                if _cmim_appended:
                    X = X_cmim
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_cmim_appended)
                    )
                    for _r in _cmim_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth CMIM: appended %d "
                            "engineered column(s): %s",
                            len(_cmim_appended), _cmim_appended[:8],
                        )
            except Exception as _cmim_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth CMIM FE raised %s: %s; "
                    "continuing without CMIM columns.",
                    type(_cmim_exc).__name__, _cmim_exc,
                )
    # 2026-06-01 Layer 68 — PER-COLUMN SCORER AUTO-SELECTION across the
    # Layer 21 / 65 / 66 / 67 scorer family (independent opt-in; does NOT
    # require fe_hybrid_orth_enable). For each engineered column the
    # bootstrap-LCB criterion picks the best scorer in
    # {plug-in, KSG, copula, dCor} and uses ITS LCB for the cross-column
    # ranking + selection. Engineered VALUES bit-equal to Layer 21 ->
    # recipes reuse the ``orth_univariate`` kind.
    if bool(getattr(self, "fe_hybrid_orth_auto_scorer_enable", False)):
        _is_pandas_for_auto = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_auto:
            warnings.warn(
                "MRMR: fe_hybrid_orth_auto_scorer_enable=True but X is "
                "not a pandas DataFrame; auto-scorer hybrid FE is "
                "skipped. Convert to pandas via X.to_pandas() before "
                "fit() if you want the auto-scorer selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_scorer_auto_fe import (
                    hybrid_orth_mi_auto_scorer_fe_with_recipes,
                )
                _y_for_auto = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _auto_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _auto_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _auto_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _auto_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
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
                X_auto, _auto_scores, _auto_recipes = (
                    hybrid_orth_mi_auto_scorer_fe_with_recipes(
                        X, _y_for_auto,
                        cols=_auto_cols,
                        degrees=_auto_degrees,
                        basis=_auto_basis,
                        top_k=_auto_top_k,
                        min_uplift=_auto_min_uplift,
                        min_abs_mi_frac=_auto_min_abs_mi_frac,
                        n_boot=_auto_n_boot,
                        random_state=int(getattr(self, "random_seed", 0) or 0),
                    )
                )
                _auto_appended = [
                    c for c in X_auto.columns
                    if c not in _X_before_auto_cols
                ]
                if _auto_appended:
                    X = X_auto
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_auto_appended)
                    )
                    for _r in _auto_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth auto-scorer: appended "
                            "%d engineered column(s): %s",
                            len(_auto_appended), _auto_appended[:8],
                        )
            except Exception as _auto_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth auto-scorer FE raised %s: %s; "
                    "continuing without auto-scorer columns.",
                    type(_auto_exc).__name__, _auto_exc,
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
        _is_pandas_for_ens = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_ens:
            warnings.warn(
                "MRMR: fe_hybrid_orth_ensemble_enable=True but X is not a "
                "pandas DataFrame; ensemble hybrid FE is skipped. Convert "
                "to pandas via X.to_pandas() before fit() if you want the "
                "ensemble selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_scorer_auto_fe import (
                    hybrid_orth_mi_ensemble_fe_with_recipes,
                )
                _y_for_ens = (
                    y.to_numpy() if hasattr(y, "to_numpy")
                    else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _ens_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns
                        and c not in _hybrid_already_appended
                    ]
                else:
                    _ens_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
                _ens_degrees = tuple(int(d) for d in getattr(
                    self, "fe_hybrid_orth_degrees", (2, 3),
                ))
                _ens_basis = str(getattr(
                    self, "fe_hybrid_orth_basis", "auto",
                ))
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
                X_ens, _ens_scores, _ens_recipes = (
                    hybrid_orth_mi_ensemble_fe_with_recipes(
                        X, _y_for_ens,
                        cols=_ens_cols,
                        degrees=_ens_degrees,
                        basis=_ens_basis,
                        top_k=_ens_top_k,
                        min_uplift=_ens_min_uplift,
                        min_abs_mi_frac=_ens_min_abs_mi_frac,
                        scorers=_ens_scorers,
                        aggregator=_ens_aggregator,
                        random_state=int(
                            getattr(self, "random_seed", 0) or 0
                        ),
                    )
                )
                _ens_appended = [
                    c for c in X_ens.columns
                    if c not in _X_before_ens_cols
                ]
                if _ens_appended:
                    X = X_ens
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_ens_appended)
                    )
                    for _r in _ens_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth ensemble: appended %d "
                            "engineered column(s) via %s aggregator: %s",
                            len(_ens_appended), _ens_aggregator,
                            _ens_appended[:8],
                        )
            except Exception as _ens_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth ensemble FE raised %s: %s; "
                    "continuing without ensemble columns.",
                    type(_ens_exc).__name__, _ens_exc,
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
        _is_pandas_for_meta = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_meta:
            warnings.warn(
                "MRMR: fe_hybrid_orth_meta_enable=True but X is not a "
                "pandas DataFrame; meta-scorer hybrid FE is skipped. "
                "Convert to pandas via X.to_pandas() before fit() if you "
                "want the meta-scorer selection applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._orthogonal_meta_scorer_fe import (
                    hybrid_orth_mi_meta_fe_with_recipes,
                )
                _y_for_meta = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _hybrid_already_appended = set(
                    getattr(self, "hybrid_orth_features_", None) or []
                )
                if getattr(self, "factors_names_to_use", None):
                    _meta_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _hybrid_already_appended
                    ]
                else:
                    _meta_cols = [
                        c for c in X.columns
                        if c not in _hybrid_already_appended
                    ]
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
                ) = hybrid_orth_mi_meta_fe_with_recipes(
                    X, _y_for_meta,
                    cols=_meta_cols,
                    degrees=_meta_degrees,
                    basis=_meta_basis,
                    top_k=_meta_top_k,
                    min_uplift=_meta_min_uplift,
                    min_abs_mi_frac=_meta_min_abs_mi_frac,
                    force_scorer=_meta_force,
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                )
                _meta_appended = [
                    c for c in X_meta.columns
                    if c not in _X_before_meta_cols
                ]
                if _meta_appended:
                    X = X_meta
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or [])
                        + list(_meta_appended)
                    )
                    for _r in _meta_recipes:
                        _hybrid_orth_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit hybrid_orth meta-scorer: dispatched "
                            "to %r (force=%r); appended %d engineered "
                            "column(s): %s",
                            _meta_chosen, _meta_force,
                            len(_meta_appended), _meta_appended[:8],
                        )
                # Expose the chosen scorer + fingerprint for downstream
                # audit / debug (also survives pickle because plain attrs).
                self.hybrid_orth_meta_chosen_scorer_ = _meta_chosen
                self.hybrid_orth_meta_fingerprint_ = dict(_meta_fp)
            except Exception as _meta_exc:
                logger.warning(
                    "MRMR.fit hybrid_orth meta-scorer FE raised %s: %s; "
                    "continuing without meta-scorer columns.",
                    type(_meta_exc).__name__, _meta_exc,
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
        _is_pandas_for_cmi = isinstance(X, pd.DataFrame)
        if not _is_pandas_for_cmi:
            warnings.warn(
                "MRMR: fe_mi_greedy_cmi_enable=True but X is not a pandas "
                "DataFrame; CMI-greedy FE is skipped. Convert to pandas via "
                "X.to_pandas() before fit() if you want CMI-greedy FE applied.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._mi_greedy_cmi_fe import greedy_cmi_fe_construct_with_recipes
                _y_for_cmi = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _eng_already_appended = (
                    set(getattr(self, "hybrid_orth_features_", None) or [])
                    | set(self.mi_greedy_features_ or [])
                )
                if getattr(self, "factors_names_to_use", None):
                    _cmi_cols = [
                        c for c in self.factors_names_to_use
                        if c in X.columns and c not in _eng_already_appended
                    ]
                else:
                    _cmi_cols = [
                        c for c in X.columns
                        if c not in _eng_already_appended
                    ]
                _X_before_cmi_cols = list(X.columns)
                X_cmi, _cmi_scores, _cmi_recipes = greedy_cmi_fe_construct_with_recipes(
                    X, _y_for_cmi,
                    cols=_cmi_cols,
                    seed_cols_count=int(self.fe_mi_greedy_cmi_seed_cols_count),
                    top_k=int(self.fe_mi_greedy_cmi_top_k),
                    include_unary=bool(getattr(self, "fe_mi_greedy_include_unary", True)),
                    include_binary=bool(getattr(self, "fe_mi_greedy_include_binary", True)),
                    min_cmi_gain=float(self.fe_mi_greedy_cmi_min_gain),
                )
                _cmi_appended = [
                    c for c in X_cmi.columns
                    if c not in _X_before_cmi_cols
                ]
                if _cmi_appended:
                    X = X_cmi
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
                            "MRMR.fit mi_greedy_cmi: appended %d engineered "
                            "column(s): %s",
                            len(_cmi_appended), _cmi_appended[:8],
                        )
            except Exception as _cmi_exc:
                logger.warning(
                    "MRMR.fit mi_greedy_cmi FE raised %s: %s; continuing "
                    "without CMI-greedy columns.",
                    type(_cmi_exc).__name__, _cmi_exc,
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
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
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
                    _y_for_cnt = (
                        y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                    )
                    X_c, _cnt_appended, _cnt_recipes = count_encode_with_recipes(
                        X, cat_cols=_cnt_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_cnt,
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
                    _y_for_freq = (
                        y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                    )
                    X_f, _freq_appended, _freq_recipes = frequency_encode_with_recipes(
                        X, cat_cols=_freq_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_freq,
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
                                mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                                mi_gate_top_k=int(
                                    getattr(self, "fe_local_mi_gate_top_k", 20)
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
                    _y_for_ind = (
                        y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                    )
                    X_i, _ind_appended, _ind_recipes = missing_indicator_with_recipes(
                        X, cols=_ind_cols,
                        mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                        mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                        y=_y_for_ind,
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
    self.group_distance_features_ = []
    _cat_pair_pre_recipes: dict = {}
    _cat_triple_pre_recipes: dict = {}
    _numeric_decompose_pre_recipes: dict = {}
    _temporal_agg_pre_recipes: dict = {}
    _modular_pre_recipes: dict = {}
    _group_distance_pre_recipes: dict = {}
    _rare_category_pre_recipes: dict = {}
    _conditional_residual_pre_recipes: dict = {}
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
        _is_pandas_l38 = isinstance(X, pd.DataFrame)
        if not _is_pandas_l38:
            warnings.warn(
                "MRMR: Layer 38 FE (ratio/log_ratio/grouped_delta/lagged_diff) "
                "enabled but X is not a pandas DataFrame; the encodings are "
                "skipped. Convert to pandas via X.to_pandas() before fit() to "
                "apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            from ._ratio_delta_fe import (
                pairwise_ratio_with_recipes,
                pairwise_log_ratio_with_recipes,
                grouped_delta_with_recipes,
                lagged_diff_with_recipes,
            )

            _l38_mi_gate = bool(getattr(self, "fe_local_mi_gate", False))
            _l38_mi_gate_top_k = int(getattr(self, "fe_local_mi_gate_top_k", 20))
            _y_for_l38 = (
                y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
            )

            # ----- Pairwise ratio --------------------------------------------
            if bool(getattr(self, "fe_pairwise_ratio_enable", False)):
                try:
                    _ratio_cols = tuple(
                        getattr(self, "fe_pairwise_ratio_cols", ()) or ()
                    )
                    _ratio_cols = [c for c in _ratio_cols if c in X.columns]
                    _eps = float(getattr(self, "fe_pairwise_ratio_eps", 1e-9))
                    _X_before_r_cols = list(X.columns)
                    X_r, _r_appended, _r_recipes = pairwise_ratio_with_recipes(
                        X, cols=_ratio_cols, eps=_eps,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38,
                    )
                    _r_appended = [
                        c for c in _r_appended if c not in _X_before_r_cols
                    ]
                    if _r_appended:
                        X = X_r
                        self.pairwise_ratio_features_ = list(_r_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_r_appended)
                        )
                        for _r in _r_recipes:
                            if _r.name in _r_appended:
                                _ratio_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit pairwise_ratio: appended %d "
                                "engineered column(s): %s",
                                len(_r_appended), _r_appended[:8],
                            )
                except Exception as _r_exc:
                    logger.warning(
                        "MRMR.fit pairwise_ratio FE raised %s: %s; "
                        "continuing without ratio columns.",
                        type(_r_exc).__name__, _r_exc,
                    )

            # ----- Pairwise log-ratio ----------------------------------------
            if bool(getattr(self, "fe_pairwise_log_ratio_enable", False)):
                try:
                    _lr_cols = tuple(
                        getattr(self, "fe_pairwise_log_ratio_cols", ()) or ()
                    )
                    _lr_cols = [c for c in _lr_cols if c in X.columns]
                    _eps_lr = float(getattr(self, "fe_pairwise_ratio_eps", 1e-9))
                    _X_before_lr_cols = list(X.columns)
                    X_lr, _lr_appended, _lr_recipes = pairwise_log_ratio_with_recipes(
                        X, cols=_lr_cols, eps=_eps_lr,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38,
                    )
                    _lr_appended = [
                        c for c in _lr_appended if c not in _X_before_lr_cols
                    ]
                    if _lr_appended:
                        X = X_lr
                        self.pairwise_log_ratio_features_ = list(_lr_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_lr_appended)
                        )
                        for _r in _lr_recipes:
                            if _r.name in _lr_appended:
                                _log_ratio_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit pairwise_log_ratio: appended %d "
                                "engineered column(s): %s",
                                len(_lr_appended), _lr_appended[:8],
                            )
                except Exception as _lr_exc:
                    logger.warning(
                        "MRMR.fit pairwise_log_ratio FE raised %s: %s; "
                        "continuing without log-ratio columns.",
                        type(_lr_exc).__name__, _lr_exc,
                    )

            # ----- Grouped delta ---------------------------------------------
            if bool(getattr(self, "fe_grouped_delta_enable", False)):
                try:
                    _gd_group = getattr(self, "fe_grouped_delta_group_col", None)
                    _gd_nums = tuple(
                        getattr(self, "fe_grouped_delta_num_cols", ()) or ()
                    )
                    _gd_nums = [c for c in _gd_nums if c in X.columns]
                    _X_before_gd_cols = list(X.columns)
                    X_gd, _gd_appended, _gd_recipes = grouped_delta_with_recipes(
                        X, group_col=_gd_group, num_cols=_gd_nums,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38,
                    )
                    _gd_appended = [
                        c for c in _gd_appended if c not in _X_before_gd_cols
                    ]
                    if _gd_appended:
                        X = X_gd
                        self.grouped_delta_features_ = list(_gd_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_gd_appended)
                        )
                        for _r in _gd_recipes:
                            if _r.name in _gd_appended:
                                _grouped_delta_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit grouped_delta: appended %d "
                                "engineered column(s): %s",
                                len(_gd_appended), _gd_appended[:8],
                            )
                except Exception as _gd_exc:
                    logger.warning(
                        "MRMR.fit grouped_delta FE raised %s: %s; "
                        "continuing without grouped-delta columns.",
                        type(_gd_exc).__name__, _gd_exc,
                    )

            # ----- Lagged diff -----------------------------------------------
            if bool(getattr(self, "fe_lagged_diff_enable", False)):
                try:
                    _ld_time = getattr(self, "fe_lagged_diff_time_col", None)
                    _ld_vals = tuple(
                        getattr(self, "fe_lagged_diff_value_cols", ()) or ()
                    )
                    _ld_vals = [c for c in _ld_vals if c in X.columns]
                    _ld_periods = tuple(
                        getattr(self, "fe_lagged_diff_periods", (1, 2)) or (1, 2)
                    )
                    _X_before_ld_cols = list(X.columns)
                    X_ld, _ld_appended, _ld_recipes = lagged_diff_with_recipes(
                        X, time_col=_ld_time, value_cols=_ld_vals,
                        periods=_ld_periods,
                        mi_gate=_l38_mi_gate, mi_gate_top_k=_l38_mi_gate_top_k,
                        y=_y_for_l38,
                    )
                    _ld_appended = [
                        c for c in _ld_appended if c not in _X_before_ld_cols
                    ]
                    if _ld_appended:
                        X = X_ld
                        self.lagged_diff_features_ = list(_ld_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_ld_appended)
                        )
                        for _r in _ld_recipes:
                            if _r.name in _ld_appended:
                                _lagged_diff_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit lagged_diff: appended %d "
                                "engineered column(s): %s",
                                len(_ld_appended), _ld_appended[:8],
                            )
                except Exception as _ld_exc:
                    logger.warning(
                        "MRMR.fit lagged_diff FE raised %s: %s; "
                        "continuing without lagged-diff columns.",
                        type(_ld_exc).__name__, _ld_exc,
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
                from ._grouped_agg_fe import hybrid_grouped_agg_fe

                # CMI gate needs a class-typed target; bin continuous y the
                # same way the Layer 60 CMI-greedy stage does.
                _y_for_ga = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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

                _ga_groups = tuple(
                    getattr(self, "fe_grouped_agg_group_cols", ()) or ()
                )
                _ga_groups = [c for c in _ga_groups if c in X.columns] or None
                _ga_nums = tuple(
                    getattr(self, "fe_grouped_agg_num_cols", ()) or ()
                )
                _ga_nums = [c for c in _ga_nums if c in X.columns] or None
                _ga_stats = tuple(
                    getattr(self, "fe_grouped_agg_stats", ())
                    or ("mean", "std", "min", "max", "nunique", "skew", "median")
                )
                _ga_top_k = int(getattr(self, "fe_grouped_agg_top_k", 10))
                _X_before_ga_cols = list(X.columns)
                X_ga, _ga_appended, _ga_recipes, _ga_scores = hybrid_grouped_agg_fe(
                    X, _y_for_ga,
                    group_cols=_ga_groups, num_cols=_ga_nums,
                    stats=_ga_stats, top_k=_ga_top_k,
                )
                _ga_appended = [
                    c for c in _ga_appended if c not in _X_before_ga_cols
                ]
                if _ga_appended:
                    X = X_ga
                    self.grouped_agg_features_ = list(_ga_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_ga_appended)
                    )
                    for _r in _ga_recipes:
                        if _r.name in _ga_appended:
                            _grouped_agg_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit grouped_agg: appended %d engineered "
                            "column(s): %s",
                            len(_ga_appended), _ga_appended[:8],
                        )
            except Exception as _ga_exc:
                logger.warning(
                    "MRMR.fit grouped_agg FE raised %s: %s; continuing "
                    "without grouped-aggregate columns.",
                    type(_ga_exc).__name__, _ga_exc,
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
                from ._composite_group_agg_fe import hybrid_composite_group_agg_fe

                _y_for_cga = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
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
                _cga_key_sets_raw = tuple(
                    getattr(self, "fe_composite_group_agg_key_sets", ()) or ()
                )
                _cga_key_sets = [
                    tuple(c for c in gset if c in X.columns)
                    for gset in _cga_key_sets_raw
                ]
                _cga_key_sets = [g for g in _cga_key_sets if len(g) >= 2] or None
                _cga_nums = tuple(
                    getattr(self, "fe_composite_group_agg_num_cols", ()) or ()
                )
                _cga_nums = [c for c in _cga_nums if c in X.columns] or None
                _cga_stats = tuple(
                    getattr(self, "fe_composite_group_agg_stats", ())
                    or ("mean", "std", "count")
                )
                _cga_max_arity = int(
                    getattr(self, "fe_composite_group_agg_max_arity", 2)
                )
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
                _cga_appended = [
                    c for c in _cga_appended if c not in _X_before_cga_cols
                ]
                if _cga_appended:
                    X = X_cga
                    self.composite_group_agg_features_ = list(_cga_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_cga_appended)
                    )
                    for _r in _cga_recipes:
                        if _r.name in _cga_appended:
                            _composite_group_agg_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit composite_group_agg: appended %d "
                            "engineered column(s): %s",
                            len(_cga_appended), _cga_appended[:8],
                        )
            except Exception as _cga_exc:
                logger.warning(
                    "MRMR.fit composite_group_agg FE raised %s: %s; continuing "
                    "without composite-aggregate columns.",
                    type(_cga_exc).__name__, _cga_exc,
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
                from ._grouped_quantile_fe import hybrid_grouped_quantile_fe

                _y_for_gq = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _gq_groups = tuple(
                    getattr(self, "fe_grouped_quantile_group_cols", ()) or ()
                )
                _gq_groups = [c for c in _gq_groups if c in X.columns] or None
                _gq_nums = tuple(
                    getattr(self, "fe_grouped_quantile_num_cols", ()) or ()
                )
                _gq_nums = [c for c in _gq_nums if c in X.columns] or None
                _gq_quantiles = tuple(
                    getattr(self, "fe_grouped_quantile_quantiles", ())
                    or (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
                )
                _gq_target_aware = bool(
                    getattr(self, "fe_grouped_quantile_target_aware", False)
                )
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
                _gq_appended = [
                    c for c in _gq_appended if c not in _X_before_gq_cols
                ]
                if _gq_appended:
                    X = X_gq
                    self.grouped_quantile_features_ = list(_gq_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_gq_appended)
                    )
                    for _r in _gq_recipes:
                        if _r.name in _gq_appended:
                            _grouped_quantile_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit grouped_quantile: appended %d engineered "
                            "column(s): %s",
                            len(_gq_appended), _gq_appended[:8],
                        )
            except Exception as _gq_exc:
                logger.warning(
                    "MRMR.fit grouped_quantile FE raised %s: %s; continuing "
                    "without grouped-quantile columns.",
                    type(_gq_exc).__name__, _gq_exc,
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
                from ._cat_pair_fe import hybrid_cat_pair_fe

                _y_for_cp = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _cp_cols = tuple(
                    getattr(self, "fe_cat_pair_cat_cols", ()) or ()
                )
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
                    _cp_cols = [
                        c for c in _raw_input_cols_pre_fe if c in X.columns
                    ] or None
                _cp_min_ii = float(
                    getattr(self, "fe_cat_pair_min_interaction_info", 0.001)
                )
                _cp_top_k = int(getattr(self, "fe_cat_pair_top_k", 5))
                _X_before_cp_cols = list(X.columns)
                X_cp, _cp_appended, _cp_recipes, _cp_scores = hybrid_cat_pair_fe(
                    X, _y_for_cp,
                    cat_cols=_cp_cols,
                    min_interaction_info=_cp_min_ii,
                    top_k=_cp_top_k,
                    random_state=int(getattr(self, "random_seed", 0) or 0),
                )
                _cp_appended = [
                    c for c in _cp_appended if c not in _X_before_cp_cols
                ]
                if _cp_appended:
                    X = X_cp
                    self.cat_pair_features_ = list(_cp_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_cp_appended)
                    )
                    for _r in _cp_recipes:
                        if _r.name in _cp_appended:
                            _cat_pair_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit cat_pair: appended %d engineered "
                            "column(s): %s",
                            len(_cp_appended), _cp_appended[:8],
                        )
            except Exception as _cp_exc:
                logger.warning(
                    "MRMR.fit cat_pair FE raised %s: %s; continuing without "
                    "cat-pair-cross columns.",
                    type(_cp_exc).__name__, _cp_exc,
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
                from ._cat_triple_fe import hybrid_cat_triple_fe

                _y_for_ct = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _ct_cols = tuple(
                    getattr(self, "fe_cat_triple_cat_cols", ()) or ()
                )
                _ct_cols = [c for c in _ct_cols if c in X.columns] or None
                # Same raw-column restriction as the cat_pair stage: auto-
                # detected triple members must be raw inputs so the cross
                # recipe replays as a pure function of X (an engineered
                # intermediate would raise KeyError at transform time).
                if _ct_cols is None:
                    _ct_cols = [
                        c for c in _raw_input_cols_pre_fe if c in X.columns
                    ] or None
                _ct_min_ii = float(
                    getattr(self, "fe_cat_triple_min_interaction_info", 0.001)
                )
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
                _ct_appended = [
                    c for c in _ct_appended if c not in _X_before_ct_cols
                ]
                if _ct_appended:
                    X = X_ct
                    self.cat_triple_features_ = list(_ct_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_ct_appended)
                    )
                    for _r in _ct_recipes:
                        if _r.name in _ct_appended:
                            _cat_triple_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit cat_triple: appended %d engineered "
                            "column(s): %s",
                            len(_ct_appended), _ct_appended[:8],
                        )
            except Exception as _ct_exc:
                logger.warning(
                    "MRMR.fit cat_triple FE raised %s: %s; continuing without "
                    "cat-triple-cross columns.",
                    type(_ct_exc).__name__, _ct_exc,
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
                from ._numeric_decompose_fe import (
                    hybrid_numeric_decompose_fe_with_recipes,
                )

                _y_for_nd = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _nd_precisions = tuple(
                    getattr(self, "fe_numeric_decompose_precisions",
                            (1, 0.1, 0.01, 0.001))
                )
                _nd_digits = tuple(
                    getattr(self, "fe_numeric_decompose_digits", (0, 1, 2))
                )
                _nd_n_boot = int(getattr(self, "fe_numeric_decompose_n_boot", 10))
                _nd_top_k = int(getattr(self, "fe_numeric_decompose_top_k", 5))
                _X_before_nd_cols = list(X.columns)
                X_nd, _nd_appended, _nd_recipes, _nd_scores = (
                    hybrid_numeric_decompose_fe_with_recipes(
                        X, _y_for_nd,
                        cols=None,
                        precisions=_nd_precisions,
                        digit_positions=_nd_digits,
                        top_k=_nd_top_k,
                        n_boot=_nd_n_boot,
                        seed=int(getattr(self, "random_seed", 0) or 0),
                    )
                )
                _nd_appended = [
                    c for c in _nd_appended if c not in _X_before_nd_cols
                ]
                if _nd_appended:
                    X = X_nd
                    self.numeric_decompose_features_ = list(_nd_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_nd_appended)
                    )
                    for _r in _nd_recipes:
                        if _r.name in _nd_appended:
                            _numeric_decompose_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit numeric_decompose: appended %d engineered "
                            "column(s): %s",
                            len(_nd_appended), _nd_appended[:8],
                        )
            except Exception as _nd_exc:
                logger.warning(
                    "MRMR.fit numeric_decompose FE raised %s: %s; continuing "
                    "without numeric-decomposition columns.",
                    type(_nd_exc).__name__, _nd_exc,
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
                from ._periodic_fe import hybrid_modular_fe_with_recipes

                _y_for_md = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _md_periods = tuple(
                    getattr(self, "fe_modular_periods", (7, 12, 24, 30, 365))
                    or (7, 12, 24, 30, 365)
                )
                _md_top_k = int(getattr(self, "fe_modular_top_k", 6))
                _X_before_md_cols = list(X.columns)
                X_md, _md_appended, _md_recipes, _md_scores = (
                    hybrid_modular_fe_with_recipes(
                        X, _y_for_md,
                        cols=None,
                        periods=_md_periods,
                        top_k=_md_top_k,
                        seed=int(getattr(self, "random_seed", 0) or 0),
                    )
                )
                _md_appended = [
                    c for c in _md_appended if c not in _X_before_md_cols
                ]
                if _md_appended:
                    X = X_md
                    self.modular_features_ = list(_md_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_md_appended)
                    )
                    for _r in _md_recipes:
                        if _r.name in _md_appended:
                            _modular_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit modular: appended %d engineered "
                            "column(s): %s",
                            len(_md_appended), _md_appended[:8],
                        )
            except Exception as _md_exc:
                logger.warning(
                    "MRMR.fit modular FE raised %s: %s; continuing without "
                    "modular columns.",
                    type(_md_exc).__name__, _md_exc,
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
                from ._group_distance_fe import hybrid_group_distance_fe

                _y_for_gd = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _gd_groups = tuple(
                    getattr(self, "fe_group_distance_group_cols", ()) or ()
                )
                _gd_groups = [c for c in _gd_groups if c in X.columns] or None
                _gd_nums = tuple(
                    getattr(self, "fe_group_distance_num_cols", ()) or ()
                )
                _gd_nums = [c for c in _gd_nums if c in X.columns] or None
                _gd_top_k = int(getattr(self, "fe_group_distance_top_k", 6))
                _X_before_gd_cols = list(X.columns)
                X_gd, _gd_appended, _gd_recipes, _gd_scores = (
                    hybrid_group_distance_fe(
                        X, _y_for_gd,
                        group_cols=_gd_groups, num_cols=_gd_nums,
                        top_k=_gd_top_k,
                    )
                )
                _gd_appended = [
                    c for c in _gd_appended if c not in _X_before_gd_cols
                ]
                if _gd_appended:
                    X = X_gd
                    self.group_distance_features_ = list(_gd_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_gd_appended)
                    )
                    for _r in _gd_recipes:
                        if _r.name in _gd_appended:
                            _group_distance_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit group_distance: appended %d engineered "
                            "column(s): %s",
                            len(_gd_appended), _gd_appended[:8],
                        )
            except Exception as _gd_exc:
                logger.warning(
                    "MRMR.fit group_distance FE raised %s: %s; continuing "
                    "without group-distance columns.",
                    type(_gd_exc).__name__, _gd_exc,
                )

    # Layer 104 (2026-06-01): THREE new recipe-based FE families.
    self.rare_category_features_ = []
    self.conditional_residual_features_ = []
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
                from ._extra_fe_families import hybrid_rare_category_fe

                _y_for_rc = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _rc_cols = tuple(
                    getattr(self, "fe_rare_category_cols", ()) or ()
                )
                _rc_cols = [c for c in _rc_cols if c in X.columns] or None
                _X_before_rc_cols = list(X.columns)
                X_rc, _rc_appended, _rc_recipes, _ = hybrid_rare_category_fe(
                    X, _y_for_rc,
                    cat_cols=_rc_cols,
                    rare_threshold=float(
                        getattr(self, "fe_rare_category_threshold", 0.01)
                    ),
                    top_k=int(getattr(self, "fe_rare_category_top_k", 10)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                )
                _rc_appended = [
                    c for c in _rc_appended if c not in _X_before_rc_cols
                ]
                if _rc_appended:
                    X = X_rc
                    self.rare_category_features_ = list(_rc_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_rc_appended)
                    )
                    for _r in _rc_recipes:
                        if _r.name in _rc_appended:
                            _rare_category_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit rare_category: appended %d engineered "
                            "column(s): %s",
                            len(_rc_appended), _rc_appended[:8],
                        )
            except Exception as _rc_exc:
                logger.warning(
                    "MRMR.fit rare_category FE raised %s: %s; continuing "
                    "without rare-category columns.",
                    type(_rc_exc).__name__, _rc_exc,
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
                from ._extra_fe_families import hybrid_conditional_residual_fe

                _y_for_cr = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _cr_cols = tuple(
                    getattr(self, "fe_conditional_residual_cols", ()) or ()
                )
                _cr_cols = [c for c in _cr_cols if c in X.columns] or None
                _X_before_cr_cols = list(X.columns)
                X_cr, _cr_appended, _cr_recipes, _ = hybrid_conditional_residual_fe(
                    X, _y_for_cr,
                    num_cols=_cr_cols,
                    n_bins=int(getattr(self, "fe_conditional_residual_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_residual_top_k", 10)),
                    max_pair_cols=int(
                        getattr(self, "fe_conditional_residual_max_pair_cols", 6)
                    ),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                )
                _cr_appended = [
                    c for c in _cr_appended if c not in _X_before_cr_cols
                ]
                if _cr_appended:
                    X = X_cr
                    self.conditional_residual_features_ = list(_cr_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_cr_appended)
                    )
                    for _r in _cr_recipes:
                        if _r.name in _cr_appended:
                            _conditional_residual_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_residual: appended %d "
                            "engineered column(s): %s",
                            len(_cr_appended), _cr_appended[:8],
                        )
            except Exception as _cr_exc:
                logger.warning(
                    "MRMR.fit conditional_residual FE raised %s: %s; continuing "
                    "without conditional-residual columns.",
                    type(_cr_exc).__name__, _cr_exc,
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
                from ._extra_fe_families import hybrid_rankgauss_fe

                _y_for_rg = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _rg_cols = tuple(
                    getattr(self, "fe_rankgauss_cols", ()) or ()
                )
                _rg_cols = [c for c in _rg_cols if c in X.columns] or None
                _X_before_rg_cols = list(X.columns)
                X_rg, _rg_appended, _rg_recipes, _ = hybrid_rankgauss_fe(
                    X, _y_for_rg,
                    num_cols=_rg_cols,
                    top_k=int(getattr(self, "fe_rankgauss_top_k", 10)),
                )
                _rg_appended = [
                    c for c in _rg_appended if c not in _X_before_rg_cols
                ]
                if _rg_appended:
                    X = X_rg
                    self.rankgauss_features_ = list(_rg_appended)
                    self.hybrid_orth_features_ = (
                        list(self.hybrid_orth_features_ or []) + list(_rg_appended)
                    )
                    for _r in _rg_recipes:
                        if _r.name in _rg_appended:
                            _rankgauss_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit rankgauss: appended %d engineered "
                            "column(s): %s",
                            len(_rg_appended), _rg_appended[:8],
                        )
            except Exception as _rg_exc:
                logger.warning(
                    "MRMR.fit rankgauss FE raised %s: %s; continuing without "
                    "rankgauss columns.",
                    type(_rg_exc).__name__, _rg_exc,
                )

    # Layer 92 (2026-06-01): temporal leak-safe grouped aggregations. Keyed on
    # a time column, only ever seeing the strict past (expanding / rolling /
    # lag). Each survivor MI-gated against y; recipes store the fit-time
    # per-entity sorted history so transform() replays test rows against TRAIN
    # history only. Routing piggybacks on hybrid_orth_features_.
    if bool(getattr(self, "fe_temporal_agg_enable", False)):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 92 temporal_agg FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from ._temporal_agg_fe import hybrid_temporal_agg_fe

                _ta_time = getattr(self, "fe_temporal_agg_time_col", None)
                _ta_entities = [
                    c for c in (getattr(self, "fe_temporal_agg_entity_cols", ()) or ())
                    if c in X.columns
                ]
                _ta_values = [
                    c for c in (getattr(self, "fe_temporal_agg_value_cols", ()) or ())
                    if c in X.columns
                ]
                if _ta_time is None or _ta_time not in X.columns or not _ta_entities or not _ta_values:
                    if verbose:
                        logger.info(
                            "MRMR.fit temporal_agg: skipped (need time_col + "
                            "entity_cols + value_cols all present in X)."
                        )
                else:
                    _y_for_ta = (
                        y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                    )
                    if _y_for_ta.dtype.kind in "fc":
                        if int(np.unique(_y_for_ta).size) <= 32:
                            _y_for_ta = _y_for_ta.astype(np.int64)
                        else:
                            try:
                                _y_for_ta = pd.qcut(
                                    _y_for_ta, q=10, labels=False, duplicates="drop",
                                ).astype(np.int64)
                            except Exception:
                                _y_for_ta = _y_for_ta.astype(np.int64)
                    _ta_stats = tuple(
                        getattr(self, "fe_temporal_agg_stats", ())
                        or ("mean", "std", "count")
                    )
                    _ta_windows = tuple(getattr(self, "fe_temporal_agg_windows", ()) or ())
                    _ta_lags = tuple(getattr(self, "fe_temporal_agg_lags", (1,)) or ())
                    _ta_top_k = int(getattr(self, "fe_temporal_agg_top_k", 10))
                    _X_before_ta_cols = list(X.columns)
                    X_ta, _ta_appended, _ta_recipes, _ta_scores = hybrid_temporal_agg_fe(
                        X, _y_for_ta,
                        entity_cols=_ta_entities, value_cols=_ta_values,
                        time_col=_ta_time, stats=_ta_stats,
                        windows=_ta_windows, lags=_ta_lags, top_k=_ta_top_k,
                    )
                    _ta_appended = [
                        c for c in _ta_appended if c not in _X_before_ta_cols
                    ]
                    if _ta_appended:
                        X = X_ta
                        self.temporal_agg_features_ = list(_ta_appended)
                        self.hybrid_orth_features_ = (
                            list(self.hybrid_orth_features_ or []) + list(_ta_appended)
                        )
                        for _r in _ta_recipes:
                            if _r.name in _ta_appended:
                                _temporal_agg_pre_recipes[_r.name] = _r
                        if verbose:
                            logger.info(
                                "MRMR.fit temporal_agg: appended %d engineered "
                                "column(s): %s",
                                len(_ta_appended), _ta_appended[:8],
                            )
            except Exception as _ta_exc:
                logger.warning(
                    "MRMR.fit temporal_agg FE raised %s: %s; continuing without "
                    "temporal-aggregate columns.",
                    type(_ta_exc).__name__, _ta_exc,
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
    _eng_cols_appended_raw = list(self.hybrid_orth_features_ or []) + list(
        self.mi_greedy_features_ or []
    )
    _eng_seen: set[str] = set()
    _eng_cols_appended = [
        _c for _c in _eng_cols_appended_raw
        if not (_c in _eng_seen or _eng_seen.add(_c))
    ]
    # ADAPTIVE-FOURIER columns are NEVER pruned by the cross-stage dedup: the
    # held-out detector already validated the frequency, and a sin/cos pair at
    # one frequency is not monotone-equivalent to a fixed-grid twin, so the
    # Spearman gate would only ever drop them on a spurious near-tie. Keeping
    # them here guarantees they remain in ``cols`` for the protection block.
    _adaptive_fourier_keep = set(
        getattr(self, "_adaptive_fourier_features_", None) or []
    )
    if len(_eng_cols_appended) >= 2 and isinstance(X, pd.DataFrame):
        _eng_keep: list[str] = []
        _eng_drop: set[str] = set()
        _eng_arrs: dict[str, np.ndarray] = {}
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
                _eng_arrs[_c] = np.asarray(
                    _col_view_a.to_numpy(), dtype=np.float64
                )
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
            # Layer 38: mirror cleanup for ratio / log_ratio / grouped_delta / lagged_diff.
            self.pairwise_ratio_features_ = [
                c for c in (getattr(self, "pairwise_ratio_features_", []) or [])
                if c not in _eng_drop
            ]
            self.pairwise_log_ratio_features_ = [
                c for c in (getattr(self, "pairwise_log_ratio_features_", []) or [])
                if c not in _eng_drop
            ]
            self.grouped_delta_features_ = [
                c for c in (getattr(self, "grouped_delta_features_", []) or [])
                if c not in _eng_drop
            ]
            self.lagged_diff_features_ = [
                c for c in (getattr(self, "lagged_diff_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 87: mirror cleanup for grouped_agg.
            self.grouped_agg_features_ = [
                c for c in (getattr(self, "grouped_agg_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 93: mirror cleanup for composite_group_agg.
            self.composite_group_agg_features_ = [
                c for c in (getattr(self, "composite_group_agg_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 88: mirror cleanup for grouped_quantile.
            self.grouped_quantile_features_ = [
                c for c in (getattr(self, "grouped_quantile_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 89: mirror cleanup for cat_pair crosses.
            self.cat_pair_features_ = [
                c for c in (getattr(self, "cat_pair_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 94: mirror cleanup for cat_triple crosses.
            self.cat_triple_features_ = [
                c for c in (getattr(self, "cat_triple_features_", []) or [])
                if c not in _eng_drop
            ]
            # Layer 90: mirror cleanup for numeric-decomposition columns.
            self.numeric_decompose_features_ = [
                c for c in (getattr(self, "numeric_decompose_features_", []) or [])
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
            for _c in list(_group_distance_pre_recipes.keys()):
                if _c in _eng_drop:
                    _group_distance_pre_recipes.pop(_c, None)
            for _c in list(_rare_category_pre_recipes.keys()):
                if _c in _eng_drop:
                    _rare_category_pre_recipes.pop(_c, None)
            for _c in list(_conditional_residual_pre_recipes.keys()):
                if _c in _eng_drop:
                    _conditional_residual_pre_recipes.pop(_c, None)
            for _c in list(_rankgauss_pre_recipes.keys()):
                if _c in _eng_drop:
                    _rankgauss_pre_recipes.pop(_c, None)
            for _c in list(_temporal_agg_pre_recipes.keys()):
                if _c in _eng_drop:
                    _temporal_agg_pre_recipes.pop(_c, None)
            if verbose:
                logger.info(
                    "MRMR.fit engineered-FE dedup: pruned %d near-duplicate "
                    "engineered column(s) at Spearman |rho| >= 0.99: %s",
                    len(_eng_drop), sorted(_eng_drop),
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
    if (
        bool(getattr(self, "fe_unified_second_pass_gate", False))
        and isinstance(X, pd.DataFrame)
    ):
        try:
            _eng_now = [
                c for c in (
                    list(self.hybrid_orth_features_ or [])
                    + list(self.mi_greedy_features_ or [])
                )
                if c in X.columns
            ]
            # Order-preserving unique.
            _seen_u: set[str] = set()
            _eng_now = [
                c for c in _eng_now if not (c in _seen_u or _seen_u.add(c))
            ]
            if len(_eng_now) >= 2:
                from ._unified_fe_gate import unified_second_pass_gate

                _raw_cols_u = [c for c in X.columns if c not in set(_eng_now)]
                _y_for_u = (
                    y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
                )
                _keep_u = set(unified_second_pass_gate(
                    X, _y_for_u,
                    raw_cols=_raw_cols_u,
                    engineered_cols=_eng_now,
                    max_keep=getattr(self, "fe_unified_second_pass_max_keep", None),
                    min_cmi_gain=float(
                        getattr(self, "fe_unified_second_pass_min_gain", 0.005)
                    ),
                ))
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
                        "conditional_residual_features_", "rankgauss_features_",
                        "temporal_agg_features_",
                    ):
                        setattr(self, _attr, [
                            c for c in (getattr(self, _attr, []) or [])
                            if c not in _eng_drop_u
                        ])
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
                        _modular_pre_recipes, _group_distance_pre_recipes,
                        _rare_category_pre_recipes,
                        _conditional_residual_pre_recipes,
                        _rankgauss_pre_recipes,
                        _temporal_agg_pre_recipes,
                    ):
                        for _c in list(_pre.keys()):
                            if _c in _eng_drop_u:
                                _pre.pop(_c, None)
                    if verbose:
                        logger.info(
                            "MRMR.fit unified second-pass CMI gate: pruned %d "
                            "cross-mechanism redundant engineered column(s): %s",
                            len(_eng_drop_u), sorted(_eng_drop_u),
                        )
        except Exception as _u_exc:
            logger.warning(
                "MRMR.fit unified_second_pass_gate raised %s: %s; continuing "
                "without the Tier-2 cross-mechanism gate.",
                type(_u_exc).__name__, _u_exc,
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
    self.feature_names_in_ = [c for c in _all_cols if c not in _engineered_names_set]
    self.n_features_in_ = len(self.feature_names_in_)

    # Accuracy gate (single point, after every setup FE stage, before screening): drop engineered columns that add NO held-out downstream uplift over their raw source. Covers every default-on FE stage (univariate basis He, adaptive Fourier/chirp sin/qsin, ...) in one place using the raw values in X + the real target y. A hijacker (transform of a monotone / MNAR / leak column) is removed so it cannot evict the raw signal from support_; a genuine win (He2 on a quadratic target, adaptive Fourier on an oscillation) keeps its positive uplift and stays. Also un-protects dropped adaptive Fourier names + drops their recipes. Opt out via fe_accuracy_gate=False.
    if _engineered_names_set and bool(getattr(self, "fe_accuracy_gate", True)) and isinstance(X, pd.DataFrame):
        try:
            from ._fe_accuracy_gate import keep_engineered_over_source
            _gy = _target_to_numpy_values(y)
            _rawset = set(self.feature_names_in_)
            _xset = set(X.columns)
            _drop_eng = []
            for _en in list(_engineered_names_set):
                _s = str(_en).split("__", 1)[0] if "__" in str(_en) else None
                if _s is None or _s not in _rawset or _en not in _xset or _s not in _xset:
                    continue
                if not keep_engineered_over_source(np.asarray(X[_s].to_numpy()), np.asarray(X[_en].to_numpy()), _gy):
                    _drop_eng.append(_en)
            if _drop_eng:
                _dset = set(_drop_eng)
                X = X.drop(columns=_drop_eng)
                _all_cols = [c for c in _all_cols if c not in _dset]
                _engineered_names_set = _engineered_names_set - _dset
                self.hybrid_orth_features_ = [c for c in (getattr(self, "hybrid_orth_features_", None) or []) if c not in _dset]
                self._adaptive_fourier_features_ = [c for c in (getattr(self, "_adaptive_fourier_features_", None) or []) if c not in _dset]
                for _dn in _drop_eng:
                    _hybrid_orth_pre_recipes.pop(_dn, None)
                if verbose:
                    logger.info("MRMR accuracy gate: dropped %d engineered column(s) with no held-out uplift over their raw source: %s", len(_drop_eng), _drop_eng[:8])
        except Exception as _agexc:
            logger.debug("MRMR accuracy gate (setup) failed (%s); keeping all engineered", _agexc)

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
        cache_dir=getattr(self, "cache_dir", None),
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
    # Layer 95 PART B: same routing for per-group distribution-distance recipes.
    if _group_distance_pre_recipes:
        engineered_recipes.update(_group_distance_pre_recipes)
    # Layer 104: rare-category / conditional-residual / rankgauss recipes.
    if _rare_category_pre_recipes:
        engineered_recipes.update(_rare_category_pre_recipes)
    if _conditional_residual_pre_recipes:
        engineered_recipes.update(_conditional_residual_pre_recipes)
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
    # 2026-06-02: tracks whether the post-FE confirming re-screen has run, so it
    # fires at most once (see the fe_reselect_after_engineering block below). The
    # re-screen re-selects from the augmented pool (raw + engineered) using the
    # estimator's own use_simple_mode (now defaulting to False = full Fleuret
    # conditional-MI redundancy), which is what drops engineered columns redundant
    # given an already-selected one and records a real gain for every survivor.
    _did_confirm_rescreen = False
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
                        # 2026-06-03 (audit dcd-core-1/dcd-swap-null-1/2):
                        # the swap null draw count, decoupled from
                        # full_npermutations. getattr fallback keeps old
                        # pickles (lacking the attr) loading at the 199 default.
                        swap_npermutations=getattr(self, "dcd_swap_npermutations", 199),
                        # Layer 47 (2026-05-31): forward the auto-tau
                        # calibration knobs (number of sampled feature pairs
                        # and RNG seed) so make_dcd_state can fingerprint
                        # the calibration sweep deterministically.
                        tau_calibration_n_pairs=getattr(
                            self, "dcd_tau_calibration_n_pairs", 100,
                        ),
                        tau_calibration_seed=getattr(
                            self, "dcd_tau_calibration_seed", 0,
                        ),
                        X_raw=X,
                        quantization_method=self.quantization_method,
                        quantization_nbins=self.quantization_nbins,
                        quantization_dtype=self.quantization_dtype,
                    )
                    if getattr(self, "dcd_enable", False) else None
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
            )
        )
        # 2026-05-30 Wave 9 — stash DCD summary on the estimator for the
        # public ``dcd_`` attribute (None when DCD was disabled).
        try:
            from ._dynamic_cluster_discovery import dcd_summary as _dcd_summary
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
                from ._cluster_hierarchy import build_cluster_hierarchy
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
        _n_cols_before_fe = len(cols)  # accuracy gate: engineered columns added this step live at indices >= this
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

        # Accuracy gate (post-FE-step): drop engineered candidates that do NOT add held-out downstream uplift over their raw source column. The scorer routing inside _run_fe_step already ran, so this only filters which engineered columns COMPETE in the screen -- a hijacker (Fourier/chirp/He/unary of a monotone / MNAR / leak column) is removed so it cannot evict the raw signal from support_, while a genuine win (He2 on a quadratic target, the routed adaptive Fourier on an oscillation) keeps its positive uplift and stays. Probes raw values from X over the injected target. Opt out via fe_accuracy_gate=False.
        if len(cols) > _n_cols_before_fe and target_names and bool(getattr(self, "fe_accuracy_gate", True)) and hasattr(X, "columns"):
            try:
                from ._fe_accuracy_gate import keep_engineered_over_source
                _gate_y = np.asarray(X[target_names[0]].to_numpy())   # raw target (probe label)
                _raw_set = set(self.feature_names_in_)
                _sv_set = set(int(s) for s in selected_vars)
                _col_to_idx = {str(c): k for k, c in enumerate(cols)}
                _data_arr = np.asarray(data, dtype=np.float64)
                _drop_idx = []
                for _i in range(_n_cols_before_fe, len(cols)):
                    if _i in _sv_set:
                        continue
                    _nm = str(cols[_i])
                    _src = _nm.split("__", 1)[0] if "__" in _nm else None
                    if _src is None or _src not in _raw_set:
                        continue
                    _src_idx = _col_to_idx.get(_src)
                    if _src_idx is None:
                        continue
                    if not keep_engineered_over_source(_data_arr[:, _src_idx], _data_arr[:, _i], _gate_y):
                        _drop_idx.append(_i)
                if _drop_idx:
                    _drop = set(_drop_idx)
                    _ds = sorted(_drop)
                    _drop_names = [str(cols[j]) for j in _drop_idx]
                    _keep = [j for j in range(len(cols)) if j not in _drop]
                    _x_cols = set(X.columns)
                    X = X.drop([c for c in _drop_names if c in _x_cols]) if _is_polars_input else X.drop(columns=[c for c in _drop_names if c in _x_cols])
                    data = data[:, _keep]
                    nbins = nbins[_keep] if isinstance(nbins, np.ndarray) else [nbins[j] for j in _keep]
                    cols = [cols[j] for j in _keep]
                    selected_vars = [int(s) - sum(1 for d in _ds if d < int(s)) for s in selected_vars]
                    n_recommended_features = max(0, n_recommended_features - len(_drop_idx))
                    if verbose:
                        logger.info("MRMR accuracy gate: dropped %d engineered candidate(s) with no held-out uplift over their raw source: %s", len(_drop_idx), _drop_names[:8])
            except Exception as _agexc:
                logger.debug("MRMR accuracy gate failed (%s); keeping all engineered candidates", _agexc)

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
            if (
                getattr(self, "fe_reselect_after_engineering", True)
                and n_recommended_features > 0
                and not _did_confirm_rescreen
            ):
                _did_confirm_rescreen = True
                continue
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
        from ._confirm_predictor import _extract_single_raw_parent  # noqa: E501
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
        _sv_set = set(selected_vars)
        _readd = []
        for _rn in _prefe_raw:
            if _rn in _cur_names or _rn in _substituted:
                continue
            try:
                _idx = cols.index(_rn)
            except ValueError:
                continue
            if _idx not in _sv_set:
                _readd.append(_idx)
                _sv_set.add(_idx)
        if _readd:
            selected_vars = list(selected_vars) + _readd
            if verbose:
                logger.info(
                    "MRMR raw-retention: re-added %d screening-confirmed raw feature(s) "
                    "dropped by the post-FE re-selection (not substituted by a sole-parent "
                    "engineered child): %s",
                    len(_readd), [cols[i] for i in _readd],
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
                    "MRMR adaptive-fourier protection: re-added %d held-out-"
                    "validated adaptive Fourier feature(s) dropped by the screen: %s",
                    len(_readd_adaptive), [cols[i] for i in _readd_adaptive],
                )

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
            _sel_names = set(X.columns[selected_vars].tolist())
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
            temp_columns = [c for c in X.columns if c not in _sel_names and c not in _excluded_from_rescue]

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

    # ``selected_vars`` holds integer column indices. Force int64 dtype so an
    # EMPTY selection (all signal folded into engineered recipes under the
    # full-mode default -> zero raw survivors) stays an integer index array.
    # ``np.array([])`` defaults to float64, and the ndarray transform path
    # (``X[:, support_]`` in _mrmr_validate_transform) then raises
    # ``IndexError: arrays used as indices must be of integer (or boolean)
    # type`` because a float array can't index. Integer dtype makes the empty
    # slice a valid no-op on both the DataFrame and the ndarray paths.
    self.support_ = np.array(selected_vars, dtype=np.int64)
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
            from ._mrmr_artifacts import compute_mrmr_artifacts
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
                "MRMR.retain_artifacts: capture failed (%s); export_artifacts() will raise. "
                "Cause: %s",
                type(_exc).__name__, _exc,
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
