"""Sibling of ``_hybrid_orth_family_variants/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
hybrid_orth family-variant block, itself further split for the 1k-LOC module-size gate).

Holds families: triplet, quadruplet, adaptive_arity, adaptive_degree, conditional_routing. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``self``/``X`` threading contract (mirrors the parent's own).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .._helpers import _orth_fe_numeric_cols, fe_decide_on_subsample

logger = logging.getLogger(__name__)


def _hybrid_orth_family_variants_group1(
    self, *, X, y, verbose, _y_np, _hybrid_orth_pre_recipes, _gbm_seeded_triplet_names, _fe_family_on,
):
    """Run the triplet, quadruplet, adaptive_arity, adaptive_degree, conditional_routing hybrid_orth family stage(s) and return the (possibly
    column-augmented) ``X``. See the package docstring for the full section this carves out."""
    if _fe_family_on("fe_hybrid_orth_triplet_enable", False) or _gbm_seeded_triplet_names:
        # Format-agnostic since the matrix-native FE seam: the isinstance(X, pd.DataFrame) skip-guard is gone - the family
        # runs on polars/pandas alike (subsample decision + native replay via fe_decide_on_subsample / _fe_frame_ops).
        try:
            from ..._orthogonal_triplet_fe import (
                hybrid_orth_mi_triplet_fe_with_recipes,
            )
            from ..._fe_frame_ops import fe_is_numeric_col, fe_append_columns, fe_extract_columns

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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the triplet FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_triplet = _y_for_triplet.astype(np.int64)
            # Triplet seed pool is restricted to RAW columns - never
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
            # features, not univariate transforms of the seeded operands - on a pure-noise
            # frame the seeded noise triples' univariate stage would otherwise engineer a
            # spurious univariate Fourier/poly on a noise operand (a noise admission). When
            # the user ALSO enabled the legacy triplet path, keep their univariate budget.
            _t_top_k_eff = _t_top_k
            if _explicit_triplets is not None and not _fe_family_on("fe_hybrid_orth_triplet_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_quadruplet_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_quadruplet_fe import (
                hybrid_orth_mi_quadruplet_fe_with_recipes,
            )
            from ..._fe_frame_ops import fe_is_numeric_col, fe_append_columns, fe_extract_columns

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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the quadruplet FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_quad = _y_for_quad.astype(np.int64)
            # Restrict the seed pool to RAW source columns - engineered
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
    if _fe_family_on("fe_hybrid_orth_adaptive_arity_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_adaptive_arity_fe import (
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the adaptive-arity FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
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
            # Only keep TRUE cross columns (arity >= 2 - one or more '*').
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
    # ``orth_univariate`` - replay reads X only, no y.
    if _fe_family_on("fe_hybrid_orth_adaptive_degree_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_adaptive_degree_fe import (
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the adaptive-degree FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_adapt = _y_for_adapt.astype(np.int64)
            # Restrict the seed pool to RAW source columns - engineered
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
    if _fe_family_on("fe_hybrid_orth_conditional_routing_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_routing_fe import (
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the conditional-routing FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_route = _y_for_route.astype(np.int64)
            # Restrict the seed pool to RAW source columns - engineered
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

    return X
