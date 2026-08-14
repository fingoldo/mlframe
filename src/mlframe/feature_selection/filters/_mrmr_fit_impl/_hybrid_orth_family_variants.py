"""Split off ``mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core`` for the sub-split
that brings ``_fit_impl_core.py`` below the project's 1k-LOC module-size gate.

Holds ``_hybrid_orth_family_variants``: the block of ~19 ``fe_hybrid_orth_*_enable``-gated FE
family stages inside ``MRMR._fit_impl`` (triplet, quadruplet, adaptive-arity, adaptive-degree,
conditional-routing, diff-basis, cluster-basis, bootstrap, three-gate, KSG, copula, dCor, HSIC,
JMIM, TC, CMIM, auto-scorer, ensemble, meta) -- one contiguous run of near-identical-shaped blocks
between the initial ``_gbm_seeded_triplet_names`` setup and the (separately-gated) MI-greedy FE
stage that follows.

Threads ``self`` plus every fit-body local this section reads as explicit keyword arguments
(mirrors the other sub-split carve-outs' own pattern), derived via ``pyutilz.dev.freevar_analysis``.
Unlike the cols-space sections (``_assign_support``, ``_friend_graph_and_redundancy``), this
section operates entirely on ``X`` (the raw/engineered pandas-or-polars frame, format-agnostic via
the matrix-native FE seam) -- each family stage appends its own winning columns onto ``X`` via
``fe_append_columns``/``fe_extract_columns`` and records recipes into ``_hybrid_orth_pre_recipes``
(a dict, mutated in place) plus ``self.hybrid_orth_features_`` (a list, reassigned in place via
``self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + [...]`` at each site). Like
``_friend_graph_and_redundancy``'s ``selected_vars``, ``X`` is passed in AND returned: confirmed by
grepping every ``X = fe_append_columns(X, ...)`` reassignment in range (19 sites, one per family)
and that the code immediately following this section keeps reading/mutating ``X``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._helpers import _orth_fe_numeric_cols, fe_decide_on_subsample

logger = logging.getLogger(__name__)


def _hybrid_orth_family_variants(
    self,
    *,
    X,
    y,
    verbose,
    _y_np,
    _hybrid_orth_pre_recipes,
    _gbm_seeded_triplet_names,
    _fe_family_on,
):
    """Run every ``fe_hybrid_orth_*_enable``-gated FE family stage and return the (possibly
    column-augmented) ``X``.

    See the module docstring for the full section this carves out.
    """
    if _fe_family_on("fe_hybrid_orth_triplet_enable", False) or _gbm_seeded_triplet_names:
        # Format-agnostic since the matrix-native FE seam: the isinstance(X, pd.DataFrame) skip-guard is gone - the family
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
    if _fe_family_on("fe_hybrid_orth_diff_basis_enable", False):
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the diff-basis FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_diff = _y_for_diff.astype(np.int64)
            # Restrict the seed pool to RAW source columns - engineered
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
    if _fe_family_on("fe_hybrid_orth_cluster_basis_enable", False):
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
                """Reject-sink callback for the per-cluster shared-basis FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the cluster-basis FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
                        _y_for_cb = _y_for_cb.astype(np.int64)
            # Restrict to RAW source columns - engineered columns from
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
    # engineered columns are bit-equal to Layer 21 - only the SELECTION
    # changes - so recipes reuse the ``orth_univariate`` kind and replay
    # is shared. Restrict to RAW columns to avoid recipes referencing
    # already-engineered columns absent at transform.
    if _fe_family_on("fe_hybrid_orth_bootstrap_enable", False):
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the bootstrap-MI FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
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
    # selected). When ``current_support`` is empty Gate 3 is skipped -
    # marginal MI from Gate 1 already covers that case. Engineered VALUES
    # are bit-equal to Layer 21 so recipes reuse the ``orth_univariate``
    # kind and replay is shared infrastructure.
    if _fe_family_on("fe_hybrid_orth_three_gate_enable", False):
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the three-gate FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
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
                """Adapt the three-gate FE generator to the subsample-funnel callback signature, threading the current_support sub-frame (columns already appended by earlier hybrid stages) through as Gate 3's conditioning set."""
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
    # engineered columns are bit-equal to Layer 21 - only the SCORING
    # (and therefore the selection) changes - so recipes reuse the
    # ``orth_univariate`` kind and replay is shared infrastructure.
    if _fe_family_on("fe_hybrid_orth_ksg_enable", False):
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
                    except Exception as exc:
                        logger.debug("mrmr: y densification failed for the KSG-MI FE seed pool; falling back to truncating int64 cast: %r", exc, exc_info=True)
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
    if _fe_family_on("fe_hybrid_orth_copula_enable", False):
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
            # (1.05 / 0.1) are too tight here - copula MI lift on a
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
    # dependence measure in the layer family - ``dCor == 0`` iff X and Y
    # are independent on ANY relationship (Pearson lacks this iff
    # guarantee). Naive dCor is O(n^2); the working sample is capped at
    # n=500 via deterministic random subsample. Engineered VALUES bit-equal
    # to Layer 21 -> recipes reuse the ``orth_univariate`` kind.
    if _fe_family_on("fe_hybrid_orth_dcor_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_hsic_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_jmim_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_tc_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_cmim_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_auto_scorer_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_ensemble_enable", False):
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
    if _fe_family_on("fe_hybrid_orth_meta_enable", False):
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

    return X
