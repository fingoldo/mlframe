"""Sibling of ``_fit_impl_core.py`` (part of the sub-split that brings the parent below
the project's 1k-LOC module-size gate).

Holds ``_fe_stage_cascade_mid_a``: Layer 87 (grouped multi-stat aggregator), Layer 93
(composite multi-column group-key aggregates), Layer 88 (per-group histogram + quantile FE),
Layer 89 (cat x cat synergy cross), Layer 94 (cat x cat x cat triple synergy cross), Layer 90
(numeric decomposition), Layer 95 PART A (periodic/modular decomposition). Every FE family
stage here reads the (possibly already-augmented) ``X`` and appends its own winning
engineered columns via ``fe_append_columns``/``fe_extract_columns`` -- mirrors the sibling
cascade modules' own ``X``-in-``X``-out contract.

All ``_*_pre_recipes`` dicts are caller-owned and mutated in place (never reassigned here --
confirmed via a systematic reassignment-vs-mutation check), so no return is needed for them.
``_raw_input_cols_pre_fe`` is read-only here (computed by an earlier cascade sibling).
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _fe_stage_cascade_mid_a(
    self,
    *,
    X,
    y,
    verbose,
    fe_max_steps,
    _y_np,
    _fe_family_on,
    _fe_budget_ok,
    _raw_input_cols_pre_fe,
    _cat_pair_pre_recipes,
    _cat_triple_pre_recipes,
    _composite_group_agg_pre_recipes,
    _conditional_gate_pre_recipes,
    _grouped_agg_pre_recipes,
    _grouped_quantile_pre_recipes,
    _integer_lattice_pre_recipes,
    _modular_pre_recipes,
    _numeric_decompose_pre_recipes,
    _pairwise_modular_pre_recipes,
    _row_argmax_pre_recipes,
):
    """Run the Layer 87/93/88/89/94/90/95A FE family stage(s) and return the (possibly
    column-augmented) ``X``. See the module docstring for the full section this carves out.
    """
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
                    from .._grouped_coerce_shared import auto_detect_group_cols as _gq_detect_groups_impl
                    from .._grouped_quantile_fe import (
                        _auto_detect_num_cols as _gq_detect_nums,
                    )
                    _gq_raw_view = X[[c for c in X.columns if c in _gq_raw]]
                    if _gq_groups is None:
                        _gq_groups = _gq_detect_groups_impl(_gq_raw_view, caller="grouped_quantile") or None
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

    return X
