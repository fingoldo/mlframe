"""Accuracy and second-pass CMI gates applied to engineered columns in ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import drop_from_fe_rosters
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

# --- end imports ---


def _second_pass_cmi_gate(self, X, _y_np, recipes, verbose):
    """Tier-2 unified second-pass CMI gate over every engineered column (catches cross-family near-duplicates the rank dedup misses)."""
    if bool(getattr(self, "fe_unified_second_pass_gate", False)) and isinstance(X, pd.DataFrame):
        try:
            _eng_now = [c for c in (list(self.hybrid_orth_features_ or []) + list(self.mi_greedy_features_ or [])) if c in X.columns]
            # Order-preserving unique.
            _seen_u: set[str] = set()
            _eng_now = [c for c in _eng_now if not (c in _seen_u or _seen_u.add(c))]  # type: ignore[func-returns-value]  # intentional order-preserving-dedup idiom: set.add()'s None return is used as the falsy side of `or`
            if len(_eng_now) >= 2:
                from mlframe.feature_selection.filters._unified_fe_gate import unified_second_pass_gate

                _eng_now_set = set(_eng_now)
                _raw_cols_u = [c for c in X.columns if c not in _eng_now_set]
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
                    drop_from_fe_rosters(self, _eng_drop_u)
                    # Private hinge / adaptive-fourier protection rosters are not
                    # in the public-roster loop above; prune them explicitly so a
                    # second-pass-dropped leg is not re-added by its protection.
                    self._hinge_features_ = [c for c in (getattr(self, "_hinge_features_", None) or []) if c not in _eng_drop_u]
                    self._adaptive_fourier_features_ = [c for c in (getattr(self, "_adaptive_fourier_features_", None) or []) if c not in _eng_drop_u]
                    for _pre in (
                        recipes.hybrid_orth,
                        recipes.mi_greedy,
                        recipes.kfold_te,
                        recipes.count_enc,
                        recipes.freq_enc,
                        recipes.cat_num,
                        recipes.miss_ind,
                        recipes.miss_cnt,
                        recipes.miss_pat,
                        recipes.ratio,
                        recipes.log_ratio,
                        recipes.grouped_delta,
                        recipes.lagged_diff,
                        recipes.grouped_agg,
                        recipes.composite_group_agg,
                        recipes.grouped_quantile,
                        recipes.cat_pair,
                        recipes.cat_triple,
                        recipes.numeric_decompose,
                        recipes.modular,
                        recipes.pairwise_modular,
                        recipes.integer_lattice,
                        recipes.row_argmax,
                        recipes.conditional_gate,
                        recipes.group_distance,
                        recipes.rare_category,
                        recipes.conditional_residual,
                        recipes.conditional_dispersion,
                        recipes.wavelet,
                        recipes.rankgauss,
                        recipes.temporal_agg,
                        recipes.conditional_quantile_rank,
                        recipes.ordinal_pattern,
                        recipes.random_fourier,
                        recipes.sir_direction,
                        recipes.lof,
                        recipes.mahalanobis_density,
                        recipes.offset_product,
                        recipes.oof_warp,
                        recipes.row_stat,
                    ):
                        for _c in list(_pre.keys()):
                            if _c in _eng_drop_u:
                                _pre.pop(_c, None)
                    if verbose:
                        logger.info(
                            "MRMR.fit unified second-pass CMI gate: pruned %d cross-mechanism redundant engineered column(s): %s",
                            len(_eng_drop_u),
                            sorted(_eng_drop_u),
                        )
        except Exception as _u_exc:
            logger.warning(
                "MRMR.fit unified_second_pass_gate raised %s: %s; continuing without the Tier-2 cross-mechanism gate.",
                type(_u_exc).__name__,
                _u_exc,
            )
    return X


def _gate_col_arr(X, _name):
    """Fetch column ``_name`` from ``X`` as a 1-D array for the held-out linear-probe accuracy gate (unwraps a duplicate-label DataFrame slice to its first column)."""
    _v = X[_name]
    if isinstance(_v, pd.DataFrame):
        _v = _v.iloc[:, 0]
    return np.asarray(_v.to_numpy(), dtype=_crit_np_dtype())


def _resolve_gate_columns(X, recipes, candidates, hybrid_set, adaptive_set):
    """Map each orth engineered column to its single raw source; flag the adaptive-Fourier / chirp columns."""
    gate_cols: list[tuple[str, str, bool]] = []
    for _gc in list(candidates):
        if _gc not in X.columns:
            continue
        _rec = recipes.hybrid_orth.get(_gc)
        # No hybrid-orth recipe => not an orth_* engineered column (missingness / TE / count / etc.): exempt.
        _src_names = tuple(getattr(_rec, "src_names", ()) or ()) if _rec is not None else ()
        if len(_src_names) != 1:
            continue
        _src = _src_names[0]
        if _src not in X.columns or _src in hybrid_set:
            continue
        _is_fourier = (_gc in adaptive_set) or (str(getattr(_rec, "kind", "")) == "orth_fourier")
        gate_cols.append((_gc, _src, _is_fourier))
    return gate_cols


def _gate_base_columns(X, gate_cols, y, seed):
    """Pass 1: keep a non-Fourier engineered column only if it beats its raw source on the held-out probe.

    Returns the surviving base columns keyed by raw source (the siblings pass 2 conditions on) and the dropped names.
    """
    from mlframe.feature_selection.filters._fe_accuracy_gate import keep_engineered_over_source

    surviving_by_src: dict[str, list[str]] = {}
    dropped: list[str] = []
    for _gc, _src, _is_fourier in gate_cols:
        if _is_fourier:
            continue
        _src_arr = _gate_col_arr(X, _src)
        _eng_arr = _gate_col_arr(X, _gc)
        if keep_engineered_over_source(_src_arr, _eng_arr, y, seed=seed):
            surviving_by_src.setdefault(_src, []).append(_gc)
        else:
            dropped.append(_gc)
    return surviving_by_src, dropped


def _gate_fourier_columns(X, gate_cols, surviving_by_src, y, seed, groups, classification):
    """Pass 2: gate each adaptive-Fourier / chirp column on the uplift over its raw source plus that source's surviving base siblings.

    A Fourier redundant with a He2 sibling (both encode x**2) adds ~0 here and is dropped; a genuine oscillation no polynomial sibling captures clears the floor and is kept.
    MNAR fail-closed first (the probe drops NaN rows).
    """
    from mlframe.feature_selection.filters._fe_accuracy_gate import _FE_UPLIFT_MIN, measure_feature_uplift

    dropped: list[str] = []
    for _gc, _src, _is_fourier in gate_cols:
        if not _is_fourier:
            continue
        _src_arr = _gate_col_arr(X, _src)
        if float(np.mean(~np.isfinite(_src_arr))) > 0.02:
            dropped.append(_gc)
            continue
        _base_sibs = surviving_by_src.get(_src, [])
        _base_mat = np.column_stack([_src_arr] + [_gate_col_arr(X, _b) for _b in _base_sibs])
        _eng_arr = _gate_col_arr(X, _gc)
        _n = _base_mat.shape[0]
        if _n > 5000:
            _rng_g = np.random.default_rng(seed)
            _idx_g = _rng_g.choice(_n, 5000, replace=False)
            _base_probe, _eng_probe, _y_probe = _base_mat[_idx_g], _eng_arr[_idx_g], y[_idx_g]
            _g_probe = groups[_idx_g] if groups is not None else None
        else:
            _base_probe, _eng_probe, _y_probe = _base_mat, _eng_arr, y
            _g_probe = groups
        _cond_uplift = measure_feature_uplift(
            _base_probe,
            _eng_probe,
            _y_probe,
            classification=classification,
            seed=seed,
            groups=_g_probe,
        )
        # Fail-open: None == probe could not measure (degenerate / exception);
        # keep the candidate rather than silently dropping it. Only a genuine
        # MEASURED sub-threshold uplift evicts.
        if _cond_uplift is not None and _cond_uplift < _FE_UPLIFT_MIN:
            dropped.append(_gc)
    return dropped


def _apply_gate_drops(self, X, recipes, gate_drop, verbose):
    """Remove the gated columns from ``X``, the fit's engineered rosters, the hinge roster and the recipe registry."""
    gate_drop_set = set(gate_drop)
    X = X.drop(columns=[c for c in gate_drop if c in X.columns])
    self.hybrid_orth_features_ = [c for c in (self.hybrid_orth_features_ or []) if c not in gate_drop_set]
    self._adaptive_fourier_features_ = [c for c in (getattr(self, "_adaptive_fourier_features_", None) or []) if c not in gate_drop_set]
    # Mirror the cleanup for hinge legs: a hinge the accuracy gate
    # drops (no held-out uplift over its raw source) must NOT be
    # re-added by the HINGE-PROTECTION block, so prune it here too.
    self._hinge_features_ = [c for c in (getattr(self, "_hinge_features_", None) or []) if c not in gate_drop_set]
    for _c in list(recipes.hybrid_orth.keys()):
        if _c in gate_drop_set:
            recipes.hybrid_orth.pop(_c, None)
    if verbose:
        logger.info(
            "MRMR.fit accuracy gate: dropped %d engineered column(s) adding no held-out uplift over their raw source (or MNAR source): %s",
            len(gate_drop),
            sorted(gate_drop),
        )
    return X


def _gate_engineered_accuracy(self, X, recipes, _y_np, verbose, groups=None):
    """Accuracy gate: drop an engineered column that adds no held-out linear-probe uplift over its raw source."""
    if bool(getattr(self, "fe_accuracy_gate", True)) and isinstance(X, pd.DataFrame) and (self.hybrid_orth_features_ or []) and recipes.hybrid_orth:
        try:
            from mlframe.feature_selection.filters._fe_accuracy_gate import infer_classification

            _y_for_gate = _y_np
            _gate_groups = np.asarray(groups).ravel() if groups is not None and len(groups) == len(_y_np) else None
            _gate_seed = int(getattr(self, "random_seed", 0) or 0)
            _gate_classif = infer_classification(_y_for_gate)
            _hybrid_set_now = set(self.hybrid_orth_features_ or [])
            _adaptive_set_now = set(getattr(self, "_adaptive_fourier_features_", None) or [])
            # Resolve each engineered column to its single raw source; split into the polynomial/base columns and the adaptive-Fourier/chirp columns (the latter are gated CONDITIONALLY
            # against their surviving base siblings, since a Fourier of x captures the SAME x**2 signal as its He2 sibling and must not dilute the support when the He2 already carries it).
            _gate_cols = _resolve_gate_columns(X, recipes, self.hybrid_orth_features_ or [], _hybrid_set_now, _adaptive_set_now)
            _surviving_base_by_src, _gate_drop = _gate_base_columns(X, _gate_cols, _y_for_gate, _gate_seed)
            _gate_drop = _gate_drop + _gate_fourier_columns(X, _gate_cols, _surviving_base_by_src, _y_for_gate, _gate_seed, _gate_groups, _gate_classif)
            if _gate_drop:
                X = _apply_gate_drops(self, X, recipes, _gate_drop, verbose)
        except Exception as _gate_exc:
            logger.warning(
                "MRMR.fit accuracy gate raised %s: %s; continuing without the accuracy gate (engineered columns kept).",
                type(_gate_exc).__name__,
                _gate_exc,
            )
    return X
