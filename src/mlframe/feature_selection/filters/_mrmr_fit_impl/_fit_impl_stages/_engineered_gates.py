"""Accuracy and second-pass CMI gates applied to engineered columns in ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._mrmr_fit_impl._fe_roster_attrs import FE_ROSTER_ATTRS
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger

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
                    for _attr in FE_ROSTER_ATTRS:
                        setattr(self, _attr, [c for c in (getattr(self, _attr, []) or []) if c not in _eng_drop_u])
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


def _gate_engineered_accuracy(self, X, recipes, _y_np, verbose):
    """Accuracy gate: drop an engineered column that adds no held-out linear-probe uplift over its raw source."""
    if bool(getattr(self, "fe_accuracy_gate", True)) and isinstance(X, pd.DataFrame) and (self.hybrid_orth_features_ or []) and recipes.hybrid_orth:
        try:
            from mlframe.feature_selection.filters._fe_accuracy_gate import (
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
                _rec = recipes.hybrid_orth.get(_gc)
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
                    _base_probe,
                    _eng_probe,
                    _y_probe,
                    classification=_gate_classif,
                    seed=_gate_seed,
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
                for _c in list(recipes.hybrid_orth.keys()):
                    if _c in _gate_drop_set:
                        recipes.hybrid_orth.pop(_c, None)
                if verbose:
                    logger.info(
                        "MRMR.fit accuracy gate: dropped %d engineered column(s) adding no held-out uplift over their raw source (or MNAR source): %s",
                        len(_gate_drop),
                        sorted(_gate_drop),
                    )
        except Exception as _gate_exc:
            logger.warning(
                "MRMR.fit accuracy gate raised %s: %s; continuing without the accuracy gate (engineered columns kept).",
                type(_gate_exc).__name__,
                _gate_exc,
            )
    return X
