"""Sibling of ``_hybrid_orth_family_variants/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
hybrid_orth family-variant block, itself further split for the 1k-LOC module-size gate).

Holds families: copula, dcor, hsic, jmim, tc. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``self``/``X`` threading contract (mirrors the parent's own).
"""

from __future__ import annotations

import logging


from .._helpers import fe_decide_on_subsample
from ..._fe_frame_ops import fe_append_columns, fe_extract_columns

logger = logging.getLogger(__name__)


def _hybrid_orth_family_variants_group3(
    self, *, X, y, verbose, _y_np, _hybrid_orth_pre_recipes, _gbm_seeded_triplet_names, _fe_family_on,
):
    """Run the copula, dcor, hsic, jmim, tc hybrid_orth family stage(s) and return the (possibly
    column-augmented) ``X``. See the package docstring for the full section this carves out."""
    if _fe_family_on("fe_hybrid_orth_copula_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_copula_mi_fe import (
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
            from ..._orthogonal_dcor_fe import (
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
            from ..._orthogonal_hsic_fe import (
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
            from ..._orthogonal_jmim_fe import (
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
            from ..._orthogonal_total_correlation_fe import (
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

    return X
