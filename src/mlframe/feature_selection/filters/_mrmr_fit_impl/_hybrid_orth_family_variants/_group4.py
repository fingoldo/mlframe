"""Sibling of ``_hybrid_orth_family_variants/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
hybrid_orth family-variant block, itself further split for the 1k-LOC module-size gate).

Holds families: cmim, auto_scorer, ensemble, meta. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``self``/``X`` threading contract (mirrors the parent's own).
"""

from __future__ import annotations

import logging

import numpy as np

from .._helpers import _orth_fe_numeric_cols, fe_decide_on_subsample
from ..._fe_frame_ops import fe_append_columns, fe_extract_columns

logger = logging.getLogger(__name__)


def _hybrid_orth_family_variants_group4(
    self, *, X, y, verbose, _y_np, _hybrid_orth_pre_recipes, _gbm_seeded_triplet_names, _fe_family_on,
):
    """Run the cmim, auto_scorer, ensemble, meta hybrid_orth family stage(s) and return the (possibly
    column-augmented) ``X``. See the package docstring for the full section this carves out."""
    if _fe_family_on("fe_hybrid_orth_cmim_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_cmim_fe import (
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
            from ..._orthogonal_scorer_auto_fe import (
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
            from ..._orthogonal_scorer_auto_fe import (
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
            from ..._orthogonal_meta_scorer_fe import (
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
