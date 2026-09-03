"""Sibling of ``_hybrid_orth_family_variants/__init__.py`` (the sub-split of ``_fit_impl_core.py``'s
hybrid_orth family-variant block, itself further split for the 1k-LOC module-size gate).

Holds families: diff_basis, cluster_basis, bootstrap, three_gate, ksg. See the package ``__init__.py`` docstring for the full
section this fans out from and the ``self``/``X`` threading contract (mirrors the parent's own).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .._helpers import _orth_fe_numeric_cols, fe_decide_on_subsample
from ..._fe_frame_ops import fe_append_columns, fe_extract_columns

logger = logging.getLogger(__name__)


def _hybrid_orth_family_variants_group2(
    self, *, X, y, verbose, _y_np, _hybrid_orth_pre_recipes, _gbm_seeded_triplet_names, _fe_family_on,
):
    """Run the diff_basis, cluster_basis, bootstrap, three_gate, ksg hybrid_orth family stage(s) and return the (possibly
    column-augmented) ``X``. See the package docstring for the full section this carves out."""
    if _fe_family_on("fe_hybrid_orth_diff_basis_enable", False):
        # Format-agnostic since the matrix-native FE seam (see triplet stage): skip-guard removed, runs on polars/pandas.
        try:
            from ..._orthogonal_diff_basis_fe import (
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
            from ..._orthogonal_cluster_basis_fe import (
                hybrid_orth_mi_cluster_basis_fe_with_recipes,
            )
            from ..._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
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
            from ..._orthogonal_bootstrap_mi_fe import (
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
            from ..._orthogonal_three_gate_mi_fe import (
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
            from ..._orthogonal_ksg_mi_fe import (
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

    return X
