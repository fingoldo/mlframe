"""Sibling of ``_fit_impl_core.py`` (part of the sub-split that brings the parent below
the project's 1k-LOC module-size gate).

Holds ``_fe_stage_cascade_mid_b``: Layer 95 PART B (per-group distribution-distance) and
Layer 104 (rare-category / conditional-residual / conditional-dispersion /
conditional-quantile-rank / ordinal-pattern / random-fourier / SIR-direction / LOF /
mahalanobis-density / wavelet / rankgauss). Every FE family stage here reads the (possibly
already-augmented) ``X`` and appends its own winning engineered columns -- mirrors the
sibling cascade modules' own ``X``-in-``X``-out contract.

All ``_*_pre_recipes`` dicts are caller-owned and mutated in place (never reassigned here --
confirmed via a systematic reassignment-vs-mutation check), so no return is needed for them.
``_raw_input_cols_pre_fe`` is read-only here (computed by an earlier cascade sibling).

This is the LAST wave of the Layer 23-104 FE cascade: after this, ``_fit_impl_core.py``'s
remaining body is the MI-greedy screen / friend-graph / assign-support / finalise machinery
already carved into their own siblings in earlier waves.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _fe_stage_cascade_mid_b(
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
    _group_distance_pre_recipes,
    _rare_category_pre_recipes,
    _conditional_residual_pre_recipes,
    _conditional_dispersion_pre_recipes,
    _conditional_quantile_rank_pre_recipes,
    _ordinal_pattern_pre_recipes,
    _random_fourier_pre_recipes,
    _sir_direction_pre_recipes,
    _lof_pre_recipes,
    _mahalanobis_density_pre_recipes,
    _wavelet_pre_recipes,
    _rankgauss_pre_recipes,
):
    """Run the Layer 95B/104 FE family stage(s) and return the (possibly column-augmented)
    ``X``. See the module docstring for the full section this carves out.
    """
    # Layer 95 PART B: per-group distribution-distance. For each
    # (group, num) emit the group-level z / KL / Wasserstein-1 distance from the
    # global distribution, broadcast to rows; each survivor MI-gated against the
    # source num_col marginal MI. Routing piggybacks on hybrid_orth_features_.
    if _fe_family_on("fe_group_distance_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 95 group_distance FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._group_distance_fe import hybrid_group_distance_fe

                _y_for_gd = _y_np
                _gd_groups = tuple(getattr(self, "fe_group_distance_group_cols", ()) or ())
                _gd_groups = [c for c in _gd_groups if c in X.columns] or None  # type: ignore[assignment]
                _gd_nums = tuple(getattr(self, "fe_group_distance_num_cols", ()) or ())
                _gd_nums = [c for c in _gd_nums if c in X.columns] or None  # type: ignore[assignment]
                _gd_top_k = int(getattr(self, "fe_group_distance_top_k", 6))
                _X_before_gd_cols = list(X.columns)
                X_gd, _gd_appended, _gd_recipes, _gd_scores = hybrid_group_distance_fe(
                    X,
                    _y_for_gd,
                    group_cols=_gd_groups,
                    num_cols=_gd_nums,
                    top_k=_gd_top_k,
                )
                _gd_appended = [c for c in _gd_appended if c not in _X_before_gd_cols]
                if _gd_appended:
                    X = X_gd
                    self.group_distance_features_ = list(_gd_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_gd_appended)
                    for _r in _gd_recipes:
                        if _r.name in _gd_appended:
                            _group_distance_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit group_distance: appended %d engineered " "column(s): %s",
                            len(_gd_appended),
                            _gd_appended[:8],
                        )
            except Exception as _gd_exc:
                logger.warning(
                    "MRMR.fit group_distance FE raised %s: %s; continuing " "without group-distance columns.",
                    type(_gd_exc).__name__,
                    _gd_exc,
                )

    # Layer 104: THREE new recipe-based FE families.
    # Family D: conditional dispersion / 2nd-moment.
    self.rare_category_features_ = []
    self.conditional_residual_features_ = []
    self.conditional_dispersion_features_ = []
    self.conditional_quantile_rank_features_ = []
    self.ordinal_pattern_features_ = []
    self.random_fourier_features_ = []
    self.sir_direction_features_ = []
    self.lof_features_ = []
    self.mahalanobis_density_features_ = []
    self.wavelet_features_ = []
    self.rankgauss_features_ = []

    # FAMILY A - rare-category indicator + frequency-band encoding. A category
    # being RARE is itself predictive; emit is_rare_{col} + freq_band_{col}.
    # MI-gated against the raw-baseline floor. Routing piggybacks on
    # hybrid_orth_features_.
    if _fe_family_on("fe_rare_category_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 104 rare_category FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_rare_category_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                # W6 follow-up: rare-category family's unified local-MI abs-MAD
                # floor kills (pure-record; selection byte-identical).
                _rc_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _rc_reject_sink(**_kw):
                    """Reject-sink callback for the rare-category FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_rc_step, **_kw)

                _y_for_rc = _y_np
                _rc_cols = tuple(getattr(self, "fe_rare_category_cols", ()) or ())
                _rc_cols = [c for c in _rc_cols if c in X.columns] or None  # type: ignore[assignment]
                _X_before_rc_cols = list(X.columns)
                _rc_raw_floor = X[[c for c in _raw_input_cols_pre_fe if c in X.columns]] if _raw_input_cols_pre_fe else None
                X_rc, _rc_appended, _rc_recipes, _ = hybrid_rare_category_fe(
                    X, _y_for_rc,
                    cat_cols=_rc_cols,
                    rare_threshold=float(getattr(self, "fe_rare_category_threshold", 0.01)),
                    top_k=int(getattr(self, "fe_rare_category_top_k", 10)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_rc_reject_sink,
                    raw_floor_X=_rc_raw_floor,
                )
                _rc_appended = [c for c in _rc_appended if c not in _X_before_rc_cols]
                if _rc_appended:
                    X = X_rc
                    self.rare_category_features_ = list(_rc_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rc_appended)
                    for _r in _rc_recipes:
                        if _r.name in _rc_appended:
                            _rare_category_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit rare_category: appended %d engineered " "column(s): %s",
                            len(_rc_appended),
                            _rc_appended[:8],
                        )
            except Exception as _rc_exc:
                logger.warning(
                    "MRMR.fit rare_category FE raised %s: %s; continuing " "without rare-category columns.",
                    type(_rc_exc).__name__,
                    _rc_exc,
                )

    # FAMILY B - NUM x NUM conditional residual x_i - E[x_i | bin(x_j)].
    # Cardinality-bounded by top raw-MI columns; MI-gated. Routing piggybacks on
    # hybrid_orth_features_.
    if _fe_family_on("fe_conditional_residual_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 104 conditional_residual FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_conditional_residual_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                # W6 follow-up: conditional-residual family's unified local-MI
                # abs-MAD floor kills (pure-record; selection byte-identical).
                _cr_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _cr_reject_sink(**_kw):
                    """Reject-sink callback for the num-x-num conditional-residual FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_cr_step, **_kw)

                _y_for_cr = _y_np
                _cr_cols = tuple(getattr(self, "fe_conditional_residual_cols", ()) or ())
                _cr_cols = [c for c in _cr_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (mirrors conditional_dispersion / wavelet): X is
                # already augmented with engineered intermediates here, and a
                # conditional-residual recipe built on an engineered x_i / x_j source
                # cannot be replayed at transform() (the engineered parent is not
                # present in the apply X) -> KeyError. Scope auto-detect to raw cols.
                if _cr_cols is None:
                    _cr_raw = set(_raw_input_cols_pre_fe)
                    _cr_cols = [c for c in X.columns if c in _cr_raw] or None
                _X_before_cr_cols = list(X.columns)
                _cr_raw_floor = X[[c for c in _raw_input_cols_pre_fe if c in X.columns]] if _raw_input_cols_pre_fe else None
                X_cr, _cr_appended, _cr_recipes, _ = hybrid_conditional_residual_fe(
                    X, _y_for_cr,
                    num_cols=_cr_cols,
                    n_bins=int(getattr(self, "fe_conditional_residual_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_residual_top_k", 10)),
                    max_pair_cols=int(getattr(self, "fe_conditional_residual_max_pair_cols", 6)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_cr_reject_sink,
                    raw_floor_X=_cr_raw_floor,
                )
                _cr_appended = [c for c in _cr_appended if c not in _X_before_cr_cols]
                if _cr_appended:
                    X = X_cr
                    self.conditional_residual_features_ = list(_cr_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cr_appended)
                    for _r in _cr_recipes:
                        if _r.name in _cr_appended:
                            _conditional_residual_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_residual: appended %d " "engineered column(s): %s",
                            len(_cr_appended),
                            _cr_appended[:8],
                        )
            except Exception as _cr_exc:
                logger.warning(
                    "MRMR.fit conditional_residual FE raised %s: %s; continuing " "without conditional-residual columns.",
                    type(_cr_exc).__name__,
                    _cr_exc,
                )

    # FAMILY D - NUM x NUM conditional DISPERSION / 2nd-moment.
    # Bin x_j; per bin store conditional STD of x_i; emit |z| / z^2 (conditional
    # dispersion anomaly). DEFAULT-ON: MI-gateable (|z| is a non-monotone fold ->
    # genuine MI on heteroscedastic targets) + SELF-LIMITING (a dual-uplift gate
    # admits a column only when its MI beats BOTH raw x_i AND the |mean-residual|
    # Family-B sibling, so homoscedastic / canonical fixtures admit 0 and the
    # operator does not perturb pair-FE recovery). Routing piggybacks on
    # hybrid_orth_features_; recipes carry no y -> leak-safe replay.
    if _fe_family_on("fe_conditional_dispersion_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Family D conditional_dispersion FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_conditional_dispersion_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                # W6 follow-up: conditional-dispersion family's unified local-MI
                # abs-MAD floor kills (pure-record; selection byte-identical).
                _cd_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _cd_reject_sink(**_kw):
                    """Reject-sink callback for the num-x-num conditional-dispersion FE stage; records abs-MAD floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_cd_step, **_kw)

                _y_for_cd = _y_np
                _cd_cols = tuple(getattr(self, "fe_conditional_dispersion_cols", ()) or ())
                _cd_cols = [c for c in _cd_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (2026-06-10 fix, same class as the wavelet stage
                # below): the all-numeric default scope over the already-augmented X
                # builds dispersion features OF engineered columns -> nested recipes
                # the 1-deep replay cannot order at transform() time (KeyError on the
                # engineered parent when it is not selected). Raw scope keeps every
                # conditional-dispersion recipe replayable.
                # ``feature_names_in_`` is not yet assigned here; scope to the raw
                # pre-FE column snapshot (the cat_pair / cat_triple guard's ledger),
                # which is strictly safer than the ``hybrid_orth_features_`` exclusion
                # - that ledger only tracks orth / hinge / wavelet columns and misses
                # ratio / grouped-agg / numeric-decompose engineered intermediates a
                # dispersion recipe would otherwise build on and fail to replay.
                if _cd_cols is None:
                    _cd_raw = set(_raw_input_cols_pre_fe)
                    _cd_cols = [c for c in X.columns if c in _cd_raw] or None
                _X_before_cd_cols = list(X.columns)
                X_cd, _cd_appended, _cd_recipes, _ = hybrid_conditional_dispersion_fe(
                    X, _y_for_cd,
                    num_cols=_cd_cols,
                    n_bins=int(getattr(self, "fe_conditional_dispersion_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_dispersion_top_k", 10)),
                    max_pair_cols=int(getattr(self, "fe_conditional_dispersion_max_pair_cols", 6)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_cd_reject_sink,
                )
                _cd_appended = [c for c in _cd_appended if c not in _X_before_cd_cols]
                if _cd_appended:
                    X = X_cd
                    self.conditional_dispersion_features_ = list(_cd_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cd_appended)
                    for _r in _cd_recipes:
                        if _r.name in _cd_appended:
                            _conditional_dispersion_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_dispersion: appended %d " "engineered column(s): %s",
                            len(_cd_appended),
                            _cd_appended[:8],
                        )
            except Exception as _cd_exc:
                logger.warning(
                    "MRMR.fit conditional_dispersion FE raised %s: %s; continuing " "without conditional-dispersion columns.",
                    type(_cd_exc).__name__,
                    _cd_exc,
                )

    # CONDITIONAL QUANTILE-RANK: 4th member of the
    # conditional-dispersion family. Bin x_j; emit q(row) = empirical_rank(x_i within bin(x_j)) -
    # the row's TRUE within-bin percentile, not a z-score. MI-gated + self-limiting (a near-
    # monotone reparametrization on homoscedastic/non-skewed data clears no uplift over raw x_i, so
    # it does not perturb genuine-feature recovery on canonical fixtures). Routing piggybacks on
    # hybrid_orth_features_; recipes carry no y -> leak-safe replay.
    if _fe_family_on("fe_conditional_quantile_rank_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: conditional_quantile_rank FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._conditional_quantile_rank_fe import hybrid_conditional_quantile_rank_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _cqr_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _cqr_reject_sink(**_kw):
                    """Reject-sink callback for the num-x-num conditional-quantile-rank FE stage; records
                    MI-floor kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_cqr_step, **_kw)

                _y_for_cqr = _y_np
                _cqr_cols = tuple(getattr(self, "fe_conditional_quantile_rank_cols", ()) or ())
                _cqr_cols = [c for c in _cqr_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion / wavelet above): the
                # all-numeric default scope over the already-augmented X builds quantile-rank
                # features OF engineered columns -> nested recipes the 1-deep replay cannot order
                # at transform() time. Raw scope keeps every recipe replayable.
                if _cqr_cols is None:
                    _cqr_raw = set(_raw_input_cols_pre_fe)
                    _cqr_cols = [c for c in X.columns if c in _cqr_raw] or None
                _X_before_cqr_cols = list(X.columns)
                X_cqr, _cqr_appended, _cqr_recipes, _ = hybrid_conditional_quantile_rank_fe(
                    X, _y_for_cqr,
                    num_cols=_cqr_cols,
                    n_bins=int(getattr(self, "fe_conditional_quantile_rank_n_bins", 10)),
                    top_k=int(getattr(self, "fe_conditional_quantile_rank_top_k", 10)),
                    max_pair_cols=int(getattr(self, "fe_conditional_quantile_rank_max_pair_cols", 6)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_cqr_reject_sink,
                )
                _cqr_appended = [c for c in _cqr_appended if c not in _X_before_cqr_cols]
                if _cqr_appended:
                    X = X_cqr
                    self.conditional_quantile_rank_features_ = list(_cqr_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_cqr_appended)
                    for _r in _cqr_recipes:
                        if _r.name in _cqr_appended:
                            _conditional_quantile_rank_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit conditional_quantile_rank: appended %d " "engineered column(s): %s",
                            len(_cqr_appended),
                            _cqr_appended[:8],
                        )
            except Exception as _cqr_exc:
                logger.warning(
                    "MRMR.fit conditional_quantile_rank FE raised %s: %s; continuing " "without conditional-quantile-rank columns.",
                    type(_cqr_exc).__name__,
                    _cqr_exc,
                )

    # ORDINAL PATTERN (Bandt-Pompe) K-fold TARGET ENCODING.
    # For each K-tuple of raw numeric columns, compute the row's rank-permutation id (0..K!-1) and
    # K-fold-TE encode it - a fused single-hop recipe: the intermediate perm_id categorical is
    # never exposed as its own column, avoiding a 2-deep nested-recipe replay the 1-deep convention
    # here cannot order. Routing piggybacks on hybrid_orth_features_; recipe carries a frozen
    # (fit-time) TE lookup, not y -> leak-safe replay.
    if _fe_family_on("fe_ordinal_pattern_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: ordinal_pattern FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._ordinal_pattern_fe import hybrid_ordinal_pattern_te_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _opat_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _opat_reject_sink(**_kw):
                    """Reject-sink callback for the ordinal-pattern-TE FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_opat_step, **_kw)

                _y_for_opat = _y_np
                _opat_cols = tuple(getattr(self, "fe_ordinal_pattern_cols", ()) or ())
                _opat_cols = [c for c in _opat_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank above): the
                # all-numeric default scope over the already-augmented X builds ordinal patterns OF
                # engineered columns -> nested recipes the 1-deep replay cannot order at
                # transform() time. Raw scope keeps every recipe replayable.
                if _opat_cols is None:
                    _opat_raw = set(_raw_input_cols_pre_fe)
                    _opat_cols = [c for c in X.columns if c in _opat_raw] or None
                _X_before_opat_cols = list(X.columns)
                X_opat, _opat_appended, _opat_recipes, _ = hybrid_ordinal_pattern_te_fe(
                    X, _y_for_opat,
                    num_cols=_opat_cols,
                    k=int(getattr(self, "fe_ordinal_pattern_k", 3)),
                    max_cols_for_tuples=int(getattr(self, "fe_ordinal_pattern_max_cols_for_tuples", 5)),
                    n_folds=int(getattr(self, "fe_ordinal_pattern_n_folds", 5)),
                    smoothing=float(getattr(self, "fe_ordinal_pattern_smoothing", 10.0)),
                    top_k=int(getattr(self, "fe_ordinal_pattern_top_k", 5)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_opat_reject_sink,
                )
                _opat_appended = [c for c in _opat_appended if c not in _X_before_opat_cols]
                if _opat_appended:
                    X = X_opat
                    self.ordinal_pattern_features_ = list(_opat_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_opat_appended)
                    for _r in _opat_recipes:
                        if _r.name in _opat_appended:
                            _ordinal_pattern_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit ordinal_pattern: appended %d " "engineered column(s): %s",
                            len(_opat_appended),
                            _opat_appended[:8],
                        )
            except Exception as _opat_exc:
                logger.warning(
                    "MRMR.fit ordinal_pattern FE raised %s: %s; continuing " "without ordinal-pattern columns.",
                    type(_opat_exc).__name__,
                    _opat_exc,
                )

    # RANDOM FOURIER FEATURES (random kitchen sinks) joint kernel-approximation block
    # . Unlike every pair/triplet/quadruplet cross-basis
    # family, this draws m random features that are jointly a smooth function of MANY (5+) raw
    # columns simultaneously without combinatorial blow-up, approximating an RBF kernel over the
    # bounded column pool. Routing piggybacks on hybrid_orth_features_; recipe carries the frozen
    # W-column/phase/bandwidth, never y -> leak-safe replay.
    if _fe_family_on("fe_random_fourier_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: random_fourier FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._random_fourier_features_fe import hybrid_random_fourier_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _rff_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _rff_reject_sink(**_kw):
                    """Reject-sink callback for the random-fourier FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_rff_step, **_kw)

                _y_for_rff = _y_np
                _rff_cols = tuple(getattr(self, "fe_random_fourier_cols", ()) or ())
                _rff_cols = [c for c in _rff_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern
                # above): the all-numeric default scope over the already-augmented X builds RFF
                # features OF engineered columns -> nested recipes the 1-deep replay cannot order at
                # transform() time. Raw scope keeps every recipe replayable.
                if _rff_cols is None:
                    _rff_raw = set(_raw_input_cols_pre_fe)
                    _rff_cols = [c for c in X.columns if c in _rff_raw] or None
                _X_before_rff_cols = list(X.columns)
                X_rff, _rff_appended, _rff_recipes, _ = hybrid_random_fourier_fe(
                    X, _y_for_rff,
                    num_cols=_rff_cols,
                    m=int(getattr(self, "fe_random_fourier_m", 64)),
                    bandwidth=getattr(self, "fe_random_fourier_bandwidth", None),
                    max_cols_for_block=int(getattr(self, "fe_random_fourier_max_cols_for_block", 8)),
                    top_k=int(getattr(self, "fe_random_fourier_top_k", 8)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_rff_reject_sink,
                )
                _rff_appended = [c for c in _rff_appended if c not in _X_before_rff_cols]
                if _rff_appended:
                    X = X_rff
                    self.random_fourier_features_ = list(_rff_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rff_appended)
                    for _r in _rff_recipes:
                        if _r.name in _rff_appended:
                            _random_fourier_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit random_fourier: appended %d " "engineered column(s): %s",
                            len(_rff_appended),
                            _rff_appended[:8],
                        )
            except Exception as _rff_exc:
                logger.warning(
                    "MRMR.fit random_fourier FE raised %s: %s; continuing " "without random-fourier columns.",
                    type(_rff_exc).__name__,
                    _rff_exc,
                )

    # SLICED INVERSE REGRESSION (SIR) oblique-direction projection (
    # fe_expansion.md). Recovers a genuinely OBLIQUE (rotated) linear combination spread thinly
    # across several correlated columns - where every individual weight is too small for that
    # column's own marginal MI to clear the screening floor, and no pairwise/triplet/quadruplet
    # product reconstructs the rotated hyperplane. Routing piggybacks on hybrid_orth_features_;
    # recipe carries the frozen centering/direction, not y -> leak-safe replay.
    if _fe_family_on("fe_sir_direction_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: sir_direction FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._sliced_inverse_regression_fe import hybrid_sir_direction_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _sir_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _sir_reject_sink(**_kw):
                    """Reject-sink callback for the SIR-direction FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_sir_step, **_kw)

                _y_for_sir = _y_np
                _sir_cols = tuple(getattr(self, "fe_sir_direction_cols", ()) or ())
                _sir_cols = [c for c in _sir_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern/
                # random_fourier above): the all-numeric default scope over the already-augmented X
                # builds SIR directions OF engineered columns -> nested recipes the 1-deep replay
                # cannot order at transform() time. Raw scope keeps every recipe replayable.
                if _sir_cols is None:
                    _sir_raw = set(_raw_input_cols_pre_fe)
                    _sir_cols = [c for c in X.columns if c in _sir_raw] or None
                _X_before_sir_cols = list(X.columns)
                X_sir, _sir_appended, _sir_recipes, _ = hybrid_sir_direction_fe(
                    X, _y_for_sir,
                    num_cols=_sir_cols,
                    n_slices=int(getattr(self, "fe_sir_direction_n_slices", 10)),
                    n_directions=int(getattr(self, "fe_sir_direction_n_directions", 2)),
                    max_cols_for_block=int(getattr(self, "fe_sir_direction_max_cols_for_block", 8)),
                    top_k=int(getattr(self, "fe_sir_direction_top_k", 2)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_sir_reject_sink,
                )
                _sir_appended = [c for c in _sir_appended if c not in _X_before_sir_cols]
                if _sir_appended:
                    X = X_sir
                    self.sir_direction_features_ = list(_sir_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_sir_appended)
                    for _r in _sir_recipes:
                        if _r.name in _sir_appended:
                            _sir_direction_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit sir_direction: appended %d " "engineered column(s): %s",
                            len(_sir_appended),
                            _sir_appended[:8],
                        )
            except Exception as _sir_exc:
                logger.warning(
                    "MRMR.fit sir_direction FE raised %s: %s; continuing " "without sir-direction columns.",
                    type(_sir_exc).__name__,
                    _sir_exc,
                )

    # LOCAL OUTLIER FACTOR / k-NN local density-ratio.
    # LOCAL and non-parametric (unlike a global Mahalanobis ellipsoid), catching a row anomalous
    # for sitting in a locally-sparse gap between well-separated clusters even when its GLOBAL
    # distance to the overall mean is unremarkable. Routing piggybacks on hybrid_orth_features_;
    # recipe carries a bounded frozen reference sample (RAM discipline), never y or the whole fit
    # frame -> leak-safe replay.
    if _fe_family_on("fe_lof_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: lof FE enabled but X is not a " "pandas DataFrame; the features are skipped. Convert via " "X.to_pandas() before fit() to apply them.",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from .._lof_fe import hybrid_lof_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _lof_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _lof_reject_sink(**_kw):
                    """Reject-sink callback for the LOF FE stage; records MI-floor kills into the
                    FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_lof_step, **_kw)

                _y_for_lof = _y_np
                _lof_cols = tuple(getattr(self, "fe_lof_cols", ()) or ())
                _lof_cols = [c for c in _lof_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern/
                # random_fourier/sir_direction above): the all-numeric default scope over the
                # already-augmented X builds LOF scores OF engineered columns -> nested recipes the
                # 1-deep replay cannot order at transform() time. Raw scope keeps every recipe replayable.
                if _lof_cols is None:
                    _lof_raw = set(_raw_input_cols_pre_fe)
                    _lof_cols = [c for c in X.columns if c in _lof_raw] or None
                _X_before_lof_cols = list(X.columns)
                X_lof, _lof_appended, _lof_recipes, _ = hybrid_lof_fe(
                    X, _y_for_lof,
                    num_cols=_lof_cols,
                    k=int(getattr(self, "fe_lof_k", 20)),
                    max_ref=int(getattr(self, "fe_lof_max_ref", 2000)),
                    max_cols_for_block=int(getattr(self, "fe_lof_max_cols_for_block", 8)),
                    top_k=int(getattr(self, "fe_lof_top_k", 1)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_lof_reject_sink,
                )
                _lof_appended = [c for c in _lof_appended if c not in _X_before_lof_cols]
                if _lof_appended:
                    X = X_lof
                    self.lof_features_ = list(_lof_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_lof_appended)
                    for _r in _lof_recipes:
                        if _r.name in _lof_appended:
                            _lof_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit lof: appended %d " "engineered column(s): %s",
                            len(_lof_appended),
                            _lof_appended[:8],
                        )
            except Exception as _lof_exc:
                logger.warning(
                    "MRMR.fit lof FE raised %s: %s; continuing " "without lof columns.",
                    type(_lof_exc).__name__,
                    _lof_exc,
                )

    # MULTIVARIATE MAHALANOBIS / GAUSSIAN-COPULA JOINT DENSITY anomaly score (
    # fe_expansion.md). Catches y depending on whether a row sits inside/outside an ELLIPSOIDAL
    # level-set of a p=15-30-way joint distribution where no single column, pair, triplet, or even
    # quadruplet cross-basis is individually extreme - the p-way generalization of the existing
    # group_distance / conditional-dispersion families' one-column-conditioned-on-one-other-column
    # scope. Routing piggybacks on hybrid_orth_features_; recipe carries the frozen Ledoit-Wolf
    # mu/Sigma_inv, never y -> leak-safe replay.
    if _fe_family_on("fe_mahalanobis_density_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: mahalanobis_density FE enabled but X is not a "
                "pandas DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._mahalanobis_density_fe import hybrid_mahalanobis_density_fe
                from .._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

                _mahal_step = int(getattr(self, "_fe_steps_executed_", -1))

                def _mahal_reject_sink(**_kw):
                    """Reject-sink callback for the Mahalanobis-density FE stage; records MI-floor
                    kills into the FE rejection ledger (pure-record, does not affect selection)."""
                    _record_fe_rejection(self, step=_mahal_step, **_kw)

                _y_for_mahal = _y_np
                _mahal_cols = tuple(getattr(self, "fe_mahalanobis_density_cols", ()) or ())
                _mahal_cols = [c for c in _mahal_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (same class as conditional_dispersion/quantile_rank/ordinal_pattern/
                # random_fourier/sir_direction/lof above): the all-numeric default scope over the
                # already-augmented X builds Mahalanobis density OF engineered columns -> nested
                # recipes the 1-deep replay cannot order at transform() time. Raw scope keeps every
                # recipe replayable.
                if _mahal_cols is None:
                    _mahal_raw = set(_raw_input_cols_pre_fe)
                    _mahal_cols = [c for c in X.columns if c in _mahal_raw] or None
                _X_before_mahal_cols = list(X.columns)
                X_mahal, _mahal_appended, _mahal_recipes, _ = hybrid_mahalanobis_density_fe(
                    X, _y_for_mahal,
                    num_cols=_mahal_cols,
                    max_cols_for_block=int(getattr(self, "fe_mahalanobis_density_max_cols_for_block", 20)),
                    top_k=int(getattr(self, "fe_mahalanobis_density_top_k", 1)),
                    mi_gate=bool(getattr(self, "fe_local_mi_gate", False)),
                    mi_gate_top_k=int(getattr(self, "fe_local_mi_gate_top_k", 20)),
                    reject_sink=_mahal_reject_sink,
                )
                _mahal_appended = [c for c in _mahal_appended if c not in _X_before_mahal_cols]
                if _mahal_appended:
                    X = X_mahal
                    self.mahalanobis_density_features_ = list(_mahal_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_mahal_appended)
                    for _r in _mahal_recipes:
                        if _r.name in _mahal_appended:
                            _mahalanobis_density_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit mahalanobis_density: appended %d " "engineered column(s): %s",
                            len(_mahal_appended),
                            _mahal_appended[:8],
                        )
            except Exception as _mahal_exc:
                logger.warning(
                    "MRMR.fit mahalanobis_density FE raised %s: %s; continuing " "without mahalanobis-density columns.",
                    type(_mahal_exc).__name__,
                    _mahal_exc,
                )

    # HAAR WAVELET / localized multiresolution basis.
    # A NEW operator for LOCALIZED bump / multiscale piecewise structure: y jumps
    # only inside a narrow sub-window of x (Fourier Gibbs-rings it, spline's fixed
    # quantile knots smooth it away). Emits a small held-out-scale-selected dyadic
    # set of Haar indicators psi_{j,k} (+1 left / -1 right half of a dyadic
    # interval). DEFAULT-ON + SELF-LIMITING: the noise-aware held-out MAD floor +
    # max-legs cap bound the candidate explosion, and each leg is admitted on its
    # held-out INCREMENTAL MI over raw x AND a complementarity guard (must beat a
    # SMOOTH location-refinement of x) - so a localized step/bump admits legs, a
    # SMOOTH (sin / monotone) column admits 0 (Fourier owns it, complementary),
    # pure noise admits 0. The leg is NON-monotone -> MI-VISIBLE, so it routes
    # through the MI-based gate (no deferred-materialise / re-add dance the
    # MI-invariant hinge needs). Recipes (``orth_wavelet``) store (lo, span) +
    # dyadic (j, k); replay is the closed-form indicator - no y, leak-safe.
    # Routing piggybacks on hybrid_orth_features_ (like Family D dispersion).
    if _fe_family_on("fe_wavelet_enable", False) and _fe_budget_ok():
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Haar wavelet FE enabled but X is not a pandas DataFrame; "
                "the features are skipped. Convert via X.to_pandas() before fit() "
                "to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._wavelet_basis_fe_recipes import hybrid_wavelet_fe_with_recipes

                _y_for_wv = _y_np
                _wv_cols = tuple(getattr(self, "fe_wavelet_cols", ()) or ())
                _wv_cols = [c for c in _wv_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (2026-06-10 fix, mirrors the extra-basis stage's
                # guard at the hybrid_orth call above): by this point X is ALREADY
                # augmented with poly/fourier/spline/hinge engineered columns, so the
                # all-numeric default scope emitted NESTED recipes (e.g.
                # ``x0__p2sin1__haar_j3k5`` - a Haar leg of an engineered Fourier
                # column) whose 1-deep replay cannot order the parent materialisation
                # and raised KeyError('x0__p2sin1') at transform() time whenever the
                # parent was not itself selected. Scoping to ``feature_names_in_``
                # keeps every wavelet recipe 1-deep and replayable.
                # NOTE: ``self.feature_names_in_`` is not assigned until the
                # target-injection block far below, so the exclusion source is the
                # ``hybrid_orth_features_`` ledger every prior univariate stage
                # appends to (the hinge stage's exact pattern).
                if _wv_cols is None:
                    _wv_already = set(getattr(self, "hybrid_orth_features_", None) or [])
                    _wv_cols = [c for c in X.columns if c not in _wv_already] or None
                _X_before_wv_cols = list(X.columns)
                X_wv, _wv_appended, _wv_recipes, _ = hybrid_wavelet_fe_with_recipes(
                    X, _y_for_wv,
                    cols=_wv_cols,
                    max_scale=int(getattr(self, "fe_wavelet_max_scale", 3)),
                    max_legs=int(getattr(self, "fe_wavelet_max_legs", 6)),
                    top_k=int(getattr(self, "fe_wavelet_top_k", 8)),
                    feature_dtype=getattr(self, "usability_feature_dtype", np.float32),
                    max_cols=getattr(self, "fe_wavelet_max_cols", None),
                )
                _wv_appended = [c for c in _wv_appended if c not in _X_before_wv_cols]
                if _wv_appended:
                    X = X_wv
                    self.wavelet_features_ = list(_wv_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_wv_appended)
                    for _r in _wv_recipes:
                        if _r.name in _wv_appended:
                            _wavelet_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit wavelet: appended %d engineered column(s): %s",
                            len(_wv_appended), _wv_appended[:8],
                        )
            except Exception as _wv_exc:
                logger.warning(
                    "MRMR.fit Haar wavelet FE raised %s: %s; continuing without " "wavelet columns.",
                    type(_wv_exc).__name__,
                    _wv_exc,
                )

    # FAMILY C - RankGauss (rank-Gaussianisation). NOT MI-gated: monotone ->
    # MI-invariant by the data-processing inequality; the pool is bounded by raw
    # marginal MI and the value is downstream (linear / NN). Routing piggybacks
    # on hybrid_orth_features_.
    if _fe_family_on("fe_rankgauss_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: Layer 104 rankgauss FE enabled but X is not a pandas "
                "DataFrame; the features are skipped. Convert via "
                "X.to_pandas() before fit() to apply them.",
                UserWarning, stacklevel=3,
            )
        else:
            try:
                from .._extra_fe_families import hybrid_rankgauss_fe

                _y_for_rg = _y_np
                _rg_cols = tuple(getattr(self, "fe_rankgauss_cols", ()) or ())
                _rg_cols = [c for c in _rg_cols if c in X.columns] or None  # type: ignore[assignment]
                # RAW columns only (2026-06-10 fix, same class as the wavelet /
                # conditional-dispersion stages): keep rankgauss recipes 1-deep and
                # replayable - never rank-Gaussianise an engineered column whose
                # parent the transform()-time replay cannot materialise first.
                # ``feature_names_in_`` is not yet assigned here; exclude via the
                # ``hybrid_orth_features_`` ledger (hinge-stage pattern).
                if _rg_cols is None:
                    _rg_already = set(getattr(self, "hybrid_orth_features_", None) or [])
                    _rg_cols = [c for c in X.columns if c not in _rg_already] or None
                _X_before_rg_cols = list(X.columns)
                X_rg, _rg_appended, _rg_recipes, _ = hybrid_rankgauss_fe(
                    X, _y_for_rg,
                    num_cols=_rg_cols,
                    top_k=int(getattr(self, "fe_rankgauss_top_k", 10)),
                )
                _rg_appended = [c for c in _rg_appended if c not in _X_before_rg_cols]
                if _rg_appended:
                    X = X_rg
                    self.rankgauss_features_ = list(_rg_appended)
                    self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rg_appended)
                    for _r in _rg_recipes:
                        if _r.name in _rg_appended:
                            _rankgauss_pre_recipes[_r.name] = _r
                    if verbose:
                        logger.info(
                            "MRMR.fit rankgauss: appended %d engineered " "column(s): %s",
                            len(_rg_appended),
                            _rg_appended[:8],
                        )
            except Exception as _rg_exc:
                logger.warning(
                    "MRMR.fit rankgauss FE raised %s: %s; continuing without " "rankgauss columns.",
                    type(_rg_exc).__name__,
                    _rg_exc,
                )

    return X
