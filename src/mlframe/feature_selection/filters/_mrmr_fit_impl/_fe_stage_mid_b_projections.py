"""FE cascade stages of ``_fe_stage_cascade_mid_b`` that project the frame or model its density: random-Fourier features, SIR directions, local outlier factor and
Mahalanobis density. Carved out of ``_fe_stage_cascade_mid_b`` (which re-exports them) to keep that module under the 1000-line budget; each stage reads the
(possibly already-augmented) ``X``, appends its own winning engineered columns and mutates its caller-owned ``_*_pre_recipes`` dict in place.
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _stage_random_fourier(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _random_fourier_pre_recipes, verbose):
    """Run the random-Fourier-feature family when it is enabled."""
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
                from mlframe.feature_selection.filters._random_fourier_features_fe import hybrid_random_fourier_fe
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

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
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_family_failure

                record_family_failure(self, "random_fourier", _rff_exc)
    return X


def _stage_sir_direction(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _sir_direction_pre_recipes, verbose):
    """Run the sliced-inverse-regression direction family when it is enabled."""
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
                from mlframe.feature_selection.filters._sliced_inverse_regression_fe import hybrid_sir_direction_fe
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

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
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_family_failure

                record_family_failure(self, "sir_direction", _sir_exc)
    return X


def _stage_lof(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _lof_pre_recipes, verbose):
    """Run the local-outlier-factor family when it is enabled."""
    if _fe_family_on("fe_lof_enable", False):
        if not isinstance(X, pd.DataFrame):
            warnings.warn(
                "MRMR: lof FE enabled but X is not a " "pandas DataFrame; the features are skipped. Convert via " "X.to_pandas() before fit() to apply them.",
                UserWarning,
                stacklevel=3,
            )
        else:
            try:
                from mlframe.feature_selection.filters._lof_fe import hybrid_lof_fe
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

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
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_family_failure

                record_family_failure(self, "lof", _lof_exc)
    return X


def _stage_mahalanobis_density(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _mahalanobis_density_pre_recipes, verbose):
    """Run the Mahalanobis density family when it is enabled."""
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
                from mlframe.feature_selection.filters._mahalanobis_density_fe import hybrid_mahalanobis_density_fe
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection

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
                from mlframe.feature_selection.filters._fe_rejection_ledger import record_family_failure

                record_family_failure(self, "mahalanobis_density", _mahal_exc)
    return X
