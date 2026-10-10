"""FE cascade stage of the out-of-fold warp family (kind ``oof_warp1d``), called from ``_fe_stage_cascade_mid_b``."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def _stage_oof_warp(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _oof_warp_pre_recipes, verbose):
    """Run the out-of-fold warp family when it is enabled; returns the frame with its accepted columns appended."""
    if not _fe_family_on("fe_oof_warp_enable", False):
        return X
    if not isinstance(X, pd.DataFrame):
        warnings.warn(
            "MRMR: oof_warp FE enabled but X is not a pandas DataFrame; the features are skipped. Convert via X.to_pandas() before fit() to apply them.",
            UserWarning, stacklevel=3,
        )
        return X
    try:
        from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
        from mlframe.feature_selection.filters._oof_warp_fe import hybrid_oof_warp_fe

        _ow_step = int(getattr(self, "_fe_steps_executed_", -1))

        def _ow_reject_sink(**_kw):
            """Reject-sink callback; records significance kills into the FE rejection ledger (pure-record, does not affect selection)."""
            _record_fe_rejection(self, step=_ow_step, **_kw)

        # RAW columns only: a recipe built on an engineered column could not be replayed in a single pass at transform() time.
        _ow_cols_cfg = tuple(getattr(self, "fe_oof_warp_cols", ()) or ())
        _ow_cols = [c for c in _ow_cols_cfg if c in X.columns] or None
        if _ow_cols is None:
            _ow_raw = set(_raw_input_cols_pre_fe)
            _ow_cols = [c for c in X.columns if c in _ow_raw] or None
        _X_before = list(X.columns)
        X_ow, _ow_appended, _ow_recipes, _ = hybrid_oof_warp_fe(
            X, _y_np,
            num_cols=_ow_cols,
            top_k=int(getattr(self, "fe_oof_warp_top_k", 5)),
            scan_rows=int(getattr(self, "fe_oof_warp_scan_rows", 100_000)),
            min_relative_gain=float(getattr(self, "fe_oof_warp_min_relative_gain", 0.05)),
            reject_sink=_ow_reject_sink,
        )
        _ow_appended = [c for c in _ow_appended if c not in _X_before]
        if _ow_appended:
            self.oof_warp_features_ = list(_ow_appended)
            self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_ow_appended)
            for _r in _ow_recipes:
                if _r.name in _ow_appended:
                    _oof_warp_pre_recipes[_r.name] = _r
            if verbose:
                logger.info("MRMR.fit oof_warp: appended %d engineered column(s): %s", len(_ow_appended), _ow_appended[:8])
            return X_ow
    except Exception as _ow_exc:
        logger.warning("MRMR.fit oof_warp FE raised %s: %s; continuing without warp columns.", type(_ow_exc).__name__, _ow_exc)
    return X
