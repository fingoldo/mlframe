"""FE cascade stage of the row-statistic family (kind ``row_stat``), called from ``_fe_stage_cascade_mid_b``."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def _stage_row_stat(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _row_stat_pre_recipes, verbose):
    """Run the row-statistic family when it is enabled; returns the frame with its accepted columns appended."""
    if not _fe_family_on("fe_row_stat_enable", False):
        return X
    if not isinstance(X, pd.DataFrame):
        warnings.warn(
            "MRMR: row_stat FE enabled but X is not a pandas DataFrame; the features are skipped. Convert via X.to_pandas() before fit() to apply them.",
            UserWarning, stacklevel=3,
        )
        return X
    try:
        from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
        from mlframe.feature_selection.filters._row_stat_fe import hybrid_row_stat_fe

        _rs_step = int(getattr(self, "_fe_steps_executed_", -1))

        def _rs_reject_sink(**_kw):
            """Reject-sink callback; records significance kills into the FE rejection ledger (pure-record, does not affect selection)."""
            _record_fe_rejection(self, step=_rs_step, **_kw)

        # RAW columns only: a recipe built on an engineered column could not be replayed in a single pass at transform() time.
        _rs_cols_cfg = tuple(getattr(self, "fe_row_stat_cols", ()) or ())
        _rs_cols = [c for c in _rs_cols_cfg if c in X.columns] or None
        if _rs_cols is None:
            _rs_raw = set(_raw_input_cols_pre_fe)
            _rs_cols = [c for c in X.columns if c in _rs_raw] or None
        _X_before = list(X.columns)
        X_rs, _rs_appended, _rs_recipes, _ = hybrid_row_stat_fe(
            X, _y_np,
            num_cols=_rs_cols,
            top_k=int(getattr(self, "fe_row_stat_top_k", 2)),
            scan_rows=int(getattr(self, "fe_row_stat_scan_rows", 20_000)),
            min_relative_gain=float(getattr(self, "fe_row_stat_min_relative_gain", 0.05)),
            reject_sink=_rs_reject_sink,
        )
        _rs_appended = [c for c in _rs_appended if c not in _X_before]
        if _rs_appended:
            self.row_stat_features_ = list(_rs_appended)
            self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_rs_appended)
            for _r in _rs_recipes:
                if _r.name in _rs_appended:
                    _row_stat_pre_recipes[_r.name] = _r
            if verbose:
                logger.info("MRMR.fit row_stat: appended %d engineered column(s): %s", len(_rs_appended), _rs_appended[:8])
            return X_rs
    except Exception as _rs_exc:
        logger.warning("MRMR.fit row_stat FE raised %s: %s; continuing without row-statistic columns.", type(_rs_exc).__name__, _rs_exc)
    return X
