"""FE cascade stage of the offset-product family (kind ``offset_product``), called from ``_fe_stage_cascade_mid_b``."""

from __future__ import annotations

import logging
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


def _stage_offset_product(self, _fe_family_on, X, _y_np, _raw_input_cols_pre_fe, _offset_product_pre_recipes, verbose):
    """Run the offset-product family ``(u + s) * (v + t)`` when it is enabled; returns the frame with its accepted columns appended."""
    if not _fe_family_on("fe_offset_product_enable", False):
        return X
    if not isinstance(X, pd.DataFrame):
        warnings.warn(
            "MRMR: offset_product FE enabled but X is not a pandas DataFrame; the features are skipped. Convert via X.to_pandas() before fit() to apply them.",
            UserWarning, stacklevel=3,
        )
        return X
    try:
        from mlframe.feature_selection.filters._fe_rejection_ledger import record_fe_rejection as _record_fe_rejection
        from mlframe.feature_selection.filters._offset_product_fe import hybrid_offset_product_fe

        _op_step = int(getattr(self, "_fe_steps_executed_", -1))

        def _op_reject_sink(**_kw):
            """Reject-sink callback; records held-out-margin kills into the FE rejection ledger (pure-record, does not affect selection)."""
            _record_fe_rejection(self, step=_op_step, **_kw)

        # RAW columns only: a recipe built on an engineered column could not be replayed in a single pass at transform() time.
        _op_cols = tuple(getattr(self, "fe_offset_product_cols", ()) or ())
        _op_cols = [c for c in _op_cols if c in X.columns] or None
        if _op_cols is None:
            _op_raw = set(_raw_input_cols_pre_fe)
            _op_cols = [c for c in X.columns if c in _op_raw] or None
        _X_before = list(X.columns)
        X_op, _op_appended, _op_recipes, _ = hybrid_offset_product_fe(
            X, _y_np,
            num_cols=_op_cols,
            max_pair_cols=int(getattr(self, "fe_offset_product_max_pair_cols", 6)),
            top_k=int(getattr(self, "fe_offset_product_top_k", 3)),
            scan_rows=int(getattr(self, "fe_offset_product_scan_rows", 100_000)),
            min_relative_gain=float(getattr(self, "fe_offset_product_min_relative_gain", 0.05)),
            reject_sink=_op_reject_sink,
        )
        _op_appended = [c for c in _op_appended if c not in _X_before]
        if _op_appended:
            self.offset_product_features_ = list(_op_appended)
            self.hybrid_orth_features_ = list(self.hybrid_orth_features_ or []) + list(_op_appended)
            for _r in _op_recipes:
                if _r.name in _op_appended:
                    _offset_product_pre_recipes[_r.name] = _r
            if verbose:
                logger.info("MRMR.fit offset_product: appended %d engineered column(s): %s", len(_op_appended), _op_appended[:8])
            return X_op
    except Exception as _op_exc:
        logger.warning("MRMR.fit offset_product FE raised %s: %s; continuing without offset-product columns.", type(_op_exc).__name__, _op_exc)
    return X
