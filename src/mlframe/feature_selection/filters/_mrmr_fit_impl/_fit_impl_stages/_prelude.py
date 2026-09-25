"""Stages of ``_fit_impl`` that run before discretisation: NaN bookkeeping, nbins gates, fit cache, relevance-gain floor."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
# --- end imports ---


def _apply_large_n_regression_nbins_gate(self, X, y):
    """Switch to fixed 20-bin quantile binning on large-n regression when the user left the quantization params at defaults."""
    if (
        getattr(self, "adaptive_nbins_large_n_reg", False)
        and getattr(self, "nbins_strategy", None) == "mdlp"
        and int(getattr(self, "quantization_nbins", 10)) == 10
    ):
        _n_rows_gate = int(X.shape[0]) if hasattr(X, "shape") else 0
        _thr = int(getattr(self, "adaptive_nbins_large_n_reg_threshold", 50_000))
        if _n_rows_gate >= _thr:
            _explicit_tt_gate = getattr(self, "target_type", None)
            if _explicit_tt_gate is not None:
                _tt_str_gate = str(_explicit_tt_gate).lower()
                _is_reg_gate = not ("classif" in _tt_str_gate or _tt_str_gate in ("binary", "multiclass", "multilabel"))
            else:
                _y_arr_gate = np.asarray(y)
                _n_unique_gate = len(np.unique(_y_arr_gate))
                _ratio_gate = len(_y_arr_gate) / max(1, _n_unique_gate)
                _is_float_gate = _y_arr_gate.dtype.kind == "f"
                _is_classification_gate = (not _is_float_gate) and _ratio_gate > 100 and _n_unique_gate <= 64
                _is_reg_gate = not _is_classification_gate
            if _is_reg_gate:
                self.nbins_strategy = None
                self.quantization_nbins = int(getattr(self, "adaptive_nbins_large_n_reg_nbins", 20))
                self._adaptive_nbins_large_n_reg_fired_ = True


def _record_fit_entry_nan_mask(X, _will_use_include_numeric_nan_guard, _will_use_missingness_fe, _include_numeric_input_nan_cols, _fit_entry_nan_mask):
    """Record which raw numeric input columns carry NaN at fit entry (the include-numeric NaN guard and missingness FE read it)."""
    if hasattr(X, "columns") and (_will_use_include_numeric_nan_guard or _will_use_missingness_fe):
        for _c in list(X.columns):
            try:
                _cv = X[_c]
                _cv_np = np.asarray(_cv.to_numpy() if hasattr(_cv, "to_numpy") else _cv, dtype=np.float64)
            except (ValueError, TypeError):
                continue
            _nan_mask_c = ~np.isfinite(_cv_np)
            if _nan_mask_c.any():
                _include_numeric_input_nan_cols.add(_c)
                _fit_entry_nan_mask[_c] = _nan_mask_c


def _resolve_min_relevance_gain(self, target_indices, data, nbins, verbose):
    """Validate min_relevance_gain_mode and resolve the effective min_relevance_gain (scaled by target entropy in relative mode)."""
    if self.min_relevance_gain_mode not in ("absolute", "relative_to_entropy"):
        raise ValueError(f"MRMR.min_relevance_gain_mode={self.min_relevance_gain_mode!r} must be 'absolute' or 'relative_to_entropy'.")
    if self.min_relevance_gain_mode == "relative_to_entropy":
        _target_col_idx = int(target_indices[0])
        _y_bins = data[:, _target_col_idx]
        _y_nbins = int(nbins[_target_col_idx])
        _y_counts = np.bincount(_y_bins, minlength=_y_nbins).astype(np.float64)
        _y_total = float(_y_counts.sum())
        if _y_total > 0:
            _p = _y_counts[_y_counts > 0] / _y_total
            _h_y_nats = float(-(_p * np.log(_p)).sum())
        else:
            _h_y_nats = 0.0
        _effective_min_relevance_gain = float(self.min_relevance_gain_frac) * _h_y_nats
        if verbose:
            logger.info(
                "MRMR min_relevance_gain resolution: mode=relative_to_entropy, H(y)=%.4f nats, frac=%.4g, effective floor=%.6g (legacy absolute would have been %.6g).",
                _h_y_nats,
                self.min_relevance_gain_frac,
                _effective_min_relevance_gain,
                self.min_relevance_gain,
            )
    else:
        _effective_min_relevance_gain = float(self.min_relevance_gain)
    # Read by the post-screen usability-aware pure-form retention, which builds its own candidate pool and
    # must not admit an engineered form below a caller-pinned absolute relevance floor.
    return _effective_min_relevance_gain
