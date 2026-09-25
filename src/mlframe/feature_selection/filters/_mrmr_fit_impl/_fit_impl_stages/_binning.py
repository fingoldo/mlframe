"""Discretisation stages of ``_fit_impl``: target re-binning and compact code storage."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import os
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from typing import Optional
# --- end imports ---


def _compact_code_storage(data):
    """Narrow the discretised code matrix to the smallest integer dtype its range fits (MLFRAME_MRMR_COMPACT_CODES=0 opts out)."""
    if data.size and os.environ.get("MLFRAME_MRMR_COMPACT_CODES", "1").strip().lower() not in ("0", "false", "off", "no"):
        try:
            _dmin = int(data.min())
            _dmax = int(data.max())
            _store_dt: Optional[type]
            if -128 <= _dmin and _dmax <= 127:
                _store_dt = np.int8
            elif -32768 <= _dmin and _dmax <= 32767:
                _store_dt = np.int16
            else:
                _store_dt = None
            if _store_dt is not None and data.dtype.itemsize > np.dtype(_store_dt).itemsize:
                data = data.astype(_store_dt, copy=False)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("mrmr: narrowing stored codes dtype failed: %r", e, exc_info=True)
            pass
    return data


def _rebin_continuous_targets(self, _nbins_strategy, target_indices, cols, _x_for_cat, nbins, data, verbose):
    """Re-bin continuous target columns with the legacy equal-frequency quantile path so target codes keep their signal."""
    if _nbins_strategy is not None and len(target_indices) > 0:
        from mlframe.feature_selection.filters.discretization import discretize_array as _t_discretize

        for _ti in target_indices:
            _t_name = cols[int(_ti)]
            try:
                _t_raw = np.asarray(_x_for_cat[_t_name].to_numpy() if hasattr(_x_for_cat[_t_name], "to_numpy") else _x_for_cat[_t_name])
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("mrmr: extracting raw values for column %r during cat-handling failed: %r", _t_name, e, exc_info=True)
                continue
            if _t_raw.dtype.kind not in "fiub" or _t_raw.ndim != 1:
                continue
            _t_finite = _t_raw[np.isfinite(_t_raw.astype(np.float64))] if _t_raw.dtype.kind == "f" else _t_raw
            if np.unique(_t_finite).size <= int(self.quantization_nbins):
                continue  # discrete / classification target: keep its native classes
            _t_codes = _t_discretize(
                arr=_t_raw.astype(np.float64),
                n_bins=int(self.quantization_nbins),
                method=str(self.quantization_method),
                dtype=self.quantization_dtype,
            )
            _t_nb = int(np.max(_t_codes)) + 1
            if _t_nb >= 2 and (int(nbins[int(_ti)]) != _t_nb or not np.array_equal(data[:, int(_ti)], _t_codes)):
                if verbose:
                    logger.info(
                        "MRMR.fit target-rebin guard: target %r re-binned from the adaptive "
                        "nbins_strategy=%r encoding (%d bins, max-bin %.1f%%) to the legacy "
                        "%s/%d equal-frequency codes (%d bins) -- the adaptive strategy is "
                        "feature-side only; on the target it degrades MI sensitivity.",
                        _t_name,
                        str(_nbins_strategy),
                        int(nbins[int(_ti)]),
                        100.0 * float(np.bincount(data[:, int(_ti)].astype(np.int64)).max()) / max(1, data.shape[0]),
                        str(self.quantization_method),
                        int(self.quantization_nbins),
                        _t_nb,
                    )
                data[:, int(_ti)] = _t_codes
                nbins[int(_ti)] = _t_nb


def _discretize_inputs(self, _is_polars_input, X, target_names):
    """Discretise the input frame (targets included) into per-column bin codes via ``categorize_dataset``, resolving the NaN
    strategy, the adaptive nbins strategy and its kwargs, and the supervised signal the supervised strategies need.
    Returns the strategy, the frame that was binned, and the codes' columns, matrix and per-column bin counts."""
    from mlframe.feature_selection.filters.mrmr import (
        categorize_dataset,
    )

    if self.nan_strategy in ("ffill_bfill",):
        # Legacy path retained for reproducibility of pre-2026-05-15 runs.
        if _is_polars_input:
            _x_for_cat = X.fill_null(strategy="forward").fill_null(strategy="backward")
        else:
            _x_for_cat = X.ffill().bfill()
        _strategy_for_categorize = "fillna_zero"  # any residual NaN -> 0 (legacy)
    else:
        _x_for_cat = X
        _strategy_for_categorize = self.nan_strategy
    # Propagate the new ``nbins_strategy`` knob through to
    # categorize_dataset so per-column adaptive bin counts (FD, QS, MDLP, Knuth,
    # OptimalJoint, ...) actually take effect inside fit(). When None,
    # categorize_dataset uses the legacy fixed ``quantization_nbins``.
    _nbins_strategy = getattr(self, "nbins_strategy", None)
    _nbins_strategy_kwargs = getattr(self, "nbins_strategy_kwargs", None)
    # When capping cardinality for the compact-codes int8 goal, also bound the NUMERIC side: the supervised MDLP
    # (fayyad_irani) recursion can emit up to 2**max_depth intervals (default max_depth=8 -> ~256), which would exceed
    # int8 just like a high-card categorical. Cap max_depth to floor(log2(cap)) so numeric bins <= cap too (unless the
    # user pinned max_depth explicitly). This makes max_categorical_cardinality a single knob for a universally-narrow
    # codes matrix - categorical tail folded AND numeric intervals bounded.
    _cap = getattr(self, "max_categorical_cardinality", None)
    if _cap and str(_nbins_strategy).lower() in ("mdlp", "fayyad_irani", "mdlp_validated", "fayyad_irani_validated"):
        _md = max(2, int(np.floor(np.log2(int(_cap)))))
        _nbins_strategy_kwargs = dict(_nbins_strategy_kwargs or {})
        _nbins_strategy_kwargs.setdefault("max_depth", _md)
    # Constructor-level shared adaptive-bin-count ceiling (knuth / bayesian_blocks / freedman_diaconis
    # - see MRMR.__init__'s max_adaptive_nbins docstring and _adaptive_nbins.MAX_ADAPTIVE_NBINS).
    # setdefault so an explicit per-method override in nbins_strategy_kwargs (e.g. "knuth_m_max_cap")
    # still wins.
    _max_adaptive_nbins = getattr(self, "max_adaptive_nbins", None)
    if _max_adaptive_nbins is not None:
        _nbins_strategy_kwargs = dict(_nbins_strategy_kwargs or {})
        _nbins_strategy_kwargs.setdefault("max_adaptive_nbins", int(_max_adaptive_nbins))
    # The supervised strategies (mdlp / optimal_joint) need y. Pull the raw
    # target column from the input frame - categorize_dataset is called with
    # _x_for_cat which is a DataFrame; the target column is one of its members
    # (target injection happens upstream in _mrmr_fit_impl).
    _y_for_strategy = None
    if _nbins_strategy is not None and str(_nbins_strategy).lower() in (
        "mdlp",
        "fayyad_irani",
        "mdlp_validated",
        "fayyad_irani_validated",
        "optimal_joint",
        "cv",
        "mah",
        "mah_sci",
        "sci",
        "marx",
    ):
        # Use the first target column as the supervised signal.
        if target_names:
            try:
                if hasattr(_x_for_cat, "to_numpy"):
                    _y_for_strategy = np.asarray(_x_for_cat[target_names[0]])
                else:
                    _y_for_strategy = np.asarray(_x_for_cat[target_names[0]])
            except Exception as exc:
                logger.debug("mrmr: y coercion for discretization-strategy selection failed: %r", exc, exc_info=True)
                _y_for_strategy = None
    data, cols, nbins = categorize_dataset(
        df=_x_for_cat,
        method=self.quantization_method,
        n_bins=self.quantization_nbins,
        dtype=self.quantization_dtype,
        max_categorical_cardinality=getattr(self, "max_categorical_cardinality", None),
        missing_strategy=_strategy_for_categorize,
        nbins_strategy=_nbins_strategy,
        nbins_strategy_kwargs=_nbins_strategy_kwargs,
        y_for_strategy=_y_for_strategy,
        cache_dir=getattr(self, "cache_dir", None),
    )
    logger.info("categorized.")
    return _nbins_strategy, _x_for_cat, cols, data, nbins
