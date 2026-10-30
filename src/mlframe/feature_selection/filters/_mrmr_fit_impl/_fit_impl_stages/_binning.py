"""Discretisation stages of ``_fit_impl``: target re-binning and compact code storage."""

from __future__ import annotations

# --- imports (managed) ---
import hashlib

import numpy as np
from mlframe._array_buffer import array_buffer
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
import os
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
from typing import Optional, Any

# --- end imports ---

_HONEST_NULL_MEMO_BYTES = 128 * 1024 * 1024  # budget of the per-fit memo of re-binned codes under permuted targets


def _validated_target_classes(y: np.ndarray, max_y_classes: int = 64) -> "np.ndarray | None":
    """The target as ``mdlp_bin_edges_oos_validated`` sees it: unchanged with at most ``max_y_classes`` distinct values, otherwise the index of its
    quantile class. ``None`` when the target has non-finite values (the production path drops those rows; the null is then left to the
    categorise-per-permutation path)."""
    y = np.asarray(y)
    if y.dtype.kind in "OUS":
        return y if np.unique(y).size <= max_y_classes else None
    if y.dtype.kind == "f" and not np.isfinite(y).all():
        return None
    if np.unique(y).size <= max_y_classes:
        return y
    edges = np.unique(np.quantile(y, np.linspace(0.0, 1.0, int(max_y_classes) + 1)[1:-1]))
    return np.searchsorted(edges, y, side="right")


def _target_dependent_columns(x, cols) -> "set[str] | None":
    """Names among ``cols`` whose supervised codes can change when the target is permuted.

    Those are the numeric columns with more distinct finite values than the low-cardinality cap of the supervised binner. Every other column (categorical, or numeric with few
    values) is binned without the target, so its observed codes are also its codes under any permutation of the target. ``None`` when ``x`` is not a pandas frame (the caller then
    re-bins every column).
    """
    import pandas as pd

    from mlframe.feature_selection.filters._honest_spec import _LOW_CARD_CAP

    if not isinstance(x, pd.DataFrame):
        return None
    return {str(c) for c in cols if c in x.columns and pd.api.types.is_numeric_dtype(x[c]) and x[c].nunique(dropna=True) > _LOW_CARD_CAP}


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
            _t_finite = _t_raw[np.isfinite(_t_raw.astype(_crit_np_dtype()))] if _t_raw.dtype.kind == "f" else _t_raw
            if np.unique(_t_finite).size <= int(self.quantization_nbins):
                continue  # discrete / classification target: keep its native classes
            _t_codes = _t_discretize(
                arr=_t_raw.astype(_crit_np_dtype()),
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
        # Legacy path retained so runs from before the separate-bin NaN default stay reproducible.
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
    _ret_discretize_inputs_part1 = _discretize_inputs_part1(self, _nbins_strategy, target_names, _x_for_cat, categorize_dataset, _strategy_for_categorize)
    if _ret_discretize_inputs_part1[0]:
        return _ret_discretize_inputs_part1[1]


def _install_honest_null_hook(
    self: Any, *, categorize_dataset: Any, _x_for_cat: Any, cols: Any, data: Any, nbins: Any, _y_for_strategy: Any, _nbins_strategy: Any,
    _nbins_strategy_kwargs: Any, _strategy_for_categorize: Any,
) -> None:
    """Attach the target-permutation null of supervised binning to ``self``: the codes of each permutation are re-binned on the permuted raw target."""
    _y_raw_for_null = np.asarray(_y_for_strategy)

    # Only the columns whose codes depend on the target are re-binned per permutation; the others keep their observed codes (copied now, so a later in-place
    # edit of ``data`` cannot change them).
    _dependent_names = _target_dependent_columns(_x_for_cat, cols)
    _dependent: set = _dependent_names if _dependent_names is not None else set()
    _fixed_codes = None
    if _dependent_names is not None and len(_dependent) < len(cols):
        _fixed_codes = {str(c): (np.array(data[:, j]), int(nbins[j])) for j, c in enumerate(cols) if str(c) not in _dependent}

    # The cross-validated bin-count search (``optimal_joint``) splits and bins the SAME column for every permutation; only the target changes. A plan cache keeps the target-independent part per
    # column, so a permutation costs the fold scoring only (identical edges; MLFRAME_OJ_PLAN_CACHE=0 restores the uncached search).
    _null_strategy_kwargs = _nbins_strategy_kwargs
    if str(_nbins_strategy).lower() in ("optimal_joint", "cv") and os.environ.get("MLFRAME_OJ_PLAN_CACHE", "1") != "0":
        from mlframe.feature_selection.filters._adaptive_nbins import OptimalJointPlanCache

        _null_strategy_kwargs = {**(_nbins_strategy_kwargs or {}), "optimal_joint_plan_cache": OptimalJointPlanCache()}

    def _honest_null_codes_uncached(row_perm):
        """Codes of every column re-binned on the RAW target permuted by ``row_perm`` (same columns, same order as ``data``)."""
        y_perm = _y_raw_for_null[np.asarray(row_perm, dtype=np.int64)]
        d_perm, _c_perm, nb_perm = categorize_dataset(
            df=_x_for_cat if _fixed_codes is None else _x_for_cat[[c for c in _x_for_cat.columns if str(c) in _dependent]],
            method=self.quantization_method,
            n_bins=self.quantization_nbins,
            dtype=self.quantization_dtype,
            max_categorical_cardinality=getattr(self, "max_categorical_cardinality", None),
            missing_strategy=_strategy_for_categorize,
            nbins_strategy=_nbins_strategy,
            nbins_strategy_kwargs=_null_strategy_kwargs,
            y_for_strategy=np.asarray(y_perm),
            cache_dir=None,
        )
        # Align to the original column order by name: a permuted-target categorisation may drop a column, and the
        # null must read the same column index the observed codes use. A dropped column carries no MI (0 bins).
        _pos = {str(c): j for j, c in enumerate(_c_perm)}
        _n_rows = d_perm.shape[0]
        _codes_aligned = np.zeros((_n_rows, len(cols)), dtype=np.int64)
        _nb_aligned = np.ones(len(cols), dtype=np.int64)
        for j, c in enumerate(cols):
            k = _pos.get(str(c))
            if k is not None:
                _codes_aligned[:, j] = d_perm[:, k]
                _nb_aligned[j] = int(nb_perm[k])
            elif _fixed_codes is not None and str(c) in _fixed_codes:
                _codes_aligned[:, j], _nb_aligned[j] = _fixed_codes[str(c)]
        return _codes_aligned, _nb_aligned

    # The screen floor is computed once before and once after the feature-engineering step, with the same seed and row count, hence the same sequence of row
    # permutations, and the re-binned codes of the input columns depend on nothing else: the second pass would repeat every re-binning of the first. Memoise by the
    # content of the permutation, stored narrow, up to a byte budget (a larger frame is simply not cached). The cache is dropped at the end of the fit.
    _memo: dict = {}
    _memo_bytes = [0]

    def _honest_null_codes(row_perm):
        """``_honest_null_codes_uncached`` memoised by the content of ``row_perm``; every call returns fresh int64 arrays."""
        perm = np.ascontiguousarray(np.asarray(row_perm, dtype=np.int64))
        key = hashlib.blake2b(array_buffer(perm), digest_size=16).digest()
        hit = _memo.get(key)
        if hit is not None:
            return hit[0].astype(np.int64), hit[1].copy()
        codes, nb = _honest_null_codes_uncached(perm)
        compact = codes.astype(np.int16) if codes.size == 0 or int(codes.max()) < np.iinfo(np.int16).max else codes
        if _memo_bytes[0] + compact.nbytes + nb.nbytes <= _HONEST_NULL_MEMO_BYTES:
            _memo[key] = (compact, nb.copy())
            _memo_bytes[0] += compact.nbytes + nb.nbytes
        return codes, nb

    def _clear_memo():
        """Drop the memoised re-binned codes (called when the fit ends)."""
        _memo.clear()
        _memo_bytes[0] = 0

    _honest_null_codes.clear_memo = _clear_memo  # type: ignore[attr-defined]

    # Numba spec for the honest null (see _mdlp_honest_njit): only the plain 'mdlp' (fayyad_irani) strategy with no extra kwargs is
    # covered by the njit chain, whose identity with categorize_dataset was verified on continuous columns; other strategies keep
    # the categorise-per-permutation path.
    _kw_keys = set((_nbins_strategy_kwargs or {}).keys())
    _strategy_lc = str(_nbins_strategy).lower()
    # ``mdlp_fast_mode=False`` is the production default, so spelling it out selects the same categorisation; only ``True`` (the
    # depth-capped path) is outside the njit chain.
    _fast_mode_off = not bool((_nbins_strategy_kwargs or {}).get("mdlp_fast_mode", False))
    if (
        _strategy_lc in ("mdlp", "fayyad_irani", "mdlp_validated", "fayyad_irani_validated")
        and _kw_keys <= {"max_depth", "max_adaptive_nbins", "mdlp_fast_mode"}
        and (_fast_mode_off or _kw_keys <= {"max_depth", "max_adaptive_nbins"})
    ):
        from mlframe.feature_selection.filters._honest_spec import build_honest_spec

        _validated = _strategy_lc in ("mdlp_validated", "fayyad_irani_validated")
        # The validated strategy bins against at most 64 target classes: a target with more distinct values is first cut into 64 quantile classes
        # (mdlp_bin_edges_oos_validated). The class of a row does not depend on the row order, so the permuted target's classes are the permuted
        # classes of the observed target and the null can run on those.
        _y_spec = _validated_target_classes(_y_raw_for_null) if _validated else _y_raw_for_null
        if _y_spec is not None:
            setattr(  # noqa: B010 - the nested function carries its spec as an attribute
                _honest_null_codes,
                "spec",
                build_honest_spec(
                    X=_x_for_cat,
                    cols=list(cols),
                    y_raw=_y_spec,
                    max_depth=int((_nbins_strategy_kwargs or {}).get("max_depth", 8)),
                    method="oos" if _validated else "mdlp",
                ),
            )
    self._honest_null_codes_fn = _honest_null_codes


def _discretize_inputs_part1(
    self: Any, _nbins_strategy: Any, target_names: Any, _x_for_cat: Any, categorize_dataset: Any, _strategy_for_categorize: Any,
) -> tuple[Any, ...]:
    """Extracted block of ``_discretize_inputs`` (part 1); returns the names the caller still uses."""
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
                logger.warning("mrmr: y coercion for discretization-strategy selection failed: %r", exc, exc_info=True)
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
    logger.debug("categorized.")
    # Supervised binning fits its cut points on the target, so the significance null must re-bin on every permuted target with the SAME
    # production categorisation; a null that permutes y over fixed codes cannot see the in-sample bias of those cuts. The closure is
    # stripped from pickles (see MRMR.__getstate__).
    _supervised = _y_for_strategy is not None and str(_nbins_strategy).lower() in (
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
    )
    if _supervised:
        _install_honest_null_hook(
            self,
            categorize_dataset=categorize_dataset,
            _x_for_cat=_x_for_cat,
            cols=cols,
            data=data,
            nbins=nbins,
            _y_for_strategy=_y_for_strategy,
            _nbins_strategy=_nbins_strategy,
            _nbins_strategy_kwargs=_nbins_strategy_kwargs,
            _strategy_for_categorize=_strategy_for_categorize,
        )
    else:
        self._honest_null_codes_fn = None
    return True, (_nbins_strategy, _x_for_cat, cols, data, nbins)
