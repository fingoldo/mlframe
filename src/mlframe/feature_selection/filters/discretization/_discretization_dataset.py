"""Dataset-level discretisation entry points.

`categorize_dataset` is the top-level entry called by ``MRMR.fit`` (pandas /
polars, NaN-aware, optional adaptive per-feature binning). It leans on the
numeric-column kernels and missing-value handling in the parent module
``discretization``; those are lazy-imported in-body to avoid an import cycle.
`create_redundant_continuous_factor` is a test / benchmark synthetic-data helper.
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _discretize_input_dtype():
    """Working dtype for the numeric matrix that ``categorize_dataset`` discretises.

    ``categorize_dataset`` copies ALL numeric columns into one dense array before binning
    (``arr = df[...].to_numpy(...)``) -- a second full-frame copy that coexists with the caller's
    (already large) engineered float frame, and is the dominant term of the large-n FE peak (a 1M-row
    fit projects to ~21GB, OOMing a 16GB box). Quantile / MDLP edges + searchsorted do NOT need
    float64: float32 edges differ only at ~1e-7 (far below the selection-altering ~1e-3 bar) and only
    for values sitting exactly on a bin edge. ``MLFRAME_DISCRETIZE_FLOAT32=1`` halves that copy.
    Default float64 (byte-for-byte legacy) so it is opt-in + reversible until validated default-on.
    """
    if os.environ.get("MLFRAME_DISCRETIZE_FLOAT32", "").strip() in ("1", "true", "True"):
        return np.float32
    return np.float64


def create_redundant_continuous_factor(
    df: pd.DataFrame,
    factors: Sequence[str],
    agg_func: object = np.sum,
    noise_percent: float = 5.0,
    dist: object = None,
    dist_args: tuple = (),
    name: str = None,
    sep: str = "_",
) -> None:
    """Out of a few continuous factors, craft a new factor with known relationship and amount of redundancy. Used by tests / benchmark harnesses, not by ``MRMR`` directly."""
    if dist:
        rvs = dist.rvs
        # Wave 31 (2026-05-20): assert -> AttributeError.
        if not callable(rvs):
            raise AttributeError(
                f"dist must have a callable .rvs method; got {dist!r}."
            )
        noise = rvs(*dist_args, size=len(df))
    else:
        noise = np.random.random(len(df))

    val_min, val_max = noise.min(), noise.max()
    if np.isclose(val_max, val_min):
        noise = np.zeros(len(noise), dtype=np.float32)
    else:
        noise = (noise - val_min) / (val_max - val_min)

    if not name:
        name = sep.join(factors) + sep + f"{noise_percent:.0f}%{dist.name if dist else ''}noise"

    df[name] = agg_func(df[factors].values, axis=1) * (1 + (noise - 0.5) * noise_percent / 100)


def categorize_dataset(
    df,
    method: str = "quantile",
    n_bins: int = 4,
    min_ncats: int = 50,
    dtype=np.int16,
    missing_strategy: str = "fillna_zero",
    nbins_strategy: str = None,
    nbins_strategy_kwargs: dict = None,
    y_for_strategy=None,
    cache_dir: str = None,
):
    """Convert a DataFrame into an ordinal-encoded ``(n_samples, n_features)`` array. Accepts pandas or polars (DataFrame or LazyFrame -- materialised at the
    boundary). ``missing_strategy`` controls NaN handling: see :func:`_handle_missing`."""
    from . import (
        _handle_missing,
        _maybe_collect_lazy,
        _multi_col_factorize_native,
        discretize_2d_array,
    )

    df = _maybe_collect_lazy(df)

    data = None
    numerical_cols = []
    categorical_factors = []

    try:
        import polars as pl
        _is_polars = isinstance(df, pl.DataFrame)
    except ImportError:
        _is_polars = False

    if _is_polars:
        def _is_pl_cat(dt):
            return (
                dt == pl.Utf8
                or dt == pl.String
                or dt == pl.Categorical
                or dt == pl.Boolean
                or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))
            )
        numerical_cols = [name for name, dt in df.schema.items() if not _is_pl_cat(dt)]
        categorical_cols_detected = [name for name, dt in df.schema.items() if _is_pl_cat(dt)]
    else:
        numerical_cols = df.head(5).select_dtypes(exclude=("category", "object", "string", "bool")).columns.values.tolist()
        categorical_cols_detected = None

    _dt = _discretize_input_dtype()
    if _is_polars:
        _num_frame = df.select(numerical_cols)
        arr = _num_frame.to_numpy().astype(_dt, copy=False)
    else:
        arr = df[numerical_cols].to_numpy(dtype=_dt, na_value=np.nan)

    # Snapshot the NaN positions BEFORE _handle_missing rewrites them: the
    # "separate_bin" strategy fills NaN with the column median so np.percentile
    # produces clean edges, then we overwrite the same positions in the
    # discretized output with bin=n_bins (max+1 per column). Net effect: NaN
    # gets its own honest category that MI estimators see correctly.
    # 2026-05-30 Wave 9.1 fix (loop iter 11): include 'propagate' alongside
    # 'separate_bin' so NaN positions get re-routed to the dedicated NaN
    # bin instead of silently colliding with the top real bin via
    # np.searchsorted(NaN -> ej.size).
    _nan_mask = (
        np.isnan(arr)
        if (missing_strategy in ("separate_bin", "propagate") and arr.size > 0)
        else None
    )

    # Unified NaN handling for both pandas and polars.
    arr = _handle_missing(arr, strategy=missing_strategy)

    # 2026-05-29 Wave 7: per-column adaptive bin chooser.
    # When ``nbins_strategy`` is provided, compute per-column edges via the
    # _adaptive_nbins dispatcher, apply them with np.searchsorted, and pad to
    # the global max nbins so downstream MRMR sees a uniform-nbins matrix.
    if nbins_strategy is not None:
        from .._adaptive_nbins import per_feature_edges
        _strategy_kwargs = dict(nbins_strategy_kwargs or {})
        # Pass y if the strategy is supervised.
        _needs_y = str(nbins_strategy).lower() in (
            "mdlp", "fayyad_irani", "optimal_joint", "cv",
            "mah", "mah_sci", "sci", "marx",
        )
        _y_arr = None
        if _needs_y and y_for_strategy is not None:
            _y_arr = np.asarray(y_for_strategy).ravel()
        edges_per_col = per_feature_edges(
            arr, y=_y_arr, method=nbins_strategy, cache_dir=cache_dir, **_strategy_kwargs,
        )
        # Per-column searchsorted; pad to global max nbins.
        n_rows = arr.shape[0]
        n_cols = arr.shape[1]
        per_col_bins = [int(e.size + 1) for e in edges_per_col]
        max_bins = max(max(per_col_bins) if per_col_bins else 1, 1)
        # Validate the requested dtype can hold ``max_bins`` (matches the
        # post-discretize NaN-bin overflow check below).
        if max_bins > np.iinfo(dtype).max:
            raise ValueError(
                f"nbins_strategy={nbins_strategy!r} produced {max_bins} bins which "
                f"exceeds dtype {dtype} max {np.iinfo(dtype).max}. "
                f"Use a wider dtype or constrain the strategy (e.g. knuth_m_max_cap=64)."
            )
        data = np.empty((n_rows, n_cols), dtype=dtype)
        for j in range(n_cols):
            ej = edges_per_col[j]
            if ej.size == 0:
                data[:, j] = 0
            else:
                data[:, j] = np.searchsorted(ej, arr[:, j].astype(np.float64),
                                              side="right").astype(dtype)
    else:
        data = discretize_2d_array(
            arr=arr, n_bins=n_bins, method=method, min_ncats=min_ncats,
            min_values=None, max_values=None, dtype=dtype,
        )

    if _nan_mask is not None and _nan_mask.any():
        # 2026-05-30 Wave 9.1 fix (loop iter 9): per-COLUMN NaN bin code.
        # Pre-fix used the constructor ``n_bins`` as the dedicated NaN code
        # for every column, but the adaptive ``nbins_strategy`` branch
        # produces per-column bin counts that often exceed ``n_bins``
        # (e.g. FD gives ~22 for n=600 N(0,1), while ctor n_bins=4). So the
        # NaN code 4 silently collided with regular real-data bin 4 - NaN
        # observations got merged into a real bin, destroying the
        # missingness signal and biasing every downstream MI / SU / MRMR
        # score. Fix: each column's NaN code is one past that column's
        # highest regular code. Per-column scheme works because downstream
        # MI estimators treat each column independently and
        # ``data.max(axis=0) + 1`` (line 1151) recomputes ``nbins`` per col.
        if nbins_strategy is not None:
            nan_codes_per_col = np.asarray(per_col_bins, dtype=np.int64)
        else:
            nan_codes_per_col = np.full(arr.shape[1], int(n_bins), dtype=np.int64)
        max_bin_after = int(nan_codes_per_col.max())
        if max_bin_after > np.iinfo(data.dtype).max:
            raise ValueError(
                f"separate_bin strategy needs dtype able to hold {max_bin_after}; "
                f"current dtype {data.dtype} max is {np.iinfo(data.dtype).max}. "
                "Pass a wider dtype to categorize_dataset."
            )
        # Per-column NaN code: broadcast across NaN-row positions.
        _rows, _c = np.where(_nan_mask)
        data[_rows, _c] = nan_codes_per_col[_c].astype(data.dtype)

    if _is_polars:
        if categorical_cols_detected:
            cast_exprs = []
            for c in categorical_cols_detected:
                dt = df.schema[c]
                if dt == pl.Boolean:
                    cast_exprs.append(pl.col(c).cast(pl.UInt32))
                elif dt in (pl.Utf8, pl.String):
                    cast_exprs.append(pl.col(c).cast(pl.Categorical).to_physical())
                else:
                    cast_exprs.append(pl.col(c).to_physical())
            _coded = df.select(cast_exprs)
            categorical_cols = categorical_cols_detected
            new_vals = _coded.to_numpy()
        else:
            categorical_cols = []
            new_vals = None
    else:
        categorical_factors = df.select_dtypes(include=("category", "object", "string", "bool"))
        categorical_cols = []
        if categorical_factors.shape[1] > 0:
            categorical_cols = categorical_factors.columns.values.tolist()
            new_vals = _multi_col_factorize_native(categorical_factors)
        else:
            new_vals = None
    if categorical_cols and new_vals is not None:
        # 2026-05-30 Wave 9.1 fix (loop iter 31): the categorical block
        # bypassed ``missing_strategy`` entirely. ``_multi_col_factorize_native``
        # / ``pd.factorize`` / ``.cat.codes`` emit ``-1`` for NaN, which then
        # silently flowed into the joint-histogram allocator and got
        # negative-index wrapped to the LAST real category bin (or, under
        # unsigned dtype, wrapped to 2^bits - 1 = a phantom huge category).
        # Net effect: NaN observations silently merged with the largest
        # real category, biasing every MI / SU / MRMR score on columns
        # with NaN in pd.Categorical / object / string / bool columns.
        # Sibling of iter 9 (numeric NaN bin collision) and iter 11
        # (propagate strategy silent merge).
        #
        # Fix: shift codes by +1 so NaN sentinel becomes 0 and real
        # categories become 1..K. Under ``missing_strategy='separate_bin'``
        # (the default) this gives NaN its own honest bin. Under
        # 'fillna_zero' the shift is equivalent: NaN ends up at bin 0
        # which any downstream code reading "0 = first category" treats
        # uniformly. Under 'raise', refuse if any -1 sentinel present.
        if _missing_strategy_str := str(missing_strategy):
            _has_nan = bool((new_vals < 0).any())
            if _has_nan and _missing_strategy_str == "raise":
                _nan_cnt = int((new_vals < 0).sum())
                raise ValueError(
                    f"categorize_dataset: {_nan_cnt} NaN value(s) in "
                    f"categorical column(s) {categorical_cols} with "
                    f"missing_strategy='raise'."
                )
            if _has_nan:
                # Shift +1: -1 -> 0, k -> k+1. Cast back to dtype after
                # shift (the shift increases the max by 1; auto-promote
                # below catches dtype overflow on the new max).
                new_vals = new_vals + 1
        max_cats = new_vals.max(axis=0)
        global_max = int(max_cats.max())
        if global_max > np.iinfo(dtype).max:
            for _candidate in (np.int16, np.int32, np.int64):
                if global_max <= np.iinfo(_candidate).max:
                    logger.warning(
                        "categorize_dataset: %d category code(s) exceeded dtype %s; auto-promoting to %s to avoid silent wraparound.",
                        int((max_cats > np.iinfo(dtype).max).sum()),
                        dtype,
                        _candidate,
                    )
                    dtype = _candidate
                    break
            else:
                raise ValueError(
                    f"categorize_dataset: category cardinality {global_max} exceeds int64 max; cannot encode."
                )
        new_vals = new_vals.astype(dtype)

        if data is None:
            data = new_vals
        else:
            data = np.append(data, new_vals, axis=1)

    nbins = data.max(axis=0).astype(np.int64) + 1

    return data, numerical_cols + categorical_cols, nbins
