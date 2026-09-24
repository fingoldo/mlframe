"""Feature filter for :class:`CompositeTargetDiscovery`.

Carved out of ``composite_discovery`` via method-rebinding to keep the parent
facade under the LOC budget. Bound onto the class at the parent module's bottom.
"""
from __future__ import annotations

import logging
from typing import Any, List, Sequence, cast

import numpy as np

from .._composite_utils import is_polars_df as _is_polars_df
from .screening import (
    _extract_column_array,
    _is_numeric_column,
    _safe_abs_corr_all,
)

logger = logging.getLogger(__name__)


# Minimum rows kept by the leak-corr adaptive sampler. Pearson correlation
# converges to ~0.001 absolute precision at 100K rows; 500K is a 2x safety
# margin so even at the smallest sampled regime the corr estimate stays a
# faithful proxy for the full-frame number. Falling below this hides
# legitimate strong correlates instead of filtering them.
_LEAK_CORR_MIN_SAMPLE_ROWS = 500_000


def _leak_corr_sample_rows(n_rows: int) -> np.ndarray | None:
    """Row positions the leak-corr test reads, or ``None`` when every row is kept.

    The test is ``|corr(x, y)| >= 0.99999``, whose standard error near 1 is ``(1 - r**2) / sqrt(n)``: past the minimum
    sample it is decided identically on a stride of the rows. Choosing them BEFORE the columns are gathered is what
    bounds the peak: the previous order held every numeric column over every train row, then stacked a second full copy,
    and only sampled once both were already allocated -- about 14 GB on a 3.2M x 500 frame.
    """
    if n_rows <= _LEAK_CORR_MIN_SAMPLE_ROWS:
        return None
    stride = max(2, n_rows // _LEAK_CORR_MIN_SAMPLE_ROWS)
    return np.arange(0, n_rows, stride)


# Headroom guard the sampler enforces. The leak-corr matrix is materialised at
# ``rows * cols * 4 B`` in one shot (column_stack copy); we sample down when
# that single allocation would consume more than this fraction of currently-
# available physical RAM. 0.30 = leave 70% of available for the rest of the
# discovery pipeline (per-base x_remaining matrices, MI tables, tiny-model
# Datasets). Tuned for the user's 128 GB host where the prior full-frame
# 6.41 GB allocation tripped numpy's MemoryError on virtual-address-space
# fragmentation despite ~20 GB free physical RAM.
_LEAK_CORR_ALLOC_AVAIL_FRACTION = 0.30


def _maybe_sample_for_leak_corr(
    candidates: list[str],
    candidate_arrays: list[np.ndarray],
    y_train: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Adaptive stride-sample for the leak-corr filter's full-frame allocation.

    Builds the (rows, cols) leak-corr matrix at full precision when the host
    has plenty of free RAM; falls back to a deterministic stride-subsample
    when the single allocation would exceed ``_LEAK_CORR_ALLOC_AVAIL_FRACTION``
    of currently-available physical RAM.

    Why "available" and not "total": on a long-running Jupyter kernel the
    process can carry tens of GB of committed-but-paged-out buffers (pyarrow
    pools, numba JIT cache, library .dll/.so mappings) that don't appear in
    USS/RSS but do count toward virtual-address-space fragmentation. numpy's
    ``np.column_stack`` requires ONE contiguous virtual block at the matrix
    size; on a 128 GB host with 20 GB free physical the next 6.4 GB contiguous
    request can MemoryError purely from address-space fragmentation. Sampling
    the rows down keeps the allocation under the fragmentation ceiling.

    Why stride and not random: correlation estimation is rotation-invariant
    so systematic sampling is unbiased for Pearson |corr|; stride is also
    fully reproducible across runs (no random_state plumbing required) and
    keeps the lag-feature ordering intact for downstream consumers that
    might inspect the survivor arrays.

    Returns the (possibly resampled) candidate_arrays + the (possibly
    resampled) y_train. No-op when sampling is unnecessary -- the caller
    receives the original arrays back so the bit-identical full-frame path
    is preserved on hosts with adequate RAM.
    """
    if not candidate_arrays:
        return candidate_arrays, y_train
    n_rows = candidate_arrays[0].shape[0]
    n_cols = len(candidate_arrays)
    # Stacked-matrix footprint at float32 (the dtype _extract_column_array
    # returns) is what np.column_stack actually allocates.
    needed_bytes = n_rows * n_cols * 4
    try:
        import psutil as _psutil
        available_bytes = int(_psutil.virtual_memory().available)
    except Exception as exc:
        # No psutil -> can't measure -> trust the legacy full-frame path. The
        # caller's existing try/except for MemoryError still catches this case;
        # we just don't have a way to AVOID the OOM here without psutil.
        logger.debug("leak-corr matrix sizing: psutil probe failed, trusting the full-frame path: %s", exc)
        return candidate_arrays, y_train
    if needed_bytes <= _LEAK_CORR_ALLOC_AVAIL_FRACTION * available_bytes:
        return candidate_arrays, y_train
    # Sampling needed. Pick the largest stride such that the resulting matrix
    # fits the headroom band; clamp to the corr-precision floor.
    budget_bytes = int(_LEAK_CORR_ALLOC_AVAIL_FRACTION * available_bytes)
    target_rows = max(_LEAK_CORR_MIN_SAMPLE_ROWS, budget_bytes // (n_cols * 4))
    target_rows = min(target_rows, n_rows)
    if target_rows >= n_rows:
        # Headroom-derived target exceeds row count -- the prior check already
        # decided we're tight; clamp to a sane sample anyway so we don't
        # full-frame-allocate on a fragmented address space.
        target_rows = min(n_rows, max(_LEAK_CORR_MIN_SAMPLE_ROWS, n_rows // 2))
    stride = max(1, n_rows // target_rows)
    sample_idx = np.arange(0, n_rows, stride)
    if sample_idx.size > target_rows:
        sample_idx = sample_idx[:target_rows]
    sampled = [arr[sample_idx] for arr in candidate_arrays]
    y_sampled = y_train[sample_idx] if y_train is not None and y_train.shape[0] == n_rows else y_train
    logger.info(
        "[CompositeTargetDiscovery] leak-corr matrix sampled from %d to %d rows "
        "(stride=%d): full-frame alloc %.2f GB would exceed %.0f%% of %.2f GB "
        "currently-available RAM, sampled alloc %.2f GB stays under the cap. "
        "Pearson |corr| precision at %d rows is ~1e-3, within the leak-filter "
        "threshold tolerance.",
        n_rows, sample_idx.size, stride,
        needed_bytes / 1024 ** 3,
        _LEAK_CORR_ALLOC_AVAIL_FRACTION * 100,
        available_bytes / 1024**3,
        (sample_idx.size * n_cols * 4) / 1024**3,
        sample_idx.size,
    )
    return sampled, y_sampled


# |corr| within this of the threshold is recomputed in float64: the vectorised float32 pass is only good to ~1e-5.
_LEAK_CORR_RECHECK_BAND = 1e-4


def _leak_corr_survivors(self, candidates, candidate_arrays, _y_leak, drops, corr_drops):
    """Split the survivors into kept columns and leak-corr drops, appending each drop to ``drops``/``corr_drops``.

    Carved out of ``_filter_features`` so the gather loop and the correlation decision stay separately readable;
    the rules are unchanged.
    """
    # Vectorised corr filter on survivors. Replaces the per-column
    # ``abs(_safe_corr(arr, y_train))`` loop. NaN rows in the survivor matrix
    # are imputed with column-mean before the corr-vs-y dot product, which is
    # a small approximation versus per-column NaN masking but only matters for
    # columns with sparse NaN -- and those have already passed the
    # ``finite_mask.sum() < 50`` gate above with at least 50 finite rows.
    # Acceptable trade-off for the ~600ms saving on 200-feature filter calls.
    kept: list[str] = []
    if candidates:
        # Adaptive headroom-aware sampler: full-frame on RAM-rich hosts, stride-
        # subsample when the single column_stack alloc would crowd available RAM.
        # The corr estimate at the sampled regime stays within ~1e-3 of full-frame
        # precision, well inside the leak-filter threshold tolerance. See helper
        # docstring for the why/why-not analysis.
        _sampled_arrays, _y_for_corr = _maybe_sample_for_leak_corr(
            candidates, candidate_arrays, _y_leak,
        )
        X_train = np.column_stack(_sampled_arrays)
        # Free the per-column ndarrays the moment they land in the stacked matrix:
        # candidate_arrays holds (n_features) views/copies that double the peak
        # footprint until we let them go (~8 GB on a 4M-row x 500-col float32 frame).
        candidate_arrays.clear()
        _sampled_arrays = []
        # nanmean over (N, F) requires no temp; nb the prior np.where(isfinite, X, nan)
        # built a SECOND full-frame copy purely to silence non-finite cells, redundant.
        col_means = np.nanmean(X_train, axis=0)
        non_finite_mask = ~np.isfinite(X_train)
        if non_finite_mask.any():
            # X_train is a freshly-allocated buffer owned by this function; mutating
            # in-place is safe (the .copy() removed here cost another full-frame
            # allocation -- ~8 GB transient on the 4M-row prod frame).
            X_train[non_finite_mask] = np.broadcast_to(
                col_means, X_train.shape,
            )[non_finite_mask]
        abs_corrs = _safe_abs_corr_all(_y_for_corr, X_train)
        # Mean-imputation dilutes |corr| by ~sqrt(frac_finite) for NaN-bearing
        # columns (imputed rows contribute 0 to the centred cross-product but
        # inflate the variance denominator). With the near-1 forbidden-base
        # threshold this lets an exact y-copy carrying even a handful of NaN
        # rows slip the leak gate and become the composite base. Recompute
        # those columns EXACTLY with per-pair finite masking in float64 (cheap:
        # only NaN-bearing columns, and they already cleared the >=50-finite
        # gate). float64 also clears the float32 ~1e-5 accumulation that sits
        # inside the threshold band.
        threshold = float(self.config.forbidden_base_corr_threshold)
        # The vectorised pass is float32, whose error (~1e-6 here, 1.8e-7 above 1 on a near-copy of y) is the same size as
        # 1 - threshold: a legitimate near-copy scored |corr| = 1.00000018, above any threshold an operator could raise it
        # to, so the logged advice to raise the threshold could not work. Columns within the band of the threshold are
        # recomputed exactly, as the NaN-bearing ones are.
        col_has_nan = non_finite_mask.any(axis=0)
        recheck = col_has_nan | (np.asarray(abs_corrs) >= threshold - _LEAK_CORR_RECHECK_BAND)
        if recheck.any():
            y64 = np.asarray(_y_for_corr, dtype=np.float64)
            y_ok = np.isfinite(y64)
            for j in np.nonzero(recheck)[0]:
                finite_rows = (~non_finite_mask[:, j]) & y_ok
                if int(finite_rows.sum()) < 3:
                    continue
                xj = X_train[finite_rows, j].astype(np.float64)
                yj = y64[finite_rows]
                x_dev = xj - xj.mean()
                y_dev = yj - yj.mean()
                var_x = float(np.dot(x_dev, x_dev))
                var_y = float(np.dot(y_dev, y_dev))
                if var_x < 1e-24 or var_y < 1e-24:
                    continue
                abs_corrs[j] = min(1.0, abs(float(np.dot(x_dev, y_dev)) / np.sqrt(var_x * var_y)))
        for col, corr_val in zip(candidates, abs_corrs.tolist()):
            if corr_val >= threshold:
                drops.append({
                    "name": col, "reason": "forbidden_base_corr_threshold",
                    "corr": float(corr_val), "threshold": threshold,
                })
                corr_drops.append((col, float(corr_val)))
            else:
                kept.append(col)
    return kept


def _polars_train_stats(self, df, feature_cols, train_idx) -> dict | None:
    """``{column: (finite count, float32 max - min over the finite values or None)}`` over the train rows of a polars frame,
    in one aggregation, or None when the per-column path is the faster one.

    The per-column loop pulls every numeric column over every train row into numpy for a count and a range. The values
    are cast to float32 in the query, as ``_extract_column_array`` casts them, and the range is a float32 subtraction as
    ``np.ptp`` on that array computes it, so every drop decision is the same. Measured at 1M x 200: when the train rows are
    the whole frame (discovery's usual case, the suite passes its filtered train frame) one query takes 0.98 s against
    2.44 s, at +108 MB peak against +10 MB; when rows must be gathered it is slower (2.08 s against 1.84 s), so that case
    keeps the per-column path. The caller uses it only when the leak-corr test samples rows (frames above
    ``_LEAK_CORR_MIN_SAMPLE_ROWS``); below that every full column is read for the correlation anyway.
    """
    if not _is_polars_df(df):
        return None
    import polars as pl

    cols = [c for c in feature_cols if c != self._target_col and not any(p.search(c) for p in self._patterns_compiled)
            and _is_numeric_column(df, c)]
    if not cols:
        return {}
    idx = np.asarray(train_idx)
    if not (idx.size == df.height and np.array_equal(idx, np.arange(df.height))):
        return None  # a row gather makes the batched query slower than the per-column path
    exprs = []
    for k, c in enumerate(cols):
        v = pl.col(c).cast(pl.Float32)
        fin = v.is_finite().fill_null(False)
        exprs += [fin.sum().alias(f"n{k}"), v.filter(fin).min().alias(f"lo{k}"), v.filter(fin).max().alias(f"hi{k}")]
    row = df.select(exprs).row(0)
    out = {}
    for k, c in enumerate(cols):
        n, lo, hi = row[3 * k], row[3 * k + 1], row[3 * k + 2]
        out[c] = (int(n or 0), None if lo is None else float(np.float32(hi) - np.float32(lo)))
    return out


def _filter_features(
    self,
    df: Any,
    feature_cols: Sequence[str],
    y_train: np.ndarray,
    train_idx: np.ndarray,
) -> list[str]:
    """Drop columns that are non-numeric, near-constant on train, match a
    forbidden name pattern, or correlate suspiciously highly with y on
    train (likely derived-from-y leakage).

    Drops are recorded on ``self._filter_drops`` (list of dicts with name +
    reason + value) so :meth:`fit` can surface them in the report and so
    callers can audit false positives -- the corr filter in particular is
    prone to misfiring on legitimate autoregressive lag features such as
    a ``y_prev`` column.
    """
    # First pass: cheap-fail filters (name patterns, type, finite count,
    # near-constant). Build a list of survivors + their train-row arrays so the
    # corr check can be vectorised across all survivors in ONE matrix op
    # (~2.2x faster vs per-column ``_safe_corr`` loop on 200 cols x 80K rows).
    drops: list[dict[str, Any]] = []
    corr_drops: list[tuple[str, float]] = []
    candidates: list[str] = []
    candidate_arrays: list[np.ndarray] = []
    _leak_rows = _leak_corr_sample_rows(int(np.asarray(train_idx).size))
    _y_leak = y_train if _leak_rows is None or y_train is None else np.asarray(y_train)[_leak_rows]
    if _leak_rows is not None:
        logger.info(
            "[CompositeTargetDiscovery] leak-corr test reads a %d-row stride of the %d train rows; the constancy and "
            "finite-row checks still read every row, so only the correlation is sampled.",
            _leak_rows.size, int(np.asarray(train_idx).size),
        )
    # Only when the leak-corr test samples rows: otherwise every full column is read for it anyway.
    _polars_stats = _polars_train_stats(self, df, feature_cols, train_idx) if _leak_rows is not None else None
    for col in feature_cols:
        if col == self._target_col:
            continue
        if any(p.search(col) for p in self._patterns_compiled):
            drops.append({"name": col, "reason": "forbidden_pattern"})
            continue
        if not _is_numeric_column(df, col):
            drops.append({"name": col, "reason": "non_numeric"})
            continue
        if _polars_stats is not None and col in _polars_stats:
            # The finite count and range came from one engine-side aggregation; only the leak-corr rows leave the frame.
            n_finite, ptp_or_none = _polars_stats[col]
            arr = None
        else:
            arr = _extract_column_array(df, col, rows=train_idx)
            finite_mask = np.isfinite(arr)
            n_finite = int(finite_mask.sum())
            ptp_or_none = float(np.ptp(arr[finite_mask])) if n_finite else None
        if n_finite < 50:
            drops.append({
                "name": col, "reason": "insufficient_finite_rows",
                "n_finite": int(n_finite),
            })
            continue
        ptp = float(ptp_or_none)
        if ptp <= self.config.constant_base_eps:
            drops.append({
                "name": col, "reason": "constant_or_near_constant",
                "ptp": ptp,
            })
            continue
        candidates.append(col)
        # Keep only the rows the leak-corr test will read: the constancy and finite-count checks above are done with,
        # so the full column can be released here instead of being held until the stack.
        if arr is None:
            _rows = np.asarray(train_idx) if _leak_rows is None else np.asarray(train_idx)[_leak_rows]
            candidate_arrays.append(_extract_column_array(df, col, rows=_rows))
        else:
            candidate_arrays.append(arr if _leak_rows is None else arr[_leak_rows])

    kept = _leak_corr_survivors(self, candidates, candidate_arrays, _y_leak, drops, corr_drops)
    self._filter_drops = drops
    # The corr filter is the one an operator may legitimately want to overrule for a named base, so the names it took
    # are kept: an explicit ``base_candidates=[...]`` entry is let back in through them (the numeric, finite-row and
    # constancy checks above are not overrulable - a base that fails those cannot be fitted at all).
    self._corr_filtered_bases_ = {name: corr for name, corr in corr_drops}
    # Loud warning for corr-threshold drops: this is the filter most likely to
    # misfire on legitimate strong predictors (autoregressive lags,
    # near-deterministic features). Make it visible at INFO so users can spot a false positive.
    if corr_drops:
        corr_drops.sort(key=lambda t: -t[1])
        preview = ", ".join(f"{n}=|corr|{c:.6f}" for n, c in corr_drops[:5])
        logger.info(
            "[CompositeTargetDiscovery] corr-threshold filter dropped "
            "%d feature(s) (threshold=%.6f): %s%s. If a legitimate "
            "lag/strong predictor was dropped, raise "
            "forbidden_base_corr_threshold or pass it via "
            "base_candidates=[...] explicitly.",
            len(corr_drops),
            self.config.forbidden_base_corr_threshold,
            preview,
            "" if len(corr_drops) <= 5 else f" (+{len(corr_drops) - 5} more)",
        )
    return cast(List[str], kept)
