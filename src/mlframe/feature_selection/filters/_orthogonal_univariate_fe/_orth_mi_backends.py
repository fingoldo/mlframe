"""Batch MI(X_j; y) backends for the orthogonal-basis FE selectors.

Two interchangeable implementations -- the sklearn ``mutual_info_score``
reference loop and the numba/cupy batch dispatcher routed through
``hermite_fe.plugin_mi_classif_batch_dispatch`` -- plus the import-time
backend chooser (`_select_mi_backend` / `_MI_BACKEND`) and the public
``_mi_classif_batch`` entry point the orth-FE family and many sibling
modules import.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _mi_classif_batch_sklearn(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Per-column quantile-bin + sklearn ``mutual_info_score`` reference path.

    Kept as the fallback when numba is unavailable AND when the caller
    explicitly opts out via ``MLFRAME_NUMBA_MI=0``. Returns MI in nats.
    """
    from sklearn.metrics import mutual_info_score
    n, p = X.shape
    mis = np.zeros(p, dtype=np.float64)
    for j in range(p):
        col = X[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            mis[j] = 0.0
            continue
        col_f = col[finite]
        try:
            edges = np.quantile(col_f, np.linspace(0.0, 1.0, nbins + 1)[1:-1])
            edges = np.unique(edges)
            if edges.size == 0:
                mis[j] = 0.0
                continue
            binned = np.searchsorted(edges, col_f)
            mis[j] = float(mutual_info_score(binned, y[finite]))
        except Exception:
            mis[j] = 0.0
    return mis


def _mi_classif_batch_numba(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Numba prange batch MI(X_j; y) for classification.

    Defers to ``plugin_mi_classif_batch_dispatch`` from ``hermite_fe``, which
    routes (n, k) to the njit prange kernel (CPU) or cupy batch kernel (GPU)
    via ``pyutilz.performance.kernel_tuning.cache`` and uses argsort-based
    equi-frequency binning. Bench at p=200 n=2000: ~6ms vs ~317ms for the
    per-column sklearn loop (~53x speedup).

    Numerical equivalence vs the sklearn reference (``_mi_classif_batch_sklearn``)
    holds to within machine epsilon — verified across 40 seeds (Gaussian and
    integer-with-noise) max abs diff < 2e-15. The argsort equi-frequency
    binning and the ``np.quantile``+``searchsorted`` binning produce different
    bin assignments only when source values have ties, but the resulting MI
    on a discrete y is numerically identical because both partitions yield
    the same effective contingency table marginals once the histogram math
    sums the per-bin entropy contributions.

    Handles partial-NaN columns by masking to the finite subset per column,
    matching ``_mi_classif_batch_sklearn`` semantics. An all-NaN column or a
    column where every value collapses to a single bin returns 0.0.
    """
    from ..hermite_fe import plugin_mi_classif_batch_dispatch

    n, p = X.shape
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    mis = np.zeros(p, dtype=np.float64)
    # Partition columns into "all-finite" (bulk path) and "partial-NaN"
    # (per-column fallback). In the hybrid_orth_mi_fe production path the
    # source frames are nan-filled upstream so partial_idx is empty and
    # everything goes through the single batch dispatch call.
    finite_per_col = np.isfinite(X).all(axis=0)
    dense_cols = np.where(finite_per_col)[0]
    partial_cols = np.where(~finite_per_col)[0]

    if dense_cols.size:
        # When EVERY column is finite (the production nan-filled path), the
        # ``X[:, dense_cols]`` fancy-index is a full (n, p) gather COPY that
        # reproduces X verbatim -- skip it and hand the (already-contiguous)
        # frame straight to the batch kernel. On a 40k x 200 all-finite frame
        # this setup dropped 3109ms -> 212ms (~14.6x) across 23 calls; the
        # gather copy was the entire self-time. Partial-NaN columns still take
        # the real gather below.
        if dense_cols.size == p:
            X_dense = np.ascontiguousarray(X)
        else:
            X_dense = np.ascontiguousarray(X[:, dense_cols])
        try:
            mis_dense = plugin_mi_classif_batch_dispatch(X_dense, y_i64, nbins)
            mis[dense_cols] = mis_dense
        except Exception:
            # If the batch path fails for any reason (cupy import error,
            # kernel tuning miss, etc.), fall back to sklearn for the
            # affected slice rather than poisoning the whole call.
            mis[dense_cols] = _mi_classif_batch_sklearn(
                X_dense, y_i64, nbins=nbins,
            )

    if partial_cols.size:
        # Partial-NaN columns get the per-column path (mask + sklearn). The
        # production hybrid path nan-fills before calling MI so this branch
        # is essentially dead code in practice; keep it for API parity.
        for j in partial_cols:
            col = X[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                mis[j] = 0.0
                continue
            col_f = np.ascontiguousarray(col[finite].reshape(-1, 1))
            y_f = np.ascontiguousarray(y_i64[finite])
            try:
                mis[j] = float(
                    plugin_mi_classif_batch_dispatch(col_f, y_f, nbins)[0],
                )
            except Exception:
                mis[j] = 0.0
    return mis


# Module-import-time decision: which backend does ``_mi_classif_batch`` use?
# - ``MLFRAME_NUMBA_MI=0``  -> force sklearn loop reference
# - ``MLFRAME_NUMBA_MI=1``  -> force numba batch (raises at first call if numba missing)
# - unset / any other value -> auto: numba batch when ``hermite_fe.plugin_mi_classif_batch_dispatch``
#   imports cleanly (the standard case in this repo), sklearn otherwise.
# Cached because hybrid_orth_mi_fe calls _mi_classif_batch twice per fit and
# the dispatcher decision is constant per process.
def _select_mi_backend() -> str:
    import os as _os
    flag = _os.environ.get("MLFRAME_NUMBA_MI", "").strip().lower()
    if flag in ("0", "false", "off", "no"):
        return "sklearn"
    if flag in ("1", "true", "on", "yes"):
        return "numba"
    # auto: try-import the numba dispatcher; on failure (e.g. numba absent
    # in a stripped-down install) fall back to sklearn rather than crashing
    # at first call.
    try:
        from ..hermite_fe import plugin_mi_classif_batch_dispatch  # noqa: F401
        return "numba"
    except Exception:
        return "sklearn"


_MI_BACKEND = _select_mi_backend()


def _mi_classif_batch(X: np.ndarray, y: np.ndarray, *, nbins: int = 10) -> np.ndarray:
    """Batch MI(X_j; y) for classification target.

    Layer 31 (2026-05-31): routes to the numba prange batch dispatcher
    (``_mi_classif_batch_numba``) when available — ~53x speedup at
    p=200 n=2000 over the per-column sklearn loop, bit-equivalent to within
    machine epsilon (< 2e-15 across 40 seeds). Set ``MLFRAME_NUMBA_MI=0``
    to force the sklearn reference if a downstream regression demands it.

    Idea #18 (2026-06-10) -- bench-rejected, default OFF: an inverse-prior
    class-balanced MI was added to test whether plain plug-in MI under-RANKS
    rare-class-discriminative features under imbalance. It does NOT: balancing is
    a near-uniform multiplicative rescale (Kendall tau 0.989 vs plain), so it
    almost never changes the rank-based selection, and where it does (13/120
    imbalanced frames) the downstream rare-class AP is a net-negative coin-flip
    (mean dAP -0.0037). Kept opt-in via ``MLFRAME_FE_IMBALANCE_MI=on`` (default
    ``off`` => this branch is skipped and the path is byte-for-byte the plain MI
    below). Full numbers in ``_imbalance_mi`` module docstring + the regression
    test ``tests/feature_selection/test_imbalance_mi.py``.
    """
    # Fast OFF short-circuit (the default): a single env read, no import / no
    # bincount, so the common path is byte-for-byte and ~free vs plain numba.
    import os as _os
    if _os.environ.get("MLFRAME_FE_IMBALANCE_MI", "").strip().lower() in ("on", "1", "true", "yes", "auto"):
        class_w = _maybe_class_weights(y)  # opt-in only
        if class_w is not None:
            cb = _mi_classif_batch_balanced(X, y, class_w, nbins=nbins)
            if cb is not None:
                return cb
    if _MI_BACKEND == "numba":
        return _mi_classif_batch_numba(X, y, nbins=nbins)
    return _mi_classif_batch_sklearn(X, y, nbins=nbins)


def _mi_chunk_cols_for(n_rows: int) -> int:
    """RAM-aware column-block width: bound the per-block float64 materialization (n_rows * cols * 8 B, plus a
    ~3x factor for V, V^2 and the binning transient) to ~10% of free RAM, capped at 1024 cols, floored at 1.
    A fixed COLUMN count alone is unsafe at large n (1024 cols x n=1M float64 = 8 GiB still OOMs); bounding the
    block BYTES makes the chunked scorer safe in n as well as p. Conservative 2 GiB fallback if psutil missing."""
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
    except Exception:
        free = 2 * 1024 ** 3
    budget = max(1, int(free * 0.10))
    by_ram = budget // (max(1, int(n_rows)) * 8 * 3)
    return int(min(1024, max(1, by_ram)))


def mi_classif_batch_chunked(X, y, *, nbins: int = 10, chunk_cols: int = None) -> np.ndarray:
    """Column-CHUNKED ``_mi_classif_batch`` for WIDE engineered matrices (2026-06-19).

    The FE MI-uplift scorers (univariate / pair-cross / triplet / quadruplet / adaptive-arity / mi-greedy)
    each materialised the FULL engineered matrix as one float64 array to batch-score MI -- O(n * n_engineered)
    peak RAM that OOMs at scale (measured (16000, 20000) float64 = 2.38 GiB), worst for the combinatorial
    triplet/quadruplet cross-basis families. MI is PER-COLUMN, so scoring in column blocks is BIT-IDENTICAL to
    the all-at-once call FOR ANY chunk size while bounding peak extra RAM. ``chunk_cols`` defaults to a RAM-aware
    width (see ``_mi_chunk_cols_for`` -- bounds the BLOCK BYTES, so it is safe at large n too, not just wide p);
    pass an explicit value to override. Accepts a pandas DataFrame (sliced via ``iloc``, so only the block is
    materialised) or a 2-D ndarray. Returns the (p,) per-column MI array."""
    # The block loop below stays SEQUENTIAL by design: ``_mi_classif_batch`` already parallelises across the
    # block's columns via the numba prange batch kernel, so each per-block call saturates all cores. Threading
    # the block loop on top would OVERSUBSCRIBE the prange pool (no speedup, likely slower) -- bench-rejected
    # (2026-06-19); the chunking is a MEMORY bound, not a missing parallelism lever.
    is_df = hasattr(X, "iloc")
    n = int(X.shape[0])
    p = int(X.shape[1])
    if chunk_cols is None:
        chunk_cols = _mi_chunk_cols_for(n)
    if p <= chunk_cols:
        arr = X.to_numpy(dtype=np.float64) if is_df else np.asarray(X, dtype=np.float64)
        return _mi_classif_batch(arr, y, nbins=nbins)
    parts = []
    for j0 in range(0, p, chunk_cols):
        if is_df:
            blk = X.iloc[:, j0:j0 + chunk_cols].to_numpy(dtype=np.float64)
        else:
            blk = np.asarray(X[:, j0:j0 + chunk_cols], dtype=np.float64)
        parts.append(_mi_classif_batch(blk, y, nbins=nbins))
        del blk
    return np.concatenate(parts)


def _maybe_class_weights(y: np.ndarray):
    """Auto-detect imbalance and return inverse-prior class weights, or ``None``.

    ``None`` => the caller falls through to the plain-MI path unchanged
    (balanced data, below the n_rare gate, non-discrete y, or override ``off``).
    Cheap: a single ``bincount`` + two comparisons; adds ~0 to the balanced path.
    """
    try:
        from ._imbalance_mi import compute_class_weights
        return compute_class_weights(y)
    except Exception:
        return None


def _mi_classif_batch_balanced(X: np.ndarray, y: np.ndarray, class_w, *, nbins: int = 10):
    """Class-balanced batch MI; mirrors ``_mi_classif_batch_numba``'s NaN handling.

    Returns ``None`` on any failure so the caller falls back to plain MI rather
    than poisoning the whole call.
    """
    try:
        from ._imbalance_mi import _class_balanced_mi_batch_njit
    except Exception:
        return None
    n, p = X.shape
    y_i64 = np.ascontiguousarray(y, dtype=np.int64)
    class_w = np.ascontiguousarray(class_w, dtype=np.float64)
    mis = np.zeros(p, dtype=np.float64)
    finite_per_col = np.isfinite(X).all(axis=0)
    dense_cols = np.where(finite_per_col)[0]
    partial_cols = np.where(~finite_per_col)[0]
    try:
        if dense_cols.size:
            if dense_cols.size == p:
                X_dense = np.ascontiguousarray(X)
            else:
                X_dense = np.ascontiguousarray(X[:, dense_cols])
            mis[dense_cols] = _class_balanced_mi_batch_njit(X_dense, y_i64, class_w, nbins)
        for j in partial_cols:
            col = X[:, j]
            finite = np.isfinite(col)
            if not finite.any():
                mis[j] = 0.0
                continue
            col_f = np.ascontiguousarray(col[finite].reshape(-1, 1))
            y_f = np.ascontiguousarray(y_i64[finite])
            mis[j] = float(_class_balanced_mi_batch_njit(col_f, y_f, class_w, nbins)[0])
    except Exception:
        return None
    return mis
