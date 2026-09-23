"""Correlation helpers for ensemble member diagnostics.

Carved out of ``base.py`` (1k-LOC house limit); ``base`` re-exports every name here, so existing importers of
``mlframe.models.ensembling.base`` and of the package are unaffected.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HAS_CUPY_CACHED: "bool | None" = None


def _has_cupy() -> bool:
    """Whether cupy is importable, probed on FIRST USE rather than at import.

    Probing at import made ``import mlframe.models`` - and everything that reaches it, which on this tree includes
    ``import mlframe.feature_selection`` - pay cupy's ~1.7s import and a CUDA context, for a dispatcher branch that
    only fires on wide or very long stacks.
    """
    global _HAS_CUPY_CACHED
    if _HAS_CUPY_CACHED is None:
        from mlframe.system import try_import_cupy

        _HAS_CUPY_CACHED = bool(try_import_cupy()[1])
    return _HAS_CUPY_CACHED


def _stacked_corrcoef(M: np.ndarray) -> np.ndarray:
    """Correlation matrix of (K, N) stacked vectors with a size-dispatcher.

    Replaces the previous O(K^2) Python pair loop. Routes to cupy when both available and (K>50 OR
    N>1M); falls back to plain numpy otherwise. Both paths emit a (K, K) ndarray of Pearson
    correlations; constant rows surface as NaN entries (caller is expected to filter).
    """
    K = M.shape[0]
    N = M.shape[1] if M.ndim > 1 else 1
    use_cupy = (K > 50 or N > 1_000_000) and _has_cupy()  # order matters: the cheap size test gates the probe
    if use_cupy:
        try:
            import cupy as _cp

            M_gpu = _cp.asarray(M)
            corr_gpu = _cp.corrcoef(M_gpu)
            return np.asarray(_cp.asnumpy(corr_gpu))
        except Exception as e:  # pragma: no cover -- defensive
            logger.debug("swallowed exception in base.py: %s", e)
            pass
    return np.asarray(np.corrcoef(M))


def compute_high_correlation_pairs(
    members: Sequence,
    member_tags: Sequence[str],
    threshold: float = 0.98,
) -> tuple[list[dict], Optional[str]]:
    """Return pairs of ensemble members whose predictions are correlated above ``threshold`` plus the split that fed the check.

    Diversity is checked once on whichever prediction array is universally available across members, in this precedence:
    ``val_preds -> test_preds -> train_preds -> val_probs -> test_probs -> train_probs``. Probabilistic outputs collapse
    to a single column (last) for the correlation proxy; full multinomial diversity is overkill for near-duplicate detection.
    Members with a constant vector on the chosen split (std == 0) or fewer than 2 finite shared samples are skipped, not flagged.

    No mutation here - the caller decides what to do (WARN, persist, drop). Today the only caller (``score_ensemble``) just warns.
    """
    pairs: list[dict] = []
    if len(members) < 2:
        return pairs, None
    arrays: list[np.ndarray] = []
    split_used: Optional[str] = None
    for attr in ("val_preds", "test_preds", "train_preds"):
        cand = [getattr(m, attr, None) for m in members]
        if all(p is not None for p in cand):
            arrays = [np.asarray(p, dtype=np.float64).ravel() for p in cand]
            split_used = attr
            break
    # When no preds attribute is available, fall back to probs. For multiclass (C>=3) we compute
    # per-class correlation matrices and AVERAGE the off-diagonal pair entries across classes --
    # the pre-fix flatten-then-Pearson interleaved per-row class entries into a single long vector
    # and computed Pearson over that, which mixes intra-row class structure with inter-row variation
    # and is not a meaningful diversity measure. Binary (C==2) collapses to a 1-column proxy because
    # the two columns are perfectly linearly dependent (sum to 1); the per-class average over the
    # two redundant columns equals the single-column Pearson by construction.
    probs_arrays: list[np.ndarray] = []
    if not arrays:
        for attr in ("val_probs", "test_probs", "train_probs"):
            cand = [getattr(m, attr, None) for m in members]
            if all(p is not None for p in cand):
                probs_arrays = [np.asarray(p, dtype=np.float64) for p in cand]
                split_used = attr
                break
    if arrays:
        if not all(a.size == arrays[0].size and a.size >= 2 for a in arrays):
            return pairs, split_used
        M_stack = np.vstack(arrays)  # (K, N)
        corr_matrix = _pairwise_corr_or_nan(M_stack)
        if corr_matrix is None:
            return pairs, split_used
        K_use = corr_matrix.shape[0]
        idx_use = np.arange(K_use)
        return _emit_pairs_above_threshold(corr_matrix, idx_use, member_tags, threshold, split_used)
    if probs_arrays:
        # All probs arrays must share (N, C) (or (N,) which we promote to (N, 1)).
        norm = []
        for a in probs_arrays:
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            norm.append(a)
        probs_arrays = norm
        if not all(a.shape == probs_arrays[0].shape and a.shape[0] >= 2 for a in probs_arrays):
            return pairs, split_used
        K_members = len(probs_arrays)
        n_classes = probs_arrays[0].shape[1]
        if n_classes == 1:
            # Single-column / binary-as-1D path: stack as (K, N) and reuse the standard corr.
            M_stack = np.vstack([a.ravel() for a in probs_arrays])
            corr_matrix = _pairwise_corr_or_nan(M_stack)
            if corr_matrix is None:
                return pairs, split_used
            return _emit_pairs_above_threshold(corr_matrix, np.arange(corr_matrix.shape[0]), member_tags, threshold, split_used)
        # Multiclass per-class correlation, then average the off-diagonal entries across classes.
        per_class_corrs: list[np.ndarray] = []
        for _ci in range(n_classes):
            M_stack_ci = np.vstack([a[:, _ci] for a in probs_arrays])
            corr_ci = _pairwise_corr_or_nan(M_stack_ci, return_full_shape=True, original_k=K_members)
            if corr_ci is not None:
                per_class_corrs.append(corr_ci)
        if not per_class_corrs:
            return pairs, split_used
        stack_corrs = np.stack(per_class_corrs, axis=0)
        with np.errstate(invalid="ignore"):
            avg_corr = np.nanmean(stack_corrs, axis=0)
        return _emit_pairs_above_threshold(avg_corr, np.arange(K_members), member_tags, threshold, split_used)
    return pairs, split_used


def _pairwise_corr_or_nan(M_stack: np.ndarray, *, return_full_shape: bool = False, original_k: Optional[int] = None) -> Optional[np.ndarray]:
    """Compute the (K, K) Pearson corr matrix of a stacked (K, N) array.

    Masks NaN-bearing columns and constant-row members (std==0). Returns ``None`` when fewer than
    2 finite columns OR fewer than 2 non-constant members remain. When ``return_full_shape=True``
    AND ``original_k`` is supplied, returns a (original_k, original_k) matrix with NaN-padded rows
    for skipped members; callers averaging across classes can then ``np.nanmean`` over per-class
    matrices of the same shape.
    """
    K = M_stack.shape[0]
    finite_cols = np.all(np.isfinite(M_stack), axis=0)
    if int(finite_cols.sum()) < 2:
        if return_full_shape and original_k is not None:
            return np.full((original_k, original_k), np.nan, dtype=np.float64)
        return None
    M_finite = M_stack[:, finite_cols]
    stds = M_finite.std(axis=1)
    nonconst = stds > 0
    if int(nonconst.sum()) < 2:
        if return_full_shape and original_k is not None:
            return np.full((original_k, original_k), np.nan, dtype=np.float64)
        return None
    M_use = M_finite[nonconst]
    idx_use = np.flatnonzero(nonconst)
    corr_used = _stacked_corrcoef(M_use)
    if return_full_shape and original_k is not None:
        out = np.full((original_k, original_k), np.nan, dtype=np.float64)
        # Vectorised NaN-padded scatter: ``out[idx_use, idx_use]`` block-assign via ``np.ix_``
        # replaces the O(K_use^2) Python double loop (bit-identical; 5-17x at K=10-20).
        out[np.ix_(idx_use, idx_use)] = corr_used
        return out
    # When return_full_shape=False the caller expects an indexed-by-use matrix and will iterate
    # via idx_use externally; we return the dense submatrix and let the caller pass idx_use to
    # ``_emit_pairs_above_threshold``.
    # We need to expose idx_use to the caller; pack the corr_used into a full (K, K) NaN-padded
    # matrix so the iteration sites stay symmetric.
    out = np.full((K, K), np.nan, dtype=np.float64)
    # Same vectorised np.ix_ block-scatter as the return_full_shape branch above (bit-identical).
    out[np.ix_(idx_use, idx_use)] = corr_used
    return out


def _emit_pairs_above_threshold(
    corr_matrix: np.ndarray,
    idx_use: np.ndarray,
    member_tags: Sequence[str],
    threshold: float,
    split_used: Optional[str],
) -> tuple[list[dict], Optional[str]]:
    """Scan the upper triangle of ``corr_matrix`` (indexed via ``idx_use``, the non-skipped members) and collect member-tag pairs whose absolute correlation exceeds ``threshold`` into report dicts."""
    pairs: list[dict] = []
    K = corr_matrix.shape[0]
    for ii in range(K):
        for jj in range(ii + 1, K):
            corr = float(corr_matrix[ii, jj])
            if not np.isfinite(corr):
                continue
            if abs(corr) > threshold:
                m1 = member_tags[ii] if ii < len(member_tags) else f"member_{ii}"
                m2 = member_tags[jj] if jj < len(member_tags) else f"member_{jj}"
                pairs.append({"m1": m1, "m2": m2, "corr": corr})
    return pairs, split_used
