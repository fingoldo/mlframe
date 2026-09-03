"""Group-blocked mutual information: the estimator of ``I(X;Y|G) = Σ_g p(g)·I_g(X;Y)`` over pre-binned integer codes.

MRMR ranks features by GLOBAL ``MI(X;Y)``. On group-aware panel data (e.g. wellbore logs split whole by well) a
feature that is predictive only through BETWEEN-group level differences gets a high global MI yet ~zero WITHIN-group
association, so it does not generalise to unseen groups (leakage). This module computes MI WITHIN each group and
aggregates, so a between-group-level feature is correctly demoted while a genuine within-group signal is retained -
including the case a global-then-demean approach cannot handle (a within-group relationship whose sign FLIPS across
groups still has high per-group MI in every group, so it survives here).

It tabulates the joint counts per group from the SAME global bin codes MRMR already produced (no per-group rebinning,
so bin edges stay comparable across groups), applies the per-group Miller-Madow debias ``(k_x-1)(k_y-1)/(2 n_g)`` -
essential because per-group ``n_g`` is small and the plug-in MI bias ``∝ 1/n_g`` would otherwise inflate high-
cardinality columns - and skips groups below ``min_rows``. Negative codes (the ``-1`` NaN/missing sentinel MRMR uses)
are dropped per group, mirroring the finite-masking of the base kernels.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from numba import njit


def prepare_group_segments(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sort_idx, group_offsets)`` for a stable segment-sort of rows by group.

    ``sort_idx`` orders rows so each group is a contiguous block (stable → within-group row order preserved);
    ``group_offsets[g]:group_offsets[g+1]`` are that block's positions in ``sort_idx``. Computed ONCE per fit
    (groups are fixed) and reused across every candidate column.
    """
    g = np.asarray(groups)
    _uniq, inv = np.unique(g, return_inverse=True)
    inv = inv.astype(np.int64, copy=False)
    sort_idx = np.argsort(inv, kind="stable").astype(np.int64)
    counts = np.bincount(inv, minlength=_uniq.size).astype(np.int64)
    offsets = np.zeros(_uniq.size + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return sort_idx, offsets


@njit(cache=True)
def _group_blocked_mi(
    codes_x: np.ndarray,
    codes_y: np.ndarray,
    sort_idx: np.ndarray,
    group_offsets: np.ndarray,
    K_x: int,
    K_y: int,
    min_rows: int,
    size_weighted: bool,
    use_mm: bool,
) -> float:
    """Compute per-group (block) MI(x, y) using the ``sort_idx``/``group_offsets`` partition from ``prepare_group_segments``, then combine into one group-blocked MI estimate (size-weighted average when ``size_weighted``); groups smaller than ``min_rows`` are skipped as too small for a reliable joint histogram."""
    n_groups = len(group_offsets) - 1
    joint = np.zeros((K_x, K_y), dtype=np.int64)
    mx = np.zeros(K_x, dtype=np.int64)
    my = np.zeros(K_y, dtype=np.int64)
    total_mi = 0.0
    total_w = 0.0
    for g in range(n_groups):
        lo = group_offsets[g]
        hi = group_offsets[g + 1]
        if hi - lo < min_rows:
            continue
        joint[:, :] = 0
        mx[:] = 0
        my[:] = 0
        nv = 0
        for k in range(lo, hi):
            row = sort_idx[k]
            cx = codes_x[row]
            cy = codes_y[row]
            if cx < 0 or cy < 0:  # -1 NaN/missing sentinel
                continue
            joint[cx, cy] += 1
            mx[cx] += 1
            my[cy] += 1
            nv += 1
        if nv < min_rows:
            continue
        inv = 1.0 / nv
        mi = 0.0
        for i in range(K_x):
            if mx[i] == 0:
                continue
            px = mx[i] * inv
            for j in range(K_y):
                jc = joint[i, j]
                if jc != 0:
                    py = my[j] * inv
                    jf = jc * inv
                    mi += jf * math.log(jf / (px * py))
        if use_mm:
            kx = 0
            for i in range(K_x):
                if mx[i] > 0:
                    kx += 1
            ky = 0
            for j in range(K_y):
                if my[j] > 0:
                    ky += 1
            if kx > 1 and ky > 1:
                mi -= (kx - 1) * (ky - 1) / (2.0 * nv)
            if mi < 0.0:
                mi = 0.0
        w = float(nv) if size_weighted else 1.0
        total_mi += w * mi
        total_w += w
    if total_w <= 0.0:
        # No group cleared min_rows: "cannot decide", NOT a genuine zero within-group signal. Return nan
        # (the same "cannot decide" sentinel as row-misalignment) so leak-exempt callers do not confuse a
        # small-groups panel with a real between-group-only leak and wrongly exempt/demote a good feature.
        return math.nan
    return total_mi / total_w


def _group_mi_bin_cap(group_offsets: np.ndarray, min_rows: int) -> int:
    """Bin-count cap for group-blocked MI codes, derived from the SMALLEST group that will actually be counted
    (the smallest ``n_g >= min_rows`` among ``group_offsets`` segment sizes - falling back to ``min_rows`` itself
    when no group clears the floor, since the kernel returns ``nan`` in that case anyway and the exact cap is
    moot) so that ``(cap-1)**2 <= n_g_min``: the Miller-Madow debias term is ``(kx-1)(ky-1)/(2*n_g)``, which this
    keeps bounded by roughly half a nat even when BOTH axes saturate the cap simultaneously on the SMALLEST
    counted group - comfortably below the ``log(cap)`` a real signal needs to clear. Deriving the cap from
    ``min_rows`` alone (rather than the actual group sizes) over-coarsens whenever groups are much larger than
    the floor - e.g. a single 4000-row group with ``min_rows=10`` would wrongly cap an 8-bin column down to 4
    bins, breaking the "1-group blocked MI equals global MI" identity. Never below 2 (binary split is still
    informative).
    """
    sizes = np.diff(np.asarray(group_offsets))
    qualifying = sizes[sizes >= int(min_rows)]
    n_ref = int(qualifying.min()) if qualifying.size else int(min_rows)
    return max(2, int(1 + math.sqrt(max(n_ref, 1))))


def _coarsen_codes(codes: np.ndarray, n_bins: int, cap: int) -> tuple[np.ndarray, int]:
    """Merge ``codes`` (values in ``[0, n_bins)`` or the ``-1`` sentinel) down to at most ``cap`` ordinal buckets
    via ``code * cap // n_bins`` - a monotonic many-to-one remap, so any monotonic (or sign-flipping-but-per-
    group-monotonic) X-Y relationship the fine codes captured survives coarsening. No-op (identity, same array)
    when ``n_bins <= cap`` - the common case for MRMR's default global quantization, so byte-identical there.
    """
    if n_bins <= cap:
        return codes, n_bins
    coarse = np.where(codes < 0, -1, (codes.astype(np.int64) * cap) // n_bins)
    return coarse.astype(codes.dtype, copy=False), cap


def group_blocked_mi(
    codes_x: np.ndarray,
    codes_y: np.ndarray,
    sort_idx: np.ndarray,
    group_offsets: np.ndarray,
    n_bins_x: int,
    n_bins_y: int,
    *,
    min_rows: int = 20,
    size_weighted: bool = True,
    use_mm: bool = True,
) -> float:
    """Aggregate within-group MI over pre-binned integer ``codes_x`` / ``codes_y`` (values in ``[0, n_bins)`` or ``-1``).

    ``size_weighted=True`` → ``Σ n_g·MI_g / Σ n_g`` (the plug-in ``I(X;Y|G)``); ``False`` → equal-weight mean over
    groups that clear ``min_rows`` (so one huge group does not dominate). ``use_mm`` applies the per-group Miller-Madow
    debias. Returns ``nan`` ("cannot decide") both when no group clears ``min_rows`` AND when the segments do not
    row-align to the codes (``group_offsets[-1] != len(codes_x)`` - e.g. the greedy screen subsampled rows but groups
    did not), so a caller cannot confuse either case with a genuine 0.0 within-group signal (a real between-group-only
    leak); it falls back to the global-MI path / skips the leak-exempt demotion instead.

    ``codes_x``/``codes_y`` are the caller's GLOBAL bin codes (MRMR's own adaptive/supervised discretization,
    tuned to maximise cross-group information - e.g. MDLP can pick 50+ bins for a column whose relationship to
    y only exists WITHIN each group and looks like noise globally). Fed straight into the per-group joint
    histogram, that fine a code space makes ``kx`` (distinct codes actually present in one ``n_g``-row group)
    approach ``n_g`` itself, so the Miller-Madow debias term ``(kx-1)(ky-1)/(2*n_g)`` swamps the raw plug-in MI
    and every group's contribution gets floored to 0 - a real within-group signal reads as exactly zero even
    though a handful of coarser bins would resolve it fine (verified: a sign-flipping-within-group feature
    computed group MI 0.0 at the global adaptive 69-bin code, 1.17 nats after this coarsening, both against the
    SAME per-group data). Coarsen BOTH axes to ``_group_mi_bin_cap(group_offsets, min_rows)`` bins first - a monotonic remap of
    the existing global codes, not a per-group rebin, so bin edges stay comparable across groups exactly as the
    module docstring requires; a no-op when the caller's codes are already coarse enough.
    """
    cx = np.ascontiguousarray(codes_x)
    cy = np.ascontiguousarray(codes_y)
    if int(group_offsets[-1]) != cx.shape[0] or cy.shape[0] != cx.shape[0]:
        return float("nan")
    _cap = _group_mi_bin_cap(group_offsets, min_rows)
    cx, n_bins_x = _coarsen_codes(cx, int(n_bins_x), _cap)
    cy, n_bins_y = _coarsen_codes(cy, int(n_bins_y), _cap)
    return float(_group_blocked_mi(
        cx, cy, np.ascontiguousarray(sort_idx), np.ascontiguousarray(group_offsets),
        int(n_bins_x), int(n_bins_y), int(min_rows), bool(size_weighted), bool(use_mm),
    ))


def group_relevance_mi(
    factors_data: np.ndarray,
    X: Any,
    classes_y: np.ndarray,
    factors_nbins: np.ndarray,
    n_bins_y: int,
    sort_idx: np.ndarray,
    group_offsets: np.ndarray,
    *,
    min_rows: int = 20,
    size_weighted: bool = True,
    dtype: type = np.int32,
) -> float:
    """Group-aware relevance ``I(X;Y|G)`` for a candidate index/tuple ``X``: merge X's columns to dense codes (the same
    ``merge_vars`` MRMR's own MI path uses) then group-block against the target codes ``classes_y``. Miller-Madow debias
    is ON (per-group ``n_g`` is small). Mirrors the merge in ``evaluation._densely_encode`` so codes are consistent.
    """
    from ._class_encoding import merge_vars
    classes_x, _freqs_x, nclasses_x = merge_vars(
        factors_data=factors_data, vars_indices=X, var_is_nominal=None,
        factors_nbins=factors_nbins, dtype=dtype,
    )
    return group_blocked_mi(
        np.asarray(classes_x), np.asarray(classes_y), sort_idx, group_offsets,
        int(nclasses_x), int(n_bins_y), min_rows=min_rows, size_weighted=size_weighted, use_mm=True,
    )
