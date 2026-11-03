"""Fused honest-null floor for ``nbins_strategy="optimal_joint"`` / ``"cv"``: every permutation inside one numba kernel instead of a Python re-binning per permutation.

Per permutation the floor (``_honest_mdlp_null.honest_fdr_floor`` -> ``_honest_perm_best``) re-bins the pool on the permuted target through
``PermutedTargetBinner.codes``: the winning candidate bin count of every column from its cached plan (``prescore_best_m``: mean fold MI of each candidate
against the binned validation target), the column's codes under that count (a table row of the binner), then the largest Miller-Madow corrected MI of the
pool against the permuted target codes (``corrected_mi_best``). Here the code rows of every (column, candidate count) are built once through the binner's
own ``_edges_slot`` (the production edge path, with the count planted as the plan's winner), and the kernel does the rest for a block of permutations:
the fold targets ``y_raw[perm[val_rows]]`` with ``K = max + 1`` (``_bin_y_for_mi``'s pass-through for small non-negative integer labels), the winner by the
loop of ``_oj_pick_candidates`` (calling the same ``_plug_in_mi_njit``), the codes by row lookup, the statistic by the formula of ``corrected_mi_best``. Same permutations, same numbers.
Declines (``None``: the caller keeps the per-permutation path) for a continuous or wide-label raw target, plans that are not all cached or packable, a column
without a scored candidate in any permutation, or ``MLFRAME_HONEST_OJ_FUSED=0``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from ._adaptive_nbins import _plug_in_mi_njit
from ._numba_utils_cross_module import _mi_codes_njit, _n_unique_codes

logger = logging.getLogger(__name__)

_PERM_BLOCK = 64  # permutations drawn on the host and scored per kernel launch


@njit(cache=True, parallel=True)
def _fused_block(perms, y_raw, yc, rows, xb, kx, nv, val_idx, pool_dep, pool_codes, pool_nb, n_eff, out):
    """``out[b]`` = the pool's largest corrected MI under permutation ``perms[b]``, ``-inf`` when a column had no scored candidate (the caller declines)."""
    n_f, p, n_m = kx.shape
    n_pool = pool_dep.shape[0]
    max_nv = val_idx.shape[1]
    for b in prange(perms.shape[0]):
        perm = perms[b]
        # fold targets of the permuted raw target
        vy = np.zeros((n_f, max_nv), dtype=np.int64)
        ky = np.zeros(n_f, dtype=np.int64)
        for f in range(n_f):
            top = 0
            for i in range(nv[f]):
                v = y_raw[perm[val_idx[f, i]]]
                vy[f, i] = v
                if v > top:
                    top = v
            ky[f] = top + 1 if nv[f] > 0 else 0
        # winners of the re-binned columns
        pick = np.full(p, -1, dtype=np.int64)
        for j in range(p):
            best = -np.inf
            best_m = -1
            for m in range(n_m):
                total = 0.0
                cells = 0
                for f in range(n_f):
                    k_x = kx[f, j, m]
                    if k_x == 0:
                        continue
                    mi = 0.0
                    if ky[f] >= 1:
                        mi = _plug_in_mi_njit(xb[f, j, m, : nv[f]], vy[f, : nv[f]], k_x, ky[f], False)
                    total += mi
                    cells += 1
                if cells > 0 and total / cells > best:
                    best = total / cells
                    best_m = m
            pick[j] = best_m
        # permuted target codes at the evaluation rows
        ycp = np.empty(rows.shape[0], dtype=np.int64)
        ny = 0
        for i in range(rows.shape[0]):
            v = yc[perm[rows[i]]]
            ycp[i] = v
            if v + 1 > ny:
                ny = v + 1
        counts = np.zeros(ny, dtype=np.int64)
        for i in range(rows.shape[0]):
            counts[ycp[i]] += 1
        ky_eff = 0.0
        for c in range(ny):
            if counts[c] > 0:
                ky_eff += 1.0
        res = 0.0
        bad = False
        x = np.empty(rows.shape[0], dtype=np.int64)
        for t in range(n_pool):
            j = pool_dep[t]
            m = 0
            if j >= 0:
                m = pick[j]
                if m < 0:
                    bad = True
                    break
            for i in range(rows.shape[0]):
                x[i] = pool_codes[t, m, rows[i]]
            kxp = pool_nb[t, m]
            mi = _mi_codes_njit(x, kxp, ycp, ny)
            kx_eff = float(_n_unique_codes(x))
            corr = mi - (kx_eff - 1.0) * (ky_eff - 1.0) / (2.0 * n_eff)
            if corr > res:
                res = corr
        out[b] = -np.inf if bad else res


def _assemble_pool(binner, pool, width, dep_rows, ms, factors_data, factors_nbins, n):
    """``(pool_dep, pool_codes, pool_nb)``: for each pool column its re-binned column index (``-1`` for a fixed one), its code rows per candidate count in
    the binner's code dtype (a fixed column fills candidate 0), and the bin counts."""
    pool_dep = np.full(len(pool), -1, dtype=np.int64)
    with binner._lock:
        table, col_max = binner._table, binner._col_max
    code_dtype = np.promote_types(table.dtype, np.asarray(factors_data).dtype) if any(c >= width for c in pool) else table.dtype
    pool_codes = np.zeros((len(pool), len(ms), n), dtype=code_dtype)
    pool_nb = np.zeros((len(pool), len(ms)), dtype=np.int64)
    for t, c in enumerate(pool):
        if c < width and int(binner._out_dep[c]) >= 0:
            j = int(binner._out_dep[c])
            pool_dep[t] = j
            for mi in range(len(ms)):
                pool_codes[t, mi] = table[dep_rows[j, mi]]
                pool_nb[t, mi] = int(col_max[dep_rows[j, mi]]) + 1
        elif c < width:
            s = int(binner._out_fixed[c])
            pool_codes[t, 0] = table[s]
            pool_nb[t, 0] = int(col_max[s]) + 1
        else:
            pool_codes[t, 0] = np.asarray(factors_data[:, c])
            pool_nb[t, 0] = int(factors_nbins[c])
    return pool_dep, pool_codes, pool_nb


def oj_fused_floor(honest_codes_fn: Any, factors_data: np.ndarray, factors_nbins: np.ndarray, y_idx: int, cols: list, n_permutations: int, quantile: float, base_seed: int,
                   eval_rows: Optional[np.ndarray]) -> Optional[float]:
    """The honest FDR floor of :func:`honest_fdr_floor` from the fused kernel, or ``None`` when it does not apply (see the module docstring)."""
    if os.environ.get("MLFRAME_HONEST_OJ_FUSED", "1").strip() == "0":
        return None
    binner = getattr(honest_codes_fn, "binner", None)
    y_raw = getattr(honest_codes_fn, "y_raw", None)
    if binner is None or y_raw is None or str(binner._strategy).lower() not in ("optimal_joint", "cv"):
        return None
    try:
        floor: Optional[float] = _floor(honest_codes_fn, binner, np.asarray(y_raw), factors_data, factors_nbins, y_idx, cols, n_permutations, quantile, base_seed, eval_rows)
        return floor
    except Exception as exc:
        logger.debug("fused optimal_joint honest floor not used (%s: %s); per-permutation path", type(exc).__name__, exc)
        return None


def _floor(honest_codes_fn, binner, y_raw, factors_data, factors_nbins, y_idx, cols, n_permutations, quantile, base_seed, eval_rows):
    """Body of :func:`oj_fused_floor`."""
    from ._adaptive_nbins import _distinct_exceeds
    from ._honest_mdlp_null import seeded_permutation

    max_y = int(binner._kwargs.get("optimal_joint_max_y_classes", 64))
    if y_raw.dtype.kind not in "iub" or y_raw.size == 0 or int(y_raw.min()) < 0 or _distinct_exceeds(y_raw, max_y):
        return None
    y_int = np.ascontiguousarray(y_raw, dtype=np.int64)
    if binner._prescore(y_int) is None or binner._packed is None:
        # the plans are built by a permutation's normal route: run the first permutation through the hook once (its codes are the production ones), then pack
        honest_codes_fn(seeded_permutation(int(factors_data.shape[0]), base_seed))
        if binner._prescore(y_int) is None or binner._packed is None:
            return None
    packed, plans = binner._packed, binner._plans
    ms = list(packed["ms"])
    n = int(factors_data.shape[0])
    n_dep = binner._arr.shape[1]
    # the code row of every (re-binned column, candidate count), through the binner's own edge path with the count planted as the plan's winner
    cache = binner._kwargs["optimal_joint_plan_cache"]
    key = cache._target_key(y_int)
    dep_rows = np.empty((n_dep, len(ms)), dtype=np.int64)
    for j in range(n_dep):
        for mi, m in enumerate(ms):
            s = binner._by_count.get((j, m))
            if s is None:
                best = plans[j].setdefault("best", {})
                prev = best.get(key)
                best[key] = m
                try:
                    s = binner._edges_slot(j, y_int)
                finally:
                    if prev is None:
                        best.pop(key, None)
                    else:
                        best[key] = prev
                if s is None:
                    return None
                binner._by_count[(j, m)] = s
            dep_rows[j, mi] = s
    width = len(binner._out_names)
    pool = [int(c) for c in cols]
    rows = np.arange(n, dtype=np.int64) if eval_rows is None else np.asarray(eval_rows, dtype=np.int64)
    # the pool's code rows, assembled once per (pool, rows count) and kept on the binner: the floor runs twice per fit over the same pool
    memo_key = (tuple(pool), n)
    memo = getattr(binner, "_fused_pool_memo", None)
    if memo is not None and memo[0] == memo_key:
        pool_dep, pool_codes, pool_nb = memo[1]
    else:
        pool_dep, pool_codes, pool_nb = _assemble_pool(binner, pool, width, dep_rows, ms, factors_data, factors_nbins, n)
        binner._fused_pool_memo = (memo_key, (pool_dep, pool_codes, pool_nb))
    folds = packed["folds"]
    nv = np.asarray(packed["nv"], dtype=np.int64)
    val_idx = np.zeros((len(folds), int(nv.max())), dtype=np.int64)
    for f, (_k, vm) in enumerate(folds):
        idx = np.flatnonzero(vm)
        val_idx[f, : idx.size] = idx
    yc = np.ascontiguousarray(factors_data[:, int(y_idx)]).astype(np.int64)
    nulls = np.empty(int(n_permutations), dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
    for b0 in range(0, int(n_permutations), _PERM_BLOCK):
        b1 = min(int(n_permutations), b0 + _PERM_BLOCK)
        perms = np.stack([seeded_permutation(n, base_seed + b) for b in range(b0, b1)]).astype(np.int64)
        out = np.empty(b1 - b0, dtype=np.float64)  # Intentional float64: statistic vector
        _fused_block(perms, y_int, yc, rows, packed["xb"], np.asarray(packed["kx"], dtype=np.int64), nv, val_idx, pool_dep, pool_codes, pool_nb, float(len(rows)), out)
        if np.isneginf(out).any():
            return None
        nulls[b0:b1] = out
    return float(np.quantile(nulls, float(quantile), method="higher"))
