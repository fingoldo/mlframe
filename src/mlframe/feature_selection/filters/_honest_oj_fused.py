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
import math
import os
from typing import Any, Optional

import numpy as np
from numba import njit, prange

from ._adaptive_nbins import _plug_in_mi_njit
from ._numba_utils_cross_module import _n_unique_codes, _permuted_codes, _pool_max
from ._numba_utils_shared import seeded_permutation
from mlframe._numba_parallel_guard import parallel_kernel_entry
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_PERM_BLOCK = 64  # permutations drawn on the host and scored per kernel launch


@njit(cache=True, nogil=True)
def _fold_targets(perm, y_raw, nv, val_idx):
    """Fold targets ``y_raw[perm[val rows]]`` of one permutation and their class counts ``K = max + 1``."""
    n_f = nv.shape[0]
    vy = np.zeros((n_f, val_idx.shape[1]), dtype=np.int64)
    ky = np.zeros(n_f, dtype=np.int64)
    for f in range(n_f):
        top = 0
        for i in range(nv[f]):
            v = y_raw[perm[val_idx[f, i]]]
            vy[f, i] = v
            top = max(top, v)
        ky[f] = top + 1 if nv[f] > 0 else 0
    return vy, ky


@njit(cache=True, nogil=True)
def _mean_fold_mi(xb, kx, nv, vy, ky, j, m):
    """Mean fold MI of candidate ``m`` of column ``j`` (``-inf`` when no fold scored it)."""
    total = 0.0
    cells = 0
    for f in range(kx.shape[0]):
        k_x = kx[f, j, m]
        if k_x == 0:
            continue
        if ky[f] >= 1:
            total += _plug_in_mi_njit(xb[f, j, m, : nv[f]], vy[f, : nv[f]], k_x, ky[f], False)
        cells += 1
    return total / cells if cells > 0 else -np.inf


@njit(cache=True, nogil=True)
def _winners(xb, kx, nv, vy, ky):
    """Winning candidate index of every re-binned column (``-1`` when none was scored): the loop of ``_oj_pick_candidates``."""
    p, n_m = kx.shape[1], kx.shape[2]
    pick = np.full(p, -1, dtype=np.int64)
    for j in range(p):
        best = -np.inf
        for m in range(n_m):
            sc = _mean_fold_mi(xb, kx, nv, vy, ky, j, m)
            if sc > best:
                best = sc
                pick[j] = m
    return pick


@njit(cache=True, parallel=True, nogil=True)
def _px_table(xb, kx, nv):
    """``(F, P, M, max k_x)`` bin counts of every fold cell: they do not depend on the permutation (the codes are fixed, only the target moves)."""
    n_f, p, n_m = kx.shape
    out = np.zeros((n_f, p, n_m, max(1, int(kx.max()))), dtype=np.float64)  # Intentional float64: count table feeding the MI statistic
    for j in prange(p):
        for f in range(n_f):
            for m in range(n_m):
                if kx[f, j, m] > 0:
                    for i in range(nv[f]):
                        out[f, j, m, xb[f, j, m, i]] += 1.0
    return out


@njit(cache=True, nogil=True)
def _minority_rows(vy, nv, ky):
    """Per fold, the rows outside the most frequent class (``(F, max nv)`` row lists, their lengths) and that class: only those rows are scattered."""
    n_f = nv.shape[0]
    idx = np.empty((n_f, vy.shape[1]), dtype=np.int64)
    cnt = np.zeros(n_f, dtype=np.int64)
    cmax = np.zeros(n_f, dtype=np.int64)
    for f in range(n_f):
        counts = np.zeros(max(1, ky[f]), dtype=np.int64)
        for i in range(nv[f]):
            counts[vy[f, i]] += 1
        cmax[f] = np.argmax(counts)
        for i in range(nv[f]):
            if vy[f, i] != cmax[f]:
                idx[f, cnt[f]] = i
                cnt[f] += 1
    return idx, cnt, cmax


@njit(cache=True, nogil=True)
def _mi_from_minority(xrow, vy_f, rows, n_rows, px, k_x, k_y, cmax, n):
    """``_plug_in_mi_njit(xrow, vy_f, k_x, k_y, False)`` bit for bit: the joint counts of the majority class column are the bin counts ``px`` minus the
    other classes' (exact integers in double), so only the minority rows are scattered; Px, Py and the MI sum run in the same order."""
    joint = np.zeros((k_x, k_y), dtype=np.float64)  # Intentional float64: joint-count accumulator for the MI statistic
    for r in range(n_rows):
        i = rows[r]
        joint[xrow[i], vy_f[i]] += 1.0
    for a in range(k_x):
        s = 0.0
        for c in range(k_y):
            if c != cmax:
                s += joint[a, c]
        joint[a, cmax] = px[a] - s
    n_f = float(n)
    Px = np.zeros(k_x, dtype=np.float64)  # Intentional float64: marginal-count accumulator for the MI statistic
    Py = np.zeros(k_y, dtype=np.float64)  # Intentional float64: marginal-count accumulator for the MI statistic
    for a in range(k_x):
        for c in range(k_y):
            Px[a] += joint[a, c]
            Py[c] += joint[a, c]
    mi = 0.0
    for a in range(k_x):
        if Px[a] <= 0.0:
            continue
        for c in range(k_y):
            v = joint[a, c]
            if v <= 0.0 or Py[c] <= 0.0:
                continue
            pr = v / n_f
            mi += pr * math.log(pr * n_f / (Px[a] * Py[c] / n_f))
    return mi


@njit(cache=True, nogil=True)
def _winners_minority(xb, kx, nv, vy, ky, px):
    """:func:`_winners` with every fold MI from :func:`_mi_from_minority` (same picks, same numbers, fewer scattered rows)."""
    p, n_m = kx.shape[1], kx.shape[2]
    idx, cnt, cmax = _minority_rows(vy, nv, ky)
    pick = np.full(p, -1, dtype=np.int64)
    for j in range(p):
        best = -np.inf
        for m in range(n_m):
            total = 0.0
            cells = 0
            for f in range(kx.shape[0]):
                k_x = kx[f, j, m]
                if k_x == 0:
                    continue
                if ky[f] >= 1 and nv[f] > 0:
                    total += _mi_from_minority(xb[f, j, m], vy[f], idx[f], cnt[f], px[f, j, m], k_x, ky[f], cmax[f], nv[f])
                cells += 1
            sc = total / cells if cells > 0 else -np.inf
            if sc > best:
                best = sc
                pick[j] = m
    return pick


@njit(cache=True, parallel=True, nogil=True)
def _fused_block(perms, y_raw, yc, rows, xb, kx, nv, val_idx, px, minority, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff, out):
    """``out[b]`` = the pool's largest corrected MI under permutation ``perms[b]``, ``-inf`` when a column had no scored candidate (the caller declines).

    ``minority`` picks the winners with :func:`_winners_minority` (bin counts ``px`` from :func:`_px_table`, the dispatched variant) or with the exact :func:`_winners` (``px``
    is then unused, pass an empty array): the same ``out`` either way, which ``test_honest_oj_minority`` pins bit for bit."""
    for b in prange(perms.shape[0]):
        perm = perms[b]
        vy, ky = _fold_targets(perm, y_raw, nv, val_idx)
        if minority:
            pick = _winners_minority(xb, kx, nv, vy, ky, px)
        else:
            pick = _winners(xb, kx, nv, vy, ky)
        ycp, ny, ky_eff = _permuted_codes(perm, yc, rows)
        out[b] = _pool_max(pick, ycp, ny, ky_eff, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff)


def _declined(reason: str) -> None:
    """Log why the fused floor stepped aside (once per reason) and return ``None``: the per-permutation floor then runs, many times slower."""
    from mlframe.utils.log_throttle import log_throttle

    log_throttle(logger, "honest_oj_fused_declined:" + reason, logging.INFO, "fused optimal_joint honest floor not used: %s; the per-permutation floor runs", reason)
    return None


@njit(cache=True, parallel=True, nogil=True)
def _perm_block(n, base_seed, b0, b1):
    """``(b1 - b0, n)`` permutations ``seeded_permutation(n, base_seed + b)`` for ``b`` in ``[b0, b1)``, drawn in parallel (the same streams)."""
    out = np.empty((b1 - b0, n), dtype=np.int64)
    for i in prange(b1 - b0):
        out[i] = seeded_permutation(n, base_seed + b0 + i)
    return out


@njit(cache=True, parallel=True, nogil=True)
def _occupied_bins(pool_rc):
    """``(n_pool, n_counts)`` float number of distinct codes of every pool code row."""
    n_pool, n_m = pool_rc.shape[0], pool_rc.shape[1]
    out = np.zeros((n_pool, n_m), dtype=np.float64)  # Intentional float64: Miller-Madow support term
    for t in prange(n_pool):
        for m in range(n_m):
            out[t, m] = float(_n_unique_codes(pool_rc[t, m]))
    return out


def _assemble_pool(binner, pool, width, dep_rows, ms, factors_data, factors_nbins, n, sub, free_rows):
    """``(pool_dep, pool_codes, pool_nb)``: for each pool column its re-binned column index (``-1`` for a fixed one), its code rows per candidate count in
    the binner's code dtype (a fixed column fills candidate 0), and the bin counts. ``pool_dep`` indexes the packed plans (``sub``); a re-binned column with
    target-free edges (``free_rows``) is placed like a fixed one."""
    pool_dep = np.full(len(pool), -1, dtype=np.int64)
    with binner._lock:
        table, col_max = binner._table, binner._col_max
    code_dtype = np.promote_types(table.dtype, np.asarray(factors_data).dtype) if any(c >= width for c in pool) else table.dtype
    pool_codes = np.zeros((len(pool), len(ms), n), dtype=code_dtype)
    pool_nb = np.zeros((len(pool), len(ms)), dtype=np.int64)
    for t, c in enumerate(pool):
        if c < width and int(binner._out_dep[c]) in free_rows:
            s = int(free_rows[int(binner._out_dep[c])])
            pool_codes[t, 0] = table[s]
            pool_nb[t, 0] = int(col_max[s]) + 1
        elif c < width and int(binner._out_dep[c]) >= 0:
            j = int(binner._out_dep[c])
            pool_dep[t] = sub[j]
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
        log_throttle(logger, "_honest_oj_fused.except_265", logging.WARNING, "fused optimal_joint honest floor not used (%s: %s); per-permutation path", type(exc).__name__, exc)
        return None


class _DeclineError(Exception):
    """The fused floor does not apply; the message is the reason logged by :func:`_declined`."""


def _cached_plans(honest_codes_fn, binner, cache, n, base_seed):
    """The cached plan of every re-binned column (``None`` where the column has none), building them through one permutation's normal route if needed."""
    from ._honest_mdlp_null import seeded_permutation

    candidates = tuple(binner._kwargs.get("candidates", (4, 8, 16, 32)))
    n_splits, seed = binner._kwargs.get("n_splits", 3), binner._kwargs.get("random_state", 0)

    def _plans():
        """The cached plans now."""
        return [cache.get(binner._arr[:, j], candidates, n_splits, "quantile", seed) for j in range(binner._arr.shape[1])]

    plans = _plans()
    if any(pl is None for pl in plans):
        # the plans are built by a permutation's normal route: run the first permutation through the hook once (its codes are the production ones)
        honest_codes_fn(seeded_permutation(n, base_seed))
        plans = _plans()
    return plans, candidates


def _free_rows(binner, plans, y_int, n, base_seed):
    """Code-table slot of every re-binned column without a plan. Such a column took an early, target-free edge rule (few values / too few rows): its row
    must not depend on the target, checked on two different permutations; the row then serves every permutation like a fixed column's."""
    from ._honest_mdlp_null import seeded_permutation

    out: dict = {}
    for j in range(len(plans)):
        if plans[j] is not None:
            continue
        s0 = binner._edges_slot(j, np.ascontiguousarray(y_int[seeded_permutation(n, base_seed)]))
        s1 = binner._edges_slot(j, np.ascontiguousarray(y_int[seeded_permutation(n, base_seed + 1)]))
        if s0 is None or s1 is None or s0 != s1:
            raise _DeclineError("a re-binned column without a plan has target-dependent edges")
        out[j] = s0
    return out


def _count_slot(binner, plan, j, m, key, y_int):
    """Code-table slot of column ``j`` binned with ``m`` bins, through the binner's own edge path with ``m`` planted as the plan's winner."""
    s = binner._by_count.get((j, m))
    if s is not None:
        return s
    best = plan.setdefault("best", {})
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
        raise _DeclineError("the binner's code table is at its byte budget")
    binner._by_count[(j, m)] = s
    return s


def _pool_arrays(binner, cols, dep_rows, ms, factors_data, factors_nbins, n, sub, free_rows):
    """``(pool_dep, pool_codes, pool_nb)``, assembled once per (pool, rows count) and kept on the binner: the floor runs twice per fit over the same pool."""
    pool = [int(c) for c in cols]
    memo_key = (tuple(pool), n)
    memo = getattr(binner, "_fused_pool_memo", None)
    if memo is not None and memo[0] == memo_key:
        return memo[1]
    arrays = _assemble_pool(binner, pool, len(binner._out_names), dep_rows, ms, factors_data, factors_nbins, n, sub, free_rows)
    binner._fused_pool_memo = (memo_key, arrays)
    return arrays


def _val_index(packed):
    """``(nv, val_idx)``: validation rows count and indices of every fold."""
    folds = packed["folds"]
    nv = np.asarray(packed["nv"], dtype=np.int64)
    val_idx = np.zeros((len(folds), int(nv.max())), dtype=np.int64)
    for f, (_k, vm) in enumerate(folds):
        idx = np.flatnonzero(vm)
        val_idx[f, : idx.size] = idx
    return nv, val_idx


def _score_block(dev, dev_pool, perms, y_int, yc, rows, xb, kx, nv, val_idx, px, pool_args):
    """``(null maxima, plain MI (B, pool) or None)`` of one block of permutations: the winners on the device when ``dev`` is set and the pool there too when
    ``dev_pool`` is (which also returns every pool column's MI), else the CPU kernels."""
    out = np.empty(perms.shape[0], dtype=np.float64)  # Intentional float64: statistic vector
    picks = None
    if dev is not None:
        try:
            picks = dev.picks(perms)
        except Exception as exc:
            logger.debug("device winners failed (%s: %s); CPU kernel", type(exc).__name__, exc)
    if picks is not None and dev_pool is not None:
        try:
            return dev_pool.maxima(perms, picks, pool_args[-1]), dev_pool.last_mi
        except Exception as exc:
            logger.debug("device pool failed (%s: %s); CPU pool", type(exc).__name__, exc)
    if picks is not None:
        from ._honest_oj_fused_gpu import _pool_block

        _pool_block(perms, picks, yc, rows, *pool_args, out)
    else:
        _fused_block(perms, y_int, yc, rows, xb, kx, nv, val_idx, px, True, *pool_args, out)
    return out, None


def _floor(honest_codes_fn, binner, y_raw, factors_data, factors_nbins, y_idx, cols, n_permutations, quantile, base_seed, eval_rows):
    """Body of :func:`oj_fused_floor`."""
    try:
        return _floor_body(honest_codes_fn, binner, y_raw, factors_data, factors_nbins, y_idx, cols, n_permutations, quantile, base_seed, eval_rows)
    except _DeclineError as exc:
        logger.debug("fused optimal_joint floor declined: %s", exc)
        return _declined(str(exc))


def _floor_body(honest_codes_fn, binner, y_raw, factors_data, factors_nbins, y_idx, cols, n_permutations, quantile, base_seed, eval_rows):
    """:func:`_floor`, raising :class:`_DeclineError` where the fused path does not apply."""
    with parallel_kernel_entry():
        from ._adaptive_nbins import _distinct_exceeds
        from ._optimal_joint_edges import pack_plans

        max_y = int(binner._kwargs.get("optimal_joint_max_y_classes", 64))
        if y_raw.dtype.kind not in "iub" or y_raw.size == 0 or int(y_raw.min()) < 0 or _distinct_exceeds(y_raw, max_y):
            raise _DeclineError("the raw target is not small non-negative integer labels")
        y_int = np.ascontiguousarray(y_raw, dtype=np.int64)
        cache = binner._kwargs.get("optimal_joint_plan_cache")
        if cache is None:
            raise _DeclineError("no optimal_joint plan cache on the binner")
        n = int(factors_data.shape[0])
        plans, candidates = _cached_plans(honest_codes_fn, binner, cache, n, base_seed)
        planned = [j for j in range(len(plans)) if plans[j] is not None]
        if not planned:
            raise _DeclineError("no column has an optimal_joint plan")
        packed = pack_plans([plans[j] for j in planned], candidates)
        if packed is None:
            raise _DeclineError("the column plans do not pack (differing folds or wide codes)")
        ms = list(packed["ms"])
        key = cache._target_key(y_int)
        free_rows = _free_rows(binner, plans, y_int, n, base_seed)
        sub = {j: i for i, j in enumerate(planned)}  # re-binned column -> its place among the packed plans
        dep_rows = np.full((len(plans), len(ms)), -1, dtype=np.int64)
        from concurrent.futures import ThreadPoolExecutor

        from ._adaptive_nbins import _resolve_edge_jobs

        def _column_slots(j):
            """The code-table slots of column ``j`` under every candidate count (columns are independent: own plan, own table rows under the binner's lock)."""
            return j, [_count_slot(binner, plans[j], j, m, key, y_int) for m in ms]

        with ThreadPoolExecutor(max_workers=_resolve_edge_jobs(None, len(planned))) as pool_ex:
            for j, slots in pool_ex.map(_column_slots, planned):
                dep_rows[j] = slots
        rows = np.arange(n, dtype=np.int64) if eval_rows is None else np.asarray(eval_rows, dtype=np.int64)
        pool_dep, pool_codes, pool_nb = _pool_arrays(binner, cols, dep_rows, ms, factors_data, factors_nbins, n, sub, free_rows)
        nv, val_idx = _val_index(packed)
        yc = np.ascontiguousarray(factors_data[:, int(y_idx)]).astype(np.int64)
        # the pool's codes at the evaluation rows and their occupied-bin counts do not depend on the permutation: built once here, not per permutation in the kernel
        pool_rc = np.ascontiguousarray(pool_codes[:, :, rows])
        pool_kxe = _occupied_bins(pool_rc)
        kx = np.asarray(packed["kx"], dtype=np.int64)
        px = _px_table(packed["xb"], kx, nv)  # per-cell bin counts: the winners scatter only the rows outside the majority class
        from ._honest_oj_fused_gpu import device_pool, device_winners

        dev = device_winners(packed["xb"], kx, nv, val_idx, y_int, px, n)
        dev_pool = device_pool(dev, pool_dep, pool_rc, pool_nb, pool_kxe, yc, rows)
        pool_args = (pool_dep, pool_rc, pool_nb, pool_kxe, float(len(rows)))
        nulls = np.empty(int(n_permutations), dtype=np.float64)  # Intentional float64: statistic vector (null max), exact permutation ties
        mi_blocks: "list | None" = []
        for b0 in range(0, int(n_permutations), _PERM_BLOCK):
            b1 = min(int(n_permutations), b0 + _PERM_BLOCK)
            perms = _perm_block(n, base_seed, b0, b1)
            out, mi = _score_block(dev, dev_pool, perms, y_int, yc, rows, packed["xb"], kx, nv, val_idx, px, pool_args)
            if np.isneginf(out).any():
                raise _DeclineError("a column had no scored candidate count in some permutation")
            nulls[b0:b1] = out
            mi_blocks = None if (mi_blocks is None or mi is None) else [*mi_blocks, mi]
        if mi_blocks:
            binner._fused_null_mi = (_null_key(base_seed, n_permutations, n, rows), {int(c): t for t, c in enumerate(cols)}, np.concatenate(mi_blocks))
        return float(np.quantile(nulls, float(quantile), method="higher"))


def _null_key(base_seed, n_permutations, n, rows) -> tuple:
    """What the pool's per-permutation MI depends on besides the binner: the permutation stream, its length, the rows count and the evaluation rows."""
    import hashlib

    return int(base_seed), int(n_permutations), int(n), hashlib.blake2b(np.ascontiguousarray(rows, dtype=np.int64).view(np.uint8).data, digest_size=16).hexdigest()


def fused_column_nfailed(honest_codes_fn: Any, col: int, observed: float, n_permutations: int, base_seed: int, n: int, rows: np.ndarray) -> Optional[int]:
    """``honest_candidate_nfailed``'s exceedance count of pool column ``col`` from the per-permutation MI the fused floor kept (same permutations, rows and
    codes: the column's winning code row under each permuted target; the candidate's shorter run is the head of the floor's stream), or ``None`` when the
    floor kept none for this stream and rows or the column is not in its pool."""
    binner = getattr(honest_codes_fn, "binner", None)
    kept = getattr(binner, "_fused_null_mi", None)
    if binner is None or kept is None:
        return None
    k_seed, k_perms, k_n, k_rows = kept[0]
    # permutation b of a stream is seeded ``base_seed + b``: a shorter run of the same stream is the floor's first ``n_permutations``
    if (k_seed, k_n) != (int(base_seed), int(n)) or int(n_permutations) > k_perms or k_rows != _null_key(base_seed, n_permutations, n, rows)[3]:
        return None
    t = kept[1].get(int(col))
    if t is None or int(col) >= len(binner._out_names):
        return None  # an engineered column appended after binning: the honest codes do not cover it (the caller's contract is ``None``)
    return int(np.count_nonzero(kept[2][: int(n_permutations), t] >= observed))
