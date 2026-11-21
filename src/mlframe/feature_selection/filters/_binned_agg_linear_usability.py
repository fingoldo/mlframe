"""Linear-usability check for a binned-aggregate column that the MI redundancy gate rejected.

The redundancy gate scores ``CMI(col; y | qbin(group), qbin(agg))``. A binned-aggregate column is, by construction, a function of
the group column's cell, which is the same partition the gate conditions on, so the column is rejected as redundant whenever the
raw group already explains ``y`` in the tree-like, binned sense. A low-capacity downstream model cannot use that group directly: with
``aux ~ N(0, sigma(g))`` and ``y = sigma(g) + noise`` a ridge model on the raw columns scores held-out R^2 of 0.00 and 0.86 once the
cell-spread column is added, while a boosted-tree model scores 0.89 either way. This module measures that gap directly.

The gain is the out-of-fold increase in R^2 of a linear model of the class indicators of ``y`` on the raw sources (group and agg
values) when the candidate column is added. The acceptance bound is calibrated, not fixed. The null must keep what the candidate is: a
function of the group's cell. Shuffling the candidate across rows destroys that and makes every aggregate of a noise column look
useful whenever the group itself is informative (a random value per cell correlates with a target that varies by cell). The null is
therefore the candidate with its per-cell values permuted among the cells.

The decision is a permutation test, not a z bound: with about ten cells the correlation of a random per-cell column is bounded by one,
so a family-wise z of four standard deviations would lie outside what any permutation can reach and nothing would pass. The column is kept
when the share of permutations whose gain is at least the observed one, ``(hits + 1) / (draws + 1)``, is at most ``level``. That needs about
``2 / level`` draws, so they are made in batches and the test stops as soon as the level can no longer be reached.

The statistics of one pair (mean, std, skew, kurt of the agg column over the cells of the group) are tested together with a single-step
max-T adjustment: every permutation of the cells is applied to all of them and a column's p-value is the share of permutations whose largest
gain is at least its own. The caller splits ``0.05`` over the pairs (Bonferroni), so the chance of keeping any column of any pair under the null
is at most ``0.05``. Splitting over every column instead would put the level below what a genuine column can reach: with ten cells its p-value
is near ``1e-4`` to ``1e-3``, the resolution of the test.

Everything that depends only on the (group, agg) pair is built once per pair: the row subsample, the folds, the base design (intercept
and raw sources), the class indicators, the cells, the base model's per-fold errors and the per-(fold, cell) sums. A permutation then costs a
few small array products and no pass over the rows.
"""

from __future__ import annotations

import math
import zlib

import numpy as np

from mlframe.feature_selection.filters._internals import dense_codes, stable_argsort_codes
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from typing import Any

from numba import njit

_MAX_ROWS = 100_000
_N_FOLDS = 5
_MAX_CLASSES = 20
_MIN_ROWS = 50
_RIDGE = 1e-8
_MIN_DRAWS = 2_000
_MAX_DRAWS = 40_000
_BATCH = 500


def _to_host(a) -> np.ndarray:
    """A host double precision array from a numpy or device array."""
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=_crit_np_dtype())


def _rows_to_host(a: Any, rows: Any) -> np.ndarray:
    """``_to_host(a)[rows]`` with the rows taken first: a device array is indexed on the device, so only the subsampled rows cross to the host (not the full column)."""
    if hasattr(a, "get"):
        import cupy as cp

        idx = rows if hasattr(rows, "get") else cp.asarray(rows)
        return np.asarray(a[idx].get(), dtype=_crit_np_dtype())
    picked: np.ndarray = _to_host(a)[rows]
    return picked


@njit(cache=True, nogil=True)
def _standardise_kernel(col: np.ndarray, out: np.ndarray) -> None:
    """``out`` = ``col`` centred and scaled by the mean and standard deviation of its finite entries (double precision), 0 for a non-finite entry and for a constant column."""
    m = col.shape[0]
    count = 0
    total = 0.0
    for i in range(m):
        v = col[i]
        if np.isfinite(v):
            count += 1
            total += v
    mean = total / count if count else 0.0
    sq = 0.0
    for i in range(m):
        v = col[i]
        if np.isfinite(v):
            d = v - mean
            sq += d * d
    std = np.sqrt(sq / count) if count else 0.0
    for i in range(m):
        v = col[i]
        out[i] = (v - mean) / std if (std > 0.0 and np.isfinite(v)) else 0.0


def _standardised(col: np.ndarray) -> np.ndarray:
    """``col`` centred and scaled, non-finite entries set to the mean; a constant column becomes all zeros. Double precision.

    One compiled pass without the interpreter lock for a float column (the numpy form, ``_standardised_numpy``, is eight array operations, each of which re-takes the lock: 2.3 ms per call
    inside a fit against 0.3 ms alone, six calls per pair). The moments are accumulated in double precision, so the values agree with the numpy form of a float32 column to about 1e-6."""
    arr = np.asarray(col)
    if arr.ndim == 1 and arr.dtype.kind == "f":
        out = np.empty(arr.shape[0], dtype=np.float64)  # Intentional float64: design columns of the normal equations
        _standardise_kernel(np.ascontiguousarray(arr), out)
        return out
    return np.asarray(_standardised_numpy(arr), dtype=np.float64)  # Intentional float64: design columns of the normal equations


def _standardised_numpy(col: np.ndarray) -> np.ndarray:
    """The numpy form of :func:`_standardised` (its reference, and the route of a non-float column)."""
    finite = np.isfinite(col)
    if not finite.any():
        return np.zeros_like(col)
    mean = float(col[finite].mean())
    std = float(col[finite].std())
    out = np.where(finite, col - mean, 0.0)
    return out / std if std > 0 else np.zeros_like(col)


@njit(cache=True, nogil=True)
def _fold_cross_sums(base: np.ndarray, Y: np.ndarray, cand: np.ndarray, starts: np.ndarray) -> tuple:
    """Per-fold cross sums of one candidate in the fold-sorted row order: ``u[f, j] = sum base[:, j] * cand``, ``cc[f] = sum cand^2``, ``cy[f, c] = sum cand * Y[:, c]`` in one pass over the rows (three
    ``reduceat`` passes with a ``(rows, p)`` and a ``(rows, k)`` temporary before); the loops release the GIL, so the pair pool scales."""
    n_f = starts.shape[0]
    m = cand.shape[0]
    p = base.shape[1]
    k = Y.shape[1]
    u = np.zeros((n_f, p), dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    cc = np.zeros(n_f, dtype=np.float64)
    cy = np.zeros((n_f, k), dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    for f in range(n_f):
        hi = starts[f + 1] if f + 1 < n_f else m
        for i in range(starts[f], hi):
            c = cand[i]
            cc[f] += c * c
            for j in range(p):
                u[f, j] += base[i, j] * c
            for j in range(k):
                cy[f, j] += Y[i, j] * c
    return u, cc, cy


def _solve_batch(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Ridge-stabilised solutions of the stacked normal equations ``gram[..., i, :, :] @ beta = rhs[..., i, :, :]`` over any leading axes."""
    size = gram.shape[-1]
    reg = gram.copy()
    reg[..., np.arange(size), np.arange(size)] += _RIDGE * np.trace(gram, axis1=-2, axis2=-1)[..., None]
    return np.linalg.solve(reg, rhs)


@njit(cache=True, nogil=True)
def _fold_grams(base: np.ndarray, Y: np.ndarray, starts: np.ndarray) -> tuple:
    """Per-fold ``base' base``, ``base' Y``, ``sum Y^2`` and the total sum of squares of ``Y`` about its fold means, for the fold-sorted rows: one pass over the rows per fold without the
    interpreter lock (the numpy form, kept as the reference of the kernel tests, is two products, two reductions and their temporaries per fold)."""
    n_f = starts.shape[0]
    m, p = base.shape
    k = Y.shape[1]
    gram = np.zeros((n_f, p, p), dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    rhs = np.zeros((n_f, p, k), dtype=np.float64)
    yy = np.zeros(n_f, dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    sst = np.zeros(n_f, dtype=np.float64)
    col_sum = np.zeros(k, dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    for f in range(n_f):
        lo = starts[f]
        hi = starts[f + 1] if f + 1 < n_f else m
        col_sum[:] = 0.0
        for i in range(lo, hi):
            for a in range(p):
                va = base[i, a]
                for b in range(a, p):
                    gram[f, a, b] += va * base[i, b]
                for j in range(k):
                    rhs[f, a, j] += va * Y[i, j]
            for j in range(k):
                v = Y[i, j]
                yy[f] += v * v
                col_sum[j] += v
        for a in range(p):
            for b in range(a):
                gram[f, a, b] = gram[f, b, a]
        rows = hi - lo
        if rows > 0:
            for i in range(lo, hi):
                for j in range(k):
                    d = Y[i, j] - col_sum[j] / rows
                    sst[f] += d * d
    return gram, rhs, yy, sst


def _prepare(base: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Fold-sorted base design and targets with every per-fold quantity that does not depend on the candidate, including the base out-of-fold errors."""
    order = stable_argsort_codes(folds, int(folds.max()) + 1 if folds.size else 1)  # fold ids are small integers: the radix order, the same permutation as the stable comparison sort
    base_s = np.ascontiguousarray(base[order], dtype=np.float64)  # Intentional float64: design matrix of the normal equations
    Y_s = np.ascontiguousarray(Y[order], dtype=np.float64)  # Intentional float64: class indicators of the normal equations
    starts = np.concatenate([[0], np.cumsum(np.bincount(folds))])[:-1]
    gram, rhs_arr, yy_arr, sst = _fold_grams(base_s, Y_s, np.ascontiguousarray(starts, dtype=np.int64))
    beta = _solve_batch(gram.sum(axis=0) - gram, rhs_arr.sum(axis=0) - rhs_arr)
    sse_base = yy_arr - 2.0 * (beta * rhs_arr).sum(axis=(1, 2)) + (beta * (gram @ beta)).sum(axis=(1, 2))
    a_train = gram.sum(axis=0) - gram
    a_train = a_train + _RIDGE * np.trace(a_train, axis1=-2, axis2=-1)[:, None, None] * np.eye(a_train.shape[-1])
    a_inv = np.linalg.inv(a_train)
    b_train = rhs_arr.sum(axis=0) - rhs_arr
    return {
        "order": order, "base": base_s, "Y": Y_s, "starts": starts, "gram": gram, "rhs": rhs_arr, "yy": yy_arr, "sst": sst, "sse_base": sse_base,
        "a_inv": a_inv, "b_train": b_train, "a_inv_b": a_inv @ b_train,
    }


def _gains_from_sums(prep: dict, u: np.ndarray, cc: np.ndarray, cy: np.ndarray) -> np.ndarray:
    """Per-fold ``R^2(base + cand) - R^2(base)`` for ``D`` candidates, from their per-fold cross sums ``u`` (D, folds, p), squares ``cc`` (D, folds) and class sums ``cy`` (D, folds, k).

    The training normal equations are the base ones bordered by the candidate, so they are solved through the Schur complement of the base block,
    whose inverse is computed once per pair: no linear solve per candidate or permutation.
    """
    u_tr = u.sum(axis=1, keepdims=True) - u
    cc_tr = cc.sum(axis=1, keepdims=True) - cc
    cy_tr = cy.sum(axis=1, keepdims=True) - cy
    w = np.einsum("kpq,dkq->dkp", prep["a_inv"], u_tr)
    schur = cc_tr - np.einsum("dkp,dkp->dk", u_tr, w)
    usable = schur > 1e-12 * np.maximum(cc_tr, 1e-300)
    beta_c = (cy_tr - np.einsum("dkp,kpj->dkj", w, prep["b_train"])) / np.where(usable, schur, 1.0)[..., None]
    beta_c = np.where(usable[..., None], beta_c, 0.0)
    beta_b = prep["a_inv_b"][None] - w[..., None] * beta_c[:, :, None, :]
    explained = np.einsum("dkpj,kpj->dk", beta_b, prep["rhs"]) + np.einsum("dkj,dkj->dk", beta_c, cy)
    quad = (
        np.einsum("dkpj,kpq,dkqj->dk", beta_b, prep["gram"], beta_b)
        + 2.0 * np.einsum("dkpj,dkp,dkj->dk", beta_b, u, beta_c)
        + cc * np.einsum("dkj,dkj->dk", beta_c, beta_c)
    )
    sse_full = prep["yy"] - 2.0 * explained + quad
    sst = prep["sst"]
    return np.where(sst > 0, (prep["sse_base"] - sse_full) / np.where(sst > 0, sst, 1.0), 0.0)


def _gain(prep: dict, cand_sorted: np.ndarray) -> np.ndarray:
    """Per-fold out-of-fold gain of one candidate given in the fold-sorted row order of ``prep``."""
    u, cc, cy = _fold_cross_sums(prep["base"], prep["Y"], np.ascontiguousarray(cand_sorted, dtype=np.float64), prep["starts"])  # Intentional float64: statistic / reference vector in double precision (compared exactly with its host twin)
    gains: np.ndarray = _gains_from_sums(prep, u[None], cc[None], cy[None])[0]
    return gains


@njit(cache=True, nogil=True)
def _cell_sums_kernel(base: np.ndarray, Y: np.ndarray, starts: np.ndarray, cells: np.ndarray, n_cells: int) -> tuple:
    """Per-(fold, cell) row counts and sums of the columns of ``base`` and ``Y`` in one pass over the fold-sorted rows (the row order of the ``bincount`` form, so the sums are the same)."""
    n_f = starts.shape[0]
    m, p = base.shape
    k = Y.shape[1]
    count = np.zeros((n_f, n_cells), dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    sb = np.zeros((n_f, n_cells, p), dtype=np.float64)
    sy = np.zeros((n_f, n_cells, k), dtype=np.float64)  # Intentional float64: double-precision accumulator / output buffer: sums and statistics keep full precision
    for f in range(n_f):
        hi = starts[f + 1] if f + 1 < n_f else m
        for i in range(starts[f], hi):
            c = cells[i]
            count[f, c] += 1.0
            for j in range(p):
                sb[f, c, j] += base[i, j]
            for j in range(k):
                sy[f, c, j] += Y[i, j]
    return count, sb, sy


def _cell_sums(prep: dict, cells_s: np.ndarray, n_cells: int) -> dict:
    """Per-(fold, cell) counts and sums of the base columns and class indicators, so a column that is constant per cell needs no pass over the rows."""
    count, sb, sy = _cell_sums_kernel(prep["base"], prep["Y"], np.ascontiguousarray(prep["starts"], dtype=np.int64), np.ascontiguousarray(cells_s, dtype=np.int64), int(n_cells))
    return {"count": count.astype(_crit_np_dtype()), "base": sb, "Y": sy}


def _mean_gains_by_cell(prep: dict, sums: dict, cell_values: np.ndarray) -> np.ndarray:
    """Mean over folds of the gain for each row of ``cell_values`` (D, cells): the column equal to ``cell_values[d][cell]`` on every row."""
    u = np.einsum("kcp,dc->dkp", sums["base"], cell_values)
    cc = (cell_values * cell_values) @ sums["count"].T
    cy = np.einsum("kcj,dc->dkj", sums["Y"], cell_values)
    mean_gains: np.ndarray = _gains_from_sums(prep, u, cc, cy).mean(axis=1)
    return mean_gains


def _pair_state(X, srcs, y_cls, cell_ids, pair_key: str):
    """Everything that depends only on the (group, agg) pair, or None when the pair cannot be checked (too few rows, classes or cells)."""
    rng = np.random.default_rng(zlib.crc32(pair_key.encode("utf-8")))
    y_all = np.asarray(y_cls).ravel()
    n = y_all.shape[0]
    rows = np.sort(rng.choice(n, size=_MAX_ROWS, replace=False)) if n > _MAX_ROWS else np.arange(n)
    classes = np.unique(y_all[rows])
    if rows.size < _MIN_ROWS or classes.size < 2 or classes.size > _MAX_CLASSES:
        return None
    cells = dense_codes(np.asarray(cell_ids).ravel()[rows])
    n_cells = int(cells.max()) + 1
    if n_cells < 3:
        return None
    src = np.column_stack([np.asarray(X[c].to_numpy(), dtype=_crit_np_dtype())[rows] for c in srcs])  # the sampled rows of each source column: no copy of the whole sub-frame first (1.5 -> 0.3 ms per pair)
    base = np.column_stack([np.ones(rows.size)] + [_standardised(src[:, j]) for j in range(src.shape[1])])
    Y = (y_all[rows][:, None] == classes[None, :]).astype(_crit_np_dtype())
    folds = np.empty(rows.size, dtype=np.int64)
    folds[rng.permutation(rows.size)] = np.arange(rows.size) % _N_FOLDS
    prep = _prepare(base, Y, folds)
    cells_s = cells[prep["order"]]
    return {"rows": rows, "prep": prep, "cells_s": cells_s, "n_cells": n_cells, "sums": _cell_sums(prep, cells_s, n_cells)}


def linear_usability_keeps(X: Any, srcs: Any, cands: dict, y_cls: Any, cell_ids: Any, level: float, pair_key: str) -> set:
    """Names of the candidates of ONE (group, agg) pair that improve an out-of-fold linear fit of ``y`` beyond what a random per-cell column does.

    ``cands`` maps a candidate name to its numeric values or bin codes, ``X[srcs]`` are the raw group and agg columns, ``y_cls`` the
    integer class labels, ``cell_ids`` the group column's cell of every row and ``level`` the family-wise level for this pair. The
    statistics of one pair are tested together: every permutation of the cells is applied to all candidates, and a candidate's p-value is
    the share of permutations whose LARGEST gain over the candidates is at least its observed gain (a single-step max-T adjustment), so
    the chance that any candidate of a pair is kept under the null is at most ``level``. Rows are subsampled to ``_MAX_ROWS``. A candidate
    whose gain is not above 2 standard errors over the folds, or that is constant, is dropped before the permutations.
    """
    state = _pair_state(X, srcs, y_cls, cell_ids, pair_key)
    if state is None:
        return set()
    prep, cells_s, n_cells = state["prep"], state["cells_s"], state["n_cells"]
    names, gains, means = [], [], []
    cell_counts = np.maximum(np.bincount(cells_s, minlength=n_cells), 1)
    rows_dev = None
    if any(hasattr(c, "get") for c in cands.values()):
        import cupy as cp

        rows_dev = cp.asarray(state["rows"])  # one index upload per pair instead of one per candidate
    for name, codes in cands.items():
        cand = _standardised(_rows_to_host(codes, rows_dev if rows_dev is not None and hasattr(codes, "get") else state["rows"]))
        if not cand.any():
            continue
        cand_s = cand[prep["order"]]
        per_fold = _gain(prep, cand_s)
        gain = float(per_fold.mean())
        if gain <= 0.0 or gain <= 2.0 * float(per_fold.std(ddof=1)) / math.sqrt(per_fold.size):
            continue
        names.append(name)
        gains.append(gain)
        means.append(np.bincount(cells_s, weights=cand_s, minlength=n_cells) / cell_counts)
    if not names:
        return set()
    total = int(min(_MAX_DRAWS, max(_MIN_DRAWS, math.ceil(2.0 / level)))) if level > 0 else _MAX_DRAWS
    rng = np.random.default_rng(zlib.crc32(pair_key.encode("utf-8")) + 1)
    hits = np.zeros(len(names), dtype=np.int64)
    done = 0
    while done < total:
        size = min(_BATCH, total - done)
        order = np.argsort(rng.random((size, n_cells)), axis=1)
        top = np.max([_mean_gains_by_cell(prep, state["sums"], m[order]) for m in means], axis=0)
        hits += (top[:, None] >= np.asarray(gains)[None, :]).sum(axis=0)
        done += size
        if ((hits + 1) / (total + 1) > level).all():
            return set()
    return {name for name, h in zip(names, hits) if (h + 1) / (total + 1) <= level}
