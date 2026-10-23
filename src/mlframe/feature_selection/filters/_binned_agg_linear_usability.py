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
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype

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


def _standardised(col: np.ndarray) -> np.ndarray:
    """``col`` centred and scaled, non-finite entries set to the mean; a constant column becomes all zeros."""
    finite = np.isfinite(col)
    if not finite.any():
        return np.zeros_like(col)
    mean = float(col[finite].mean())
    std = float(col[finite].std())
    out = np.where(finite, col - mean, 0.0)
    return out / std if std > 0 else np.zeros_like(col)


def _solve_batch(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Ridge-stabilised solutions of the stacked normal equations ``gram[..., i, :, :] @ beta = rhs[..., i, :, :]`` over any leading axes."""
    size = gram.shape[-1]
    reg = gram.copy()
    reg[..., np.arange(size), np.arange(size)] += _RIDGE * np.trace(gram, axis1=-2, axis2=-1)[..., None]
    return np.linalg.solve(reg, rhs)


def _prepare(base: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Fold-sorted base design and targets with every per-fold quantity that does not depend on the candidate, including the base out-of-fold errors."""
    order = np.argsort(folds, kind="stable")
    base_s, Y_s = base[order], Y[order]
    starts = np.concatenate([[0], np.cumsum(np.bincount(folds))])[:-1]
    bounds = np.concatenate([starts, [base_s.shape[0]]])
    grams, rhs, yy, sst = [], [], [], []
    for f in range(starts.size):
        d, y = base_s[bounds[f] : bounds[f + 1]], Y_s[bounds[f] : bounds[f + 1]]
        grams.append(d.T @ d)
        rhs.append(d.T @ y)
        yy.append(float((y * y).sum()))
        sst.append(float(((y - y.mean(axis=0)) ** 2).sum()))
    gram, rhs_arr = np.stack(grams), np.stack(rhs)
    beta = _solve_batch(gram.sum(axis=0) - gram, rhs_arr.sum(axis=0) - rhs_arr)
    yy_arr = np.asarray(yy)
    sse_base = yy_arr - 2.0 * (beta * rhs_arr).sum(axis=(1, 2)) + (beta * (gram @ beta)).sum(axis=(1, 2))
    a_train = gram.sum(axis=0) - gram
    a_train = a_train + _RIDGE * np.trace(a_train, axis1=-2, axis2=-1)[:, None, None] * np.eye(a_train.shape[-1])
    a_inv = np.linalg.inv(a_train)
    b_train = rhs_arr.sum(axis=0) - rhs_arr
    return {
        "order": order, "base": base_s, "Y": Y_s, "starts": starts, "gram": gram, "rhs": rhs_arr, "yy": yy_arr, "sst": np.asarray(sst), "sse_base": sse_base,
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
    starts = prep["starts"]
    u = np.add.reduceat(prep["base"] * cand_sorted[:, None], starts, axis=0)
    cc = np.add.reduceat(cand_sorted * cand_sorted, starts)
    cy = np.add.reduceat(cand_sorted[:, None] * prep["Y"], starts, axis=0)
    gains: np.ndarray = _gains_from_sums(prep, u[None], cc[None], cy[None])[0]
    return gains


def _cell_sums(prep: dict, cells_s: np.ndarray, n_cells: int) -> dict:
    """Per-(fold, cell) counts and sums of the base columns and class indicators, so a column that is constant per cell needs no pass over the rows."""
    n_folds = prep["starts"].size
    fold_s = np.repeat(np.arange(n_folds), np.diff(np.concatenate([prep["starts"], [cells_s.size]])))
    flat = fold_s * n_cells + cells_s
    size = n_folds * n_cells

    def summed(weights: np.ndarray) -> np.ndarray:
        """Per-(fold, cell) sums of every column of ``weights``, shaped (folds, cells, columns)."""
        return np.stack([np.bincount(flat, weights=weights[:, j], minlength=size) for j in range(weights.shape[1])], axis=1).reshape(n_folds, n_cells, -1)

    return {"count": np.bincount(flat, minlength=size).reshape(n_folds, n_cells).astype(_crit_np_dtype()), "base": summed(prep["base"]), "Y": summed(prep["Y"])}


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
    _, cells = np.unique(np.asarray(cell_ids).ravel()[rows], return_inverse=True)
    n_cells = int(cells.max()) + 1
    if n_cells < 3:
        return None
    src = X[list(srcs)].to_numpy(dtype=_crit_np_dtype())[rows]
    base = np.column_stack([np.ones(rows.size)] + [_standardised(src[:, j]) for j in range(src.shape[1])])
    Y = (y_all[rows][:, None] == classes[None, :]).astype(_crit_np_dtype())
    folds = np.empty(rows.size, dtype=np.int64)
    folds[rng.permutation(rows.size)] = np.arange(rows.size) % _N_FOLDS
    prep = _prepare(base, Y, folds)
    cells_s = cells[prep["order"]]
    return {"rows": rows, "prep": prep, "cells_s": cells_s, "n_cells": n_cells, "sums": _cell_sums(prep, cells_s, n_cells)}


def linear_usability_keeps(X, srcs, cands: dict, y_cls, cell_ids, level: float, pair_key: str) -> set:
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
    for name, codes in cands.items():
        cand = _standardised(_to_host(codes)[state["rows"]])
        if not cand.any():
            continue
        cand_s = cand[prep["order"]]
        per_fold = _gain(prep, cand_s)
        gain = float(per_fold.mean())
        if gain <= 0.0 or gain <= 2.0 * float(per_fold.std(ddof=1)) / math.sqrt(per_fold.size):
            continue
        names.append(name)
        gains.append(gain)
        means.append(np.bincount(cells_s, weights=cand_s, minlength=n_cells) / np.maximum(np.bincount(cells_s, minlength=n_cells), 1))
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
