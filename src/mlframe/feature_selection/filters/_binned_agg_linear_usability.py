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
therefore the candidate with its per-cell values permuted among the cells, and the column is kept when its gain clears
``mean + z * std`` of that null, with ``z`` the family-wise z the caller already uses for the CMI null. A candidate whose gain does not
exceed the largest of the first few null gains is dropped without running the rest.

The base model (intercept and raw sources) does not depend on the candidate, so its per-fold statistics and errors are computed once
per call; each candidate or null draw then needs only its cross sums with the base columns and the class indicators.
"""

from __future__ import annotations

import numpy as np

_MAX_ROWS = 100_000
_N_FOLDS = 5
_MAX_CLASSES = 20
_RIDGE = 1e-8


def _to_host(a) -> np.ndarray:
    """A host float64 array from a numpy or device array."""
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)


def _standardised(col: np.ndarray) -> np.ndarray:
    """``col`` centred and scaled, non-finite entries set to the mean; a constant column becomes all zeros."""
    finite = np.isfinite(col)
    if not finite.any():
        return np.zeros_like(col)
    mean = float(col[finite].mean())
    std = float(col[finite].std())
    out = np.where(finite, col - mean, 0.0)
    return out / std if std > 0 else np.zeros_like(col)


def _solve(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Ridge-stabilised solution of the normal equations ``gram @ beta = rhs``."""
    return np.linalg.solve(gram + _RIDGE * np.trace(gram) * np.eye(gram.shape[0]), rhs)


def _prepare(base: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> dict:
    """Fold-sorted base design and targets with every per-fold quantity that does not depend on the candidate, including the base out-of-fold errors."""
    order = np.argsort(folds, kind="stable")
    base_s, Y_s = base[order], Y[order]
    bounds = np.concatenate([[0], np.cumsum(np.bincount(folds))])
    grams, rhs, yy, sst = [], [], [], []
    for f in range(bounds.size - 1):
        d, y = base_s[bounds[f] : bounds[f + 1]], Y_s[bounds[f] : bounds[f + 1]]
        grams.append(d.T @ d)
        rhs.append(d.T @ y)
        yy.append(float((y * y).sum()))
        sst.append(float(((y - y.mean(axis=0)) ** 2).sum()))
    g_tot, b_tot = sum(grams), sum(rhs)
    sse_base = []
    for f in range(len(grams)):
        beta = _solve(g_tot - grams[f], b_tot - rhs[f])
        sse_base.append(yy[f] - 2.0 * float((beta * rhs[f]).sum()) + float((beta * (grams[f] @ beta)).sum()))
    return {"order": order, "base": base_s, "Y": Y_s, "bounds": bounds, "gram": grams, "rhs": rhs, "yy": yy, "sst": sst, "sse_base": sse_base}


def _gains(prep: dict, cand_sorted: np.ndarray) -> np.ndarray:
    """Per-fold out-of-fold ``R^2(base + cand) - R^2(base)`` for a candidate given in the fold-sorted row order of ``prep``."""
    bounds, base, Y = prep["bounds"], prep["base"], prep["Y"]
    n_folds = bounds.size - 1
    cross, cc, cy = [], [], []
    for f in range(n_folds):
        c = cand_sorted[bounds[f] : bounds[f + 1]]
        cross.append(base[bounds[f] : bounds[f + 1]].T @ c)
        cc.append(float(c @ c))
        cy.append(c @ Y[bounds[f] : bounds[f + 1]])
    cross_tot, cc_tot, cy_tot = sum(cross), sum(cc), sum(cy)
    g_base_tot, b_base_tot = sum(prep["gram"]), sum(prep["rhs"])
    gains = np.empty(n_folds, dtype=np.float64)
    for f in range(n_folds):
        u_te, u_tr = cross[f], cross_tot - cross[f]
        g_te = np.block([[prep["gram"][f], u_te[:, None]], [u_te[None, :], np.array([[cc[f]]])]])
        g_tr = np.block([[g_base_tot - prep["gram"][f], u_tr[:, None]], [u_tr[None, :], np.array([[cc_tot - cc[f]]])]])
        b_te = np.vstack([prep["rhs"][f], cy[f][None, :]])
        b_tr = np.vstack([b_base_tot - prep["rhs"][f], (cy_tot - cy[f])[None, :]])
        beta = _solve(g_tr, b_tr)
        sse_full = prep["yy"][f] - 2.0 * float((beta * b_te).sum()) + float((beta * (g_te @ beta)).sum())
        gains[f] = (prep["sse_base"][f] - sse_full) / prep["sst"][f] if prep["sst"][f] > 0 else 0.0
    return gains


def _cell_permuted(cand: np.ndarray, cell_ids: np.ndarray, n_cells: int, rng: np.random.Generator) -> np.ndarray:
    """``cand`` replaced by its per-cell mean with the cell means permuted among the cells: a random function of the cell with the same cell-level values."""
    counts = np.bincount(cell_ids, minlength=n_cells)
    means = np.bincount(cell_ids, weights=cand, minlength=n_cells) / np.maximum(counts, 1)
    return means[rng.permutation(n_cells)][cell_ids]


def linear_usability_keeps(sources, cand_codes, y_cls, cell_ids, null_z: float, seed: int, n_perm: int = 32, n_screen: int = 4) -> bool:
    """Whether adding the candidate column improves an out-of-fold linear fit of ``y`` by more than its cell-permuted null.

    ``sources`` is the (n, s) float array of the raw group and agg values, ``cand_codes`` the candidate's numeric values or bin codes,
    ``y_cls`` the integer class labels, ``cell_ids`` the group column's cell of every row and ``null_z`` the family-wise z. Rows are
    subsampled to ``_MAX_ROWS`` with a generator seeded by ``seed``. A candidate that does not beat the largest of the first ``n_screen``
    null gains is dropped without computing the others. Returns False for a constant candidate or a target with more than
    ``_MAX_CLASSES`` classes.
    """
    rng = np.random.default_rng(int(seed))
    y_all = np.asarray(y_cls).ravel()
    n = y_all.shape[0]
    rows = np.sort(rng.choice(n, size=_MAX_ROWS, replace=False)) if n > _MAX_ROWS else np.arange(n)
    classes = np.unique(y_all[rows])
    if classes.size < 2 or classes.size > _MAX_CLASSES:
        return False
    cand = _standardised(_to_host(cand_codes)[rows])
    if not cand.any():
        return False
    _, cells = np.unique(np.asarray(cell_ids).ravel()[rows], return_inverse=True)
    n_cells = int(cells.max()) + 1
    if n_cells < 3:
        return False
    src = np.asarray(sources, dtype=np.float64)[rows]
    base = np.column_stack([np.ones(rows.size)] + [_standardised(src[:, j]) for j in range(src.shape[1])])
    Y = (y_all[rows][:, None] == classes[None, :]).astype(np.float64)
    folds = np.empty(rows.size, dtype=np.int64)
    folds[rng.permutation(rows.size)] = np.arange(rows.size) % _N_FOLDS
    prep = _prepare(base, Y, folds)
    cand_s, cells_s = cand[prep["order"]], cells[prep["order"]]
    observed = _gains(prep, cand_s)
    gain = float(observed.mean())
    if gain <= 0.0 or gain <= 2.0 * float(observed.std(ddof=1)) / np.sqrt(observed.size):
        return False
    null = []
    for i in range(int(n_perm)):
        null.append(_gains(prep, _cell_permuted(cand_s, cells_s, n_cells, rng)).mean())
        if i + 1 == int(n_screen) and gain <= max(null):
            return False
    null_arr = np.asarray(null)
    return gain > float(null_arr.mean() + float(null_z) * null_arr.std(ddof=1))
