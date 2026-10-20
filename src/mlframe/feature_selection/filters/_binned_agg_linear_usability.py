"""Linear-usability check for a binned-aggregate column that the MI redundancy gate rejected.

The redundancy gate scores ``CMI(col; y | qbin(group), qbin(agg))``. A binned-aggregate column is, by construction, a function of
the group column's cell, which is the same partition the gate conditions on, so the column is rejected as redundant whenever the
raw group already explains ``y`` in the tree-like, binned sense. A low-capacity downstream model cannot use that group directly: with
``aux ~ N(0, sigma(g))`` and ``y = sigma(g) + noise`` a ridge model on the raw columns scores held-out R^2 of 0.00 and 0.86 once the
cell-spread column is added, while a boosted-tree model scores 0.89 either way. This module measures that gap directly.

The gain is the out-of-fold increase in R^2 of a linear model of the class indicators of ``y`` on the raw sources (group and agg
values) when the candidate column is added. The acceptance bound is calibrated, not fixed: the same gain is recomputed with the
candidate column shuffled across rows (the candidate then keeps its marginal and loses its link to ``y``), and the column is kept
when its gain clears ``mean + z * std`` of that null, with ``z`` the family-wise z the caller already uses for the CMI null.
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


def _fold_gains(base: np.ndarray, cand: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Per-fold out-of-fold ``R^2(base + cand) - R^2(base)`` of the least-squares fit of ``Y`` (n x k), via per-fold normal equations."""
    full = np.column_stack([base, cand])
    n_folds = int(folds.max()) + 1
    stats = []
    for f in range(n_folds):
        mask = folds == f
        d_base, d_full, y_f = base[mask], full[mask], Y[mask]
        mean_f = y_f.mean(axis=0)
        stats.append((d_base.T @ d_base, d_base.T @ y_f, d_full.T @ d_full, d_full.T @ y_f, float((y_f * y_f).sum()), float(((y_f - mean_f) ** 2).sum())))
    total = [sum(s[i] for s in stats) for i in range(4)]
    gains = np.empty(n_folds, dtype=np.float64)
    for f, (g_b, b_b, g_f, b_f, yy, sst) in enumerate(stats):
        sse = []
        for g_tot, b_tot, g_te, b_te in ((total[0], total[1], g_b, b_b), (total[2], total[3], g_f, b_f)):
            g_tr, b_tr = g_tot - g_te, b_tot - b_te
            beta = np.linalg.solve(g_tr + _RIDGE * np.trace(g_tr) * np.eye(g_tr.shape[0]), b_tr)
            sse.append(yy - 2.0 * float((beta * b_te).sum()) + float((beta * (g_te @ beta)).sum()))
        gains[f] = (sse[0] - sse[1]) / sst if sst > 0 else 0.0
    return gains


def linear_usability_keeps(sources, cand_codes, y_cls, null_z: float, seed: int, n_perm: int = 16) -> bool:
    """Whether adding the candidate column improves an out-of-fold linear fit of ``y`` by more than its own shuffled-candidate null.

    ``sources`` is the (n, s) float array of the raw group and agg values, ``cand_codes`` the candidate's numeric values or bin codes,
    ``y_cls`` the integer class labels and ``null_z`` the family-wise z. Rows are subsampled to ``_MAX_ROWS`` with a generator seeded
    by ``seed``. Returns False for a constant candidate or a target with more than ``_MAX_CLASSES`` classes.
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
    src = np.asarray(sources, dtype=np.float64)[rows]
    base = np.column_stack([np.ones(rows.size)] + [_standardised(src[:, j]) for j in range(src.shape[1])])
    Y = (y_all[rows][:, None] == classes[None, :]).astype(np.float64)
    folds = np.empty(rows.size, dtype=np.int64)
    folds[rng.permutation(rows.size)] = np.arange(rows.size) % _N_FOLDS
    observed = _fold_gains(base, cand, Y, folds)
    gain = float(observed.mean())
    if gain <= 0.0 or gain <= 2.0 * float(observed.std(ddof=1)) / np.sqrt(observed.size):
        return False
    null = np.array([_fold_gains(base, cand[rng.permutation(rows.size)], Y, folds).mean() for _ in range(int(n_perm))])
    return gain > float(null.mean() + float(null_z) * null.std(ddof=1))
