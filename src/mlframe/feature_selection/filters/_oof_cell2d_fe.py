"""Out-of-fold 2-D cell table ``E[y | x_a, x_b]`` of a column pair, offered to the linear-downstream pool.

A linear model cannot form an arbitrary function of two columns (a bump, a ridge, a checkerboard). The table maps a pair onto the target's own scale: the (rank-scaled) target's mean in each
``K x K`` quantile cell, shrunk toward the ADDITIVE fit of the pair (row mean + column mean - grand mean), so a cell leaves additivity only on evidence. Cell values of a training row are
cross-fitted (the table fitted on the other folds); new data gets the full table. The pairs come from the residual screen (``_pair_residual_screen``): only pairs with a significant interaction
beyond the additive model are tabulated, so there is nothing to fit on additive or noise targets.

This operator has no MI-list stage: the factory runs showed ridge gains of 45-55% MAE on interaction targets but a small loss for gradient boosting, so the tables are offered to the usability pool
(``usability_aware_lists=True``) only, where the linear-downstream greedy keeps one if it improves the held-out linear fit. Replay (kind ``oof_cell2d``) stores the two edge vectors, the table,
the fill and the output range; it is a pure function of the two source columns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from .engineered_recipes import EngineeredRecipe

logger = logging.getLogger(__name__)

__all__ = ["fit_oof_cell2d", "apply_oof_cell2d_recipe", "build_oof_cell2d_recipe", "cell2d_pool_candidates"]

K_BINS = 10  # quantile bins per axis
SHRINK = 3.0  # pseudo-counts that pull a cell toward the additive fit of its row and column
N_FOLDS = 5
MAX_PAIRS = 3  # tables offered to the pool: bounds the extra cross-validated candidates
DEFAULT_SCAN_ROWS = 100_000


def _edges(x: np.ndarray, k: int = K_BINS) -> np.ndarray:
    """Unique interior equal-frequency edges of the finite values of ``x``."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.empty(0)
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, k + 1)[1:-1]))


def _cells(a: np.ndarray, b: np.ndarray, ea: np.ndarray, eb: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Cell indices of the rows (``-1`` where either value is not finite) and the cell count ``(len(ea) + 1) * (len(eb) + 1)``."""
    nb = len(eb) + 1
    ia = np.searchsorted(ea, np.where(np.isfinite(a), a, 0.0), side="right")
    ib = np.searchsorted(eb, np.where(np.isfinite(b), b, 0.0), side="right")
    cell = ia * nb + ib
    return np.where(np.isfinite(a) & np.isfinite(b), cell, -1), np.int64((len(ea) + 1) * nb)


def _table(sums: np.ndarray, counts: np.ndarray, mu: float, shrink: float) -> np.ndarray:
    """Shrunk cell means of a ``(rows, cols)`` sum / count table: the prior of a cell is ``row mean + column mean - grand mean`` (each mean shrunk toward ``mu``)."""
    rx = (sums.sum(1) + shrink * mu) / (counts.sum(1) + shrink)
    cz = (sums.sum(0) + shrink * mu) / (counts.sum(0) + shrink)
    prior = rx[:, None] + cz[None, :] - mu
    return (sums + shrink * prior) / (counts + shrink)


def fit_oof_cell2d(a: np.ndarray, b: np.ndarray, y: np.ndarray, *, n_folds: int = N_FOLDS, shrink: float = SHRINK, seed: int = 0) -> dict:
    """Fit the table of the pair ``(a, b)`` against the bounded target ``y`` and return ``{"oof", "ea", "eb", "tab", "fill"}``.

    ``oof`` is the cross-fitted value of every row (a row in fold ``f`` takes the table fitted on the other folds); ``tab`` is the full ``(len(ea)+1, len(eb)+1)`` table and ``fill`` the mean
    target, the value for a row with a non-finite coordinate. The edges depend on ``a`` and ``b`` only."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ea, eb = _edges(a), _edges(b)
    cell, n_cell = _cells(a, b, ea, eb)
    shape = (len(ea) + 1, len(eb) + 1)
    ok = (cell >= 0) & np.isfinite(y)
    fold = (np.random.default_rng(seed).permutation(len(a)) % int(n_folds)).astype(np.int64)
    flat = fold[ok] * n_cell + cell[ok]
    S = np.bincount(flat, weights=y[ok], minlength=n_folds * n_cell).reshape(n_folds, *shape)
    N = np.bincount(flat, minlength=n_folds * n_cell).reshape(n_folds, *shape).astype(np.float64)
    mu_all = float(y[ok].mean()) if ok.any() else 0.0
    tab_all = _table(S.sum(0), N.sum(0), mu_all, shrink)
    oof = np.full(len(a), mu_all)
    for f in range(n_folds):
        n_tr = float(N.sum() - N[f].sum())
        mu_f = float((S.sum() - S[f].sum()) / n_tr) if n_tr > 0 else mu_all
        tab_f = _table(S.sum(0) - S[f], N.sum(0) - N[f], mu_f, shrink)
        rows = (fold == f) & (cell >= 0)
        oof[rows] = tab_f.reshape(-1)[cell[rows]]
        oof[(fold == f) & (cell < 0)] = mu_f
    return {"oof": oof, "ea": ea, "eb": eb, "tab": tab_all, "fill": mu_all}


def build_oof_cell2d_recipe(*, name: str, src: Sequence[str], ea: np.ndarray, eb: np.ndarray, tab: np.ndarray, fill: float, lo: float, hi: float) -> "EngineeredRecipe":
    """Frozen recipe of one table column: the two edge vectors, the full table, the fill and the output range."""
    from .engineered_recipes import EngineeredRecipe

    return EngineeredRecipe(
        name=name,
        kind="oof_cell2d",
        src_names=tuple(str(c) for c in src),
        extra={
            "ea": np.asarray(ea, dtype=np.float64).copy(), "eb": np.asarray(eb, dtype=np.float64).copy(), "tab": np.asarray(tab, dtype=np.float64).copy(),
            "fill": float(fill), "lo": float(lo), "hi": float(hi),
        },
    )


def apply_oof_cell2d_recipe(recipe, X) -> np.ndarray:
    """Replay one table column from the stored table; a pure function of the two source columns."""
    from .engineered_recipes.shared import extract_column

    ex = recipe.extra
    a = np.asarray(extract_column(X, recipe.src_names[0]), dtype=np.float64)
    b = np.asarray(extract_column(X, recipe.src_names[1]), dtype=np.float64)
    cell, _ = _cells(a, b, ex["ea"], ex["eb"])
    out = np.where(cell >= 0, ex["tab"].reshape(-1)[np.maximum(cell, 0)], ex["fill"])
    return np.clip(out, ex["lo"], ex["hi"])


def cell2d_pool_candidates(df: Any, y_cont: np.ndarray, base_names: Sequence[str], feature_dtype: Any, quantization_nbins: int, *, scan_rows: int = DEFAULT_SCAN_ROWS) -> list:
    """``UsableCandidate`` objects for the tables of the pairs with a significant interaction beyond the additive model (empty on any failure or when no pair is significant)."""
    from scipy.stats import rankdata

    from ._mi_greedy_cmi_fe import _quantile_bin, precompute_marginal_y_terms
    from ._pair_residual_screen import pair_interaction_screen
    from ._usability_aware_selection import UsableCandidate, _binned_mi, _scrub

    try:
        pairs = pair_interaction_screen(df, y_cont, list(base_names), scan_rows=scan_rows)[:MAX_PAIRS]
        if not pairs:
            return []
        y_arr = np.asarray(y_cont, dtype=np.float64).ravel()
        n = len(y_arr)
        rows = np.arange(n) if n <= scan_rows else np.linspace(0, n - 1, int(scan_rows)).astype(np.int64)
        yr = rankdata(y_arr[rows], method="average") / float(len(rows))
        y_codes = _quantile_bin(y_cont, quantization_nbins, host_only=True)
        y_terms = precompute_marginal_y_terms(y_codes)
        out = []
        for ca, cb, _t in pairs:
            a = np.asarray(df[ca].to_numpy(), dtype=np.float64)
            b = np.asarray(df[cb].to_numpy(), dtype=np.float64)
            fit = fit_oof_cell2d(a[rows], b[rows], yr)
            cell, _ = _cells(a, b, fit["ea"], fit["eb"])
            col = np.where(cell >= 0, fit["tab"].reshape(-1)[np.maximum(cell, 0)], fit["fill"])
            col[rows] = fit["oof"]  # the fitting rows take their cross-fitted value, every other row the full table
            values = _scrub(col, feature_dtype)
            if float(np.std(values)) <= 1e-9:
                continue
            name = f"oofcell({ca},{cb})"
            rec = build_oof_cell2d_recipe(name=name, src=(ca, cb), ea=fit["ea"], eb=fit["eb"], tab=fit["tab"], fill=fit["fill"], lo=float(col.min()), hi=float(col.max()))
            out.append(UsableCandidate(name, values, _binned_mi(values, y_codes, quantization_nbins, y_terms), rec, (str(ca), str(cb)), ()))
        return out
    except Exception as e:  # an optional enrichment of the pool; never sink the usability pass
        logger.debug("cell2d usability candidates skipped: %s", e)
        return []
