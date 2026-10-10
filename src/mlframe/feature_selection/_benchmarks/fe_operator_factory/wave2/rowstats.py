"""Operator G: row statistics (min, max, median, range, std, mean, soft-max / soft-min log-sum-exp) over a learned column subset.

``fit_rowstat`` standardises the candidate columns (mean / sd frozen in the recipe); for every statistic it finds the subset that maximises the training MI of the statistic
(exhaustive over pairs, greedy growth while the MI improves by more than ``grow_margin / n``) on a row subsample (``scan_rows``) and returns the best (statistic, subset).
``replay_rowstat`` is a pure function of the recipe and the source columns.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..common.binning import mi_b, qbin

__all__ = ["STATS", "fit_rowstat", "replay_rowstat", "row_stat"]

LSE_BETA = 4.0
STATS = ("min", "max", "med", "rng", "std", "mean", "lse_pos", "lse_neg")


def row_stat(name: str, Z: np.ndarray) -> np.ndarray:
    """Statistic ``name`` of every row of ``Z`` (rows x selected columns); ``lse_pos`` / ``lse_neg`` are soft max / soft min via log-mean-exp with sharpness +-4."""
    if name == "min":
        return Z.min(1)
    if name == "max":
        return Z.max(1)
    if name == "med":
        return np.median(Z, 1)
    if name == "rng":
        return Z.max(1) - Z.min(1)
    if name == "std":
        return Z.std(1)
    if name == "mean":
        return Z.mean(1)
    b = LSE_BETA if name == "lse_pos" else -LSE_BETA
    t = b * Z
    mx = t.max(1, keepdims=True)
    return (mx[:, 0] + np.log(np.exp(t - mx).mean(1))) / b


def _z(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Standardise with frozen moments; NaN becomes 0 (the training mean)."""
    Z = (X - mu) / sd
    return np.where(np.isfinite(Z), Z, 0.0)


def fit_rowstat(X: np.ndarray, y: np.ndarray, src: list, scan_rows: int = 20000, grow_margin: float = 4.0, seed: int = 0) -> dict:
    """Learn (statistic, subset) from the candidate columns ``src`` (positions in ``X``); returns a recipe dict with ``stat``, ``src``, ``mu``, ``sd`` and the train MI."""
    A = np.asarray(X, float)[:, src]
    mu = np.nanmean(A, 0)
    sd = np.where(np.nanstd(A, 0) > 0, np.nanstd(A, 0), 1.0)
    Z = _z(A, mu, sd)
    n = len(y)
    sub = np.arange(n) if n <= scan_rows else np.random.default_rng(seed).choice(n, scan_rows, replace=False)
    Zs = Z[sub]
    yb = qbin(y[sub], 10)
    p = Zs.shape[1]
    ns = len(sub)

    def score(nm, S):
        """Train MI of statistic ``nm`` over the column subset ``S``."""
        return mi_b(qbin(row_stat(nm, Zs[:, list(S)]), 10), yb, 10, 10)

    best = (-1.0, None, None)
    for nm in STATS:
        m, S = max(((score(nm, S), list(S)) for S in itertools.combinations(range(p), 2)), key=lambda t: t[0])
        while len(S) < p:
            mj, j = max((score(nm, [*S, j]), j) for j in range(p) if j not in S)
            if mj > m + grow_margin / ns:
                S.append(j)
                m = mj
            else:
                break
        if m > best[0]:
            best = (m, nm, sorted(S))
    S = best[2]
    return {"kind": "row_stat", "src": [src[j] for j in S], "stat": best[1], "mu": mu[S].tolist(), "sd": sd[S].tolist(), "train_mi": float(best[0])}


def replay_rowstat(recipe: dict, X) -> np.ndarray:
    """Transform-time row statistic from the recipe and the source columns of ``X`` (DataFrame: by name, ndarray: by position)."""
    if hasattr(X, "columns"):
        A = np.column_stack([np.asarray(X[s], dtype=np.float64) for s in recipe["src"]])
    else:
        A = np.asarray(X, float)[:, recipe["src"]]
    out = row_stat(recipe["stat"], _z(A, np.asarray(recipe["mu"]), np.asarray(recipe["sd"])))
    return np.clip(out, -1e6, 1e6)
