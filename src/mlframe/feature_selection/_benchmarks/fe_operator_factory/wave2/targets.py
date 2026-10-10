"""Synthetic W / N / 0 targets (idealised, as in the brainstorm) and harder realistic W / N targets for operators B, C, G and the pair screen M.

Every generator is ``gen(rng, n) -> (X, y, truth)``; ``truth`` is the oracle feature (or None). ``HARD`` cases add irrelevant and correlated columns, a second weak signal and more noise.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CASES", "gen_M"]


def _noisy(s: np.ndarray, rng, nz: float) -> np.ndarray:
    """``s`` plus Gaussian noise at ``nz`` times the std of ``s``."""
    return s + nz * s.std() * rng.standard_normal(len(s))


def bumps(X: np.ndarray) -> np.ndarray:
    """Two Gaussian bumps on the unit square (columns 0 and 1)."""

    def f(c1, c2):
        """One bump centred at ``(c1, c2)``."""
        return np.exp(-((X[:, 0] - c1) ** 2 + (X[:, 1] - c2) ** 2) / 0.02)

    return f(0.25, 0.25) + f(0.75, 0.7)


def _mk(p, sig, truth=None, nz=0.3, null=False, corr3=False):
    """Case factory: ``p`` uniform columns (optionally column 3 = column 0 + noise), signal ``sig``, noise ``nz``, or pure-noise y when ``null``."""

    def gen(rng, n):
        """Draw one data set."""
        X = rng.random((n, p))
        if corr3:
            X[:, 3] = np.clip(X[:, 0] + 0.3 * rng.standard_normal(n), -1.5, 2.5)
        s = sig(X)
        y = rng.standard_normal(n) if null else _noisy(s, rng, nz)
        return X, y, (None if null else (truth or sig)(X))

    return gen


def _bump_hard(X):
    """B hard signal: bumps plus a weak linear term in column 2."""
    return bumps(X) + 0.4 * X[:, 2]


def _warp_hard(X):
    """C hard signal: three-period sine of column 0, a weak linear term and a weak quadratic."""
    return np.sin(9.4 * X[:, 0]) + 0.4 * X[:, 1] + 0.3 * (X[:, 2] - 0.5) ** 2


def _rng_hard(X):
    """G hard signal: range of the first five columns plus a weak linear term in column 5."""
    return X[:, :5].max(1) - X[:, :5].min(1) + 0.5 * X[:, 5]


CASES = {
    "B": {
        "W": (_mk(2, bumps), [0, 1]),
        "N": (_mk(2, lambda X: X[:, 0] + X[:, 1] ** 2), [0, 1]),
        "0": (_mk(2, bumps, null=True), [0, 1]),
        "H": (_mk(6, _bump_hard, nz=0.8, corr3=True), list(range(6))),
        "HN": (_mk(6, lambda X: X[:, 0] + X[:, 1] ** 2 + 0.4 * X[:, 2], nz=0.8, corr3=True), list(range(6))),
    },
    "C": {
        "W": (_mk(2, lambda X: np.sin(9.4 * X[:, 0])), [0, 1]),
        "N": (_mk(2, lambda X: 2 * X[:, 0]), [0, 1]),
        "0": (_mk(2, lambda X: X[:, 0], null=True), [0, 1]),
        "H": (_mk(6, _warp_hard, nz=0.8, corr3=True), list(range(6))),
        "HN": (_mk(6, lambda X: 2 * X[:, 0] + 0.4 * X[:, 1], nz=0.8, corr3=True), list(range(6))),
    },
    "G": {
        "W": (_mk(4, lambda X: X.max(1) - X.min(1)), [0, 1, 2, 3]),
        "N": (_mk(4, lambda X: X[:, 0] / (X[:, 1] + 0.1)), [0, 1, 2, 3]),
        "0": (_mk(4, lambda X: X.max(1), null=True), [0, 1, 2, 3]),
        "H": (_mk(10, _rng_hard, nz=0.8), list(range(10))),
        "HN": (_mk(10, lambda X: X[:, 0] / (X[:, 1] + 0.1) + 0.5 * X[:, 5], nz=0.8), list(range(10))),
    },
}


def gen_M(rng, n: int, kind: str, hard: bool = False) -> tuple:
    """Pair-screen data: a strong additive part on columns 0-3 plus, for ``W``, the interaction ``log(2 x4) sin(3 x5)``; ``N`` has none; ``0`` is pure-noise y.

    ``hard`` uses 10 columns (2 irrelevant), noise 0.6 of the signal std, an interaction carrying a smaller share of the variance and a second weak interaction ``x6 * x7`` that is also
    a true pair. Returns ``(X, y, true_pairs)``.
    """
    p = 10 if hard else 8
    X = rng.random((n, p))
    add = np.sin(6 * X[:, 0]) + 2 * X[:, 1] ** 2 + 1.5 * X[:, 2] - 3 * np.abs(X[:, 3] - 0.5)
    inter = np.zeros(n)
    pairs = []
    if kind == "W":
        inter = (1.0 if hard else 2.5) * np.log(2 * X[:, 4]) * np.sin(3 * X[:, 5])
        pairs.append((4, 5))
        if hard:
            inter = inter + 1.0 * (X[:, 6] - 0.5) * (X[:, 7] - 0.5) * 4
            pairs.append((6, 7))
    s = add + inter
    y = rng.standard_normal(n) if kind == "0" else _noisy(s, rng, 0.6 if hard else 0.2)
    return X, y, pairs
