"""Planted-signal generators for deciding which FE families may run on a smaller column pool (see ``audits/2026-11-09/fe_candidate_pool_plan.md``).

Each generator returns a frame of ``p`` columns (the planted ones named ``sig_*``, every other column ``noise_*``), a binary target, and the names of the planted columns. All are
deterministic in ``seed``. The planted effects are the ones a relevance filter on marginal binned MI would lose:

* :func:`make_weak_marginal_shape`: a U / window shape of one column, near-zero linear correlation and a marginal MI that the ``strength`` knob moves from invisible to clear.
* :func:`make_hidden_periodic`: ``sin(2 pi f x)`` with ``f`` a whole number of periods per decile bin, so that 10-bin quantile MI of ``x`` is about zero.
* :func:`make_grouped_by_irrelevant_key`: the target depends on ``x - mean_by_key(x)``; the integer key has no marginal effect.
* :func:`make_ratio_irrelevant_normaliser`: the target depends on ``a / b`` with ``b`` irrelevant alone.
* :func:`make_conditional_hinge`: a hinge of ``x`` whose direction is set by the sign of ``z``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(rng: np.random.Generator, n: int, p: int, planted: "dict[str, np.ndarray]") -> "tuple[pd.DataFrame, list[str]]":
    """``planted`` columns first, then ``p - len(planted)`` standard-normal noise columns."""
    cols: dict = dict(planted)
    for j in range(p - len(planted)):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    return pd.DataFrame(cols), list(planted)


def _binary(rng: np.random.Generator, score: np.ndarray, noise: float) -> np.ndarray:
    """Binary target: ``score`` plus Gaussian noise (of ``noise`` times the score's spread) above the median."""
    s = score + noise * float(np.std(score)) * rng.standard_normal(score.shape[0])
    return (s > np.median(s)).astype(np.int8)


def make_weak_marginal_shape(n: int = 20000, p: int = 100, seed: int = 0, strength: float = 1.0, shape: str = "u") -> "tuple[pd.DataFrame, np.ndarray, list[str]]":
    """Target depends on a U (``x**2``) or window (``|x| < 0.7``) shape of ``sig_x``; ``strength`` scales the signal against unit noise."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    score = x**2 if shape == "u" else (np.abs(x) < 0.7).astype(float)
    y = _binary(rng, strength * (score - np.mean(score)) / np.std(score), 1.0)
    X, planted = _frame(rng, n, p, {"sig_x": x})
    return X, y, planted


def make_hidden_periodic(n: int = 20000, p: int = 100, seed: int = 0, freq: float = 20.0, noise: float = 0.5) -> "tuple[pd.DataFrame, np.ndarray, list[str]]":
    """Target depends on ``sin(2 pi freq x)`` of ``sig_x``; at 20 (two full periods per decile bin of a uniform column) the 10-bin quantile MI of ``x`` is about zero."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = _binary(rng, np.sin(2.0 * np.pi * freq * x), noise)
    X, planted = _frame(rng, n, p, {"sig_x": x})
    return X, y, planted


def make_grouped_by_irrelevant_key(n: int = 20000, p: int = 100, seed: int = 0, n_groups: int = 20, noise: float = 0.5) -> "tuple[pd.DataFrame, np.ndarray, list[str]]":
    """Target depends on ``sig_x - mean of sig_x within sig_key``; the group means differ widely, so ``sig_x`` alone is dominated by the key and ``sig_key`` carries no marginal effect."""
    rng = np.random.default_rng(seed)
    key = rng.integers(0, n_groups, n)
    means = 3.0 * rng.standard_normal(n_groups)
    x = means[key] + rng.standard_normal(n)
    centred = x - pd.Series(x).groupby(key).transform("mean").to_numpy()
    y = _binary(rng, centred, noise)
    X, planted = _frame(rng, n, p, {"sig_x": x, "sig_key": key.astype(np.float32)})
    return X, y, planted


def make_ratio_irrelevant_normaliser(n: int = 20000, p: int = 100, seed: int = 0, noise: float = 0.4) -> "tuple[pd.DataFrame, np.ndarray, list[str]]":
    """Target depends on ``sig_a / sig_b`` (``sig_b`` positive, spread over a decade, no marginal effect of its own)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    b = np.exp(rng.uniform(-1.2, 1.2, n))
    y = _binary(rng, a / b, noise)
    X, planted = _frame(rng, n, p, {"sig_a": a, "sig_b": b})
    return X, y, planted


def make_conditional_hinge(n: int = 20000, p: int = 100, seed: int = 0, noise: float = 0.5) -> "tuple[pd.DataFrame, np.ndarray, list[str]]":
    """Target depends on ``max(sig_x, 0)`` where ``sig_z > 0`` and on ``max(-sig_x, 0)`` elsewhere (equal means, so ``sig_z`` has no marginal effect; ``sig_x`` only a symmetric ``|x|`` one)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = _binary(rng, np.where(z > 0.0, np.maximum(x, 0.0), np.maximum(-x, 0.0)), noise)
    X, planted = _frame(rng, n, p, {"sig_x": x, "sig_z": z})
    return X, y, planted


GENERATORS = {
    "weak_marginal_shape": make_weak_marginal_shape,
    "hidden_periodic": make_hidden_periodic,
    "grouped_by_irrelevant_key": make_grouped_by_irrelevant_key,
    "ratio_irrelevant_normaliser": make_ratio_irrelevant_normaliser,
    "conditional_hinge": make_conditional_hinge,
}
