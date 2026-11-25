"""FRESH fixtures for the target-transform recommender (agent B3, 2026-11-24): none of them reuses a wave-1 generator, coefficient set, noise level or feature design.

Differences from wave 1 (standard-normal independent features, fixed coefficients 0.9/0.6 or 1/0.7, 8 columns, one noise level):
* features are CORRELATED (block factors: 0.7 inside a block of 4 columns, 0.2 across blocks); ``mixed`` mode adds binary and skewed columns;
* coefficients are drawn per (fixture, seed): random active columns, random signs, magnitudes U(0.4, 1.0), the linear predictor is standardised;
* nonlinear mean (``log_comp``: log y = a x_a + b clip(x_b, -2, 2)^2 + noise, clipped so that the variance of y is finite; ``lin_quad``), a transform AND heteroscedastic noise at once (``hetero_*``), a skewed-noise control (``lin_skewnoise``),
  an atom at zero (``zero_infl``), a right transform that is not offered (``neg_skew``, ``exp_add``), heavy tails (``lin_t3``);
* p is a parameter (5 and 40 are used), n in {2000, 20000}.

``KIND`` says whether identity is the intended answer (``control``) or a transform helps (``transform``); the ORACLE (tt_a4 protocol, see tt_b3_judge) is the final authority for ``transform`` kinds
whose best candidate is not obvious (``exp_add``, ``neg_skew``, ``zero_infl``), so those carry ``ambiguous``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

KIND: Dict[str, str] = {
    "lognormal": "transform", "log_comp": "transform", "gamma_log": "transform", "cube": "transform", "sqrt_pos": "transform", "recip": "transform", "logit_bnd": "transform",
    "hetero_log": "transform", "hetero_cube": "transform", "mixed_log": "transform",
    "exp_add": "ambiguous", "zero_infl": "ambiguous", "neg_skew": "ambiguous",
    "lin": "control", "lin_hetero": "control", "lin_mixed": "control", "lin_quad": "control", "lin_t3": "control", "lin_skewnoise": "control", "lin_pos": "control",
}
FIXTURES = tuple(KIND)
EXPECTED = {
    "lognormal": "log", "log_comp": "log", "gamma_log": "log", "cube": "cbrt", "sqrt_pos": "signed_sqrt", "recip": "reciprocal", "logit_bnd": "logit", "hetero_log": "log", "hetero_cube": "cbrt",
    "mixed_log": "log",
}
NONLINEAR_MEAN = ("log_comp", "lin_quad")


def _features(rng: np.random.Generator, n: int, p: int, mixed: bool) -> np.ndarray:
    """Correlated columns; ``mixed`` turns every third column binary and every fifth column right-skewed."""
    nb = (p + 3) // 4
    g = rng.standard_normal((n, 1))
    blocks = rng.standard_normal((n, nb))
    e = rng.standard_normal((n, p))
    Z = np.sqrt(0.2) * g + np.sqrt(0.5) * blocks[:, np.arange(p) // 4] + np.sqrt(0.3) * e
    if mixed:
        for j in range(p):
            if j % 3 == 2:
                Z[:, j] = (Z[:, j] > np.quantile(Z[:, j], rng.uniform(0.3, 0.7))).astype(np.float64)
            elif j % 5 == 4:
                Z[:, j] = np.exp(0.7 * Z[:, j])
    return np.asarray(Z)


def make(kind: str, n: int, p: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Features ``X (n, p)`` and target ``y`` of the fixture ``kind``; deterministic for ``(kind, n, p, seed)``."""
    rng = np.random.default_rng([seed, sum(ord(c) * (i + 1) for i, c in enumerate(kind)), p, n])
    mixed = kind in ("lin_mixed", "mixed_log")
    X = _features(rng, n, p, mixed)
    k_act = min(p, 3 if p <= 6 else 5)
    cols = rng.permutation(p)
    if mixed:
        binary = [j for j in range(p) if j % 3 == 2]
        skew = [j for j in range(p) if j % 3 != 2 and j % 5 == 4]
        forced = [binary[0]] + ([skew[0]] if skew else [])
        cols = np.array(forced + [c for c in cols if c not in forced])
    act = cols[:k_act]
    beta = rng.choice([-1.0, 1.0], k_act) * rng.uniform(0.4, 1.0, k_act)
    eta = X[:, act] @ beta
    eta = (eta - eta.mean()) / eta.std()
    xs = (X - X.mean(0)) / X.std(0)
    xa, xb = xs[:, act[0]], xs[:, act[min(1, k_act - 1)]]
    cont = [j for j in act if np.unique(X[:, j]).size > 10]
    xc = xs[:, cont[0]] if cont else xa
    e = rng.standard_normal(n)
    if kind == "lognormal":
        y = np.exp(0.8 * eta + 0.35 * e)
    elif kind == "mixed_log":
        y = np.exp(0.8 * eta + 0.35 * e)
    elif kind == "log_comp":
        y = np.exp(0.7 * xa + 0.35 * np.clip(xb, -2.0, 2.0) ** 2 + 0.3 * e)
    elif kind == "gamma_log":
        y = rng.gamma(5.0, np.exp(0.6 * eta) / 5.0)
    elif kind == "cube":
        y = (1.1 * eta + 0.3 * e) ** 3
    elif kind == "sqrt_pos":
        y = (5.0 + eta + 0.4 * e) ** 2
    elif kind == "recip":
        y = 1.0 / (4.0 + 0.45 * eta + 0.08 * e)
    elif kind == "logit_bnd":
        y = 1.0 / (1.0 + np.exp(-(1.3 * eta + 0.5 * e)))
    elif kind == "exp_add":
        y = np.exp(0.7 * eta) + 0.2 * e
    elif kind == "hetero_log":
        y = np.exp(0.8 * eta + 0.3 * (1.0 + np.abs(xc)) * e)
    elif kind == "hetero_cube":
        y = (1.1 * eta + 0.25 * (1.0 + 0.8 * np.abs(xc)) * e) ** 3
    elif kind == "zero_infl":
        y = np.where(rng.random(n) < 0.25, 0.0, np.exp(0.6 * eta + 0.4 * e))
    elif kind == "neg_skew":
        y = 6.0 - np.exp(0.7 * eta + 0.3 * e)
    elif kind in ("lin", "lin_mixed"):
        y = 2.0 + 1.5 * eta + 0.7 * e
    elif kind == "lin_hetero":
        y = 1.5 * eta + 0.5 * (1.0 + np.abs(xc)) * e
    elif kind == "lin_quad":
        y = eta + 0.7 * xb**2 + 0.5 * e
    elif kind == "lin_t3":
        y = 1.5 * eta + 0.5 * rng.standard_t(3, n)
    elif kind == "lin_skewnoise":
        y = 8.0 + 1.5 * eta + 1.5 * (rng.gamma(2.0, 1.0, n) - 2.0)
    elif kind == "lin_pos":
        y = 25.0 + 2.0 * eta + e
    else:
        raise ValueError(kind)
    return X, y
