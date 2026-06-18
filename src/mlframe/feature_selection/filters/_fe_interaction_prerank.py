"""O(p*n) interaction-propensity pre-rank for the MRMR-FE synergy pool on WIDE frames.

The synergy bootstrap (``_mrmr_fe_step_helpers.apply_synergy_bootstrap``) seeds the all-pairs joint-MI
sweep with the raw numeric columns so PURE-interaction pairs (a*b, sign products, log(c)*sin(d)) -- whose
operands carry ~0 MARGINAL MI and so never screen in individually -- get joint-MI screened. That sweep is
O(p^2) and is hard-capped at ``fe_synergy_screen_max_features`` (default 250): historically, above the cap
the bootstrap simply SKIPPED, so on a wide frame (p >> 250) a zero-marginal interaction was engineered as
NOTHING. The cap can't be raised blindly -- a full exhaustive sweep at p=10k is ~17 min on a GTX 1050 Ti
(bench 2026-06-18) -- so we need to choose WHICH ~250 columns enter the sweep.

Marginal MI is the WRONG ranking for this: a pure-interaction operand has ~0 marginal MI by construction
(that is the whole reason the bootstrap exists), so ranking the pool by marginal MI drops exactly the
operands we are hunting. The fix is an interaction-propensity score that detects a variable's propensity to
participate in ANY interaction even when its linear marginal is zero -- the classic interaction-screening
idea (Fan-Kong-Li-Zhao 2015 "innovated interaction screening"; Hao-Zhang 2014): an interaction leaks into
HIGHER MOMENTS of (x, y) even when the first-moment marginal is flat. The cheap, vectorised proxy here is

    score(x_j) = |corr(x_j^2, y)| + |corr(x_j, y^2)|              ("second_moment")

which the H2 benchmark (n=8000, p=2000, 5 seeds, K=6 planted pure pair interactions) showed recovers the
true operands into the top-250 at recall ~0.88 at realistic leakage L=0.1 (vs marginal-MI 0.68 and the m/p
random baseline 0.12), at ~5s for p=10000 -- 18x cheaper than the LightGBM split-frequency criterion that
scored marginally higher. cond_resp_var (a first-moment bin-mean statistic) gave NO lift over marginal MI;
distance correlation underperformed (it captures monotone dependence, not interaction leakage).

IRREDUCIBLE FLOOR (do not paper over it): a PERFECTLY balanced zero-higher-moment interaction (exact 50/50
XOR, operands independent of y in every moment) is information-theoretically invisible to ANY O(p)
per-variable score -- at L=0.0 every criterion sits at the random baseline. That measure-zero case can only
be recovered by the exhaustive O(p^2) sweep itself; the pre-rank does not claim to find it. For realistic
interactions with any higher-moment leakage (L>=0.1) the pre-rank lets the needle SURVIVE into the sweep
where the old "skip past the cap" dropped it entirely.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _abs_col_corr(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised |Pearson corr| between every column of ``M`` (n, p) and the vector ``v`` (n,).

    Returns a length-p array; a constant column (zero variance) scores 0 (its corr is undefined)."""
    Mc = M - M.mean(axis=0)
    vc = v - v.mean()
    num = Mc.T @ vc                                   # (p,)
    den = np.sqrt((Mc * Mc).sum(axis=0) * float(vc @ vc))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 0.0, num / den, 0.0)
    return np.abs(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0))


def second_moment_propensity(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interaction-propensity score ``|corr(x^2, y)| + |corr(x, y^2)|`` per column of ``values`` (n, p).

    ``values`` may be raw floats or quantile bin-codes (a monotone transform preserves the even/odd
    higher-moment structure the score exploits). ``y`` is encoded to a float vector first (class codes for
    a classification target, raw values for regression). O(n*p), fully vectorised."""
    V = np.ascontiguousarray(values, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"values must be 2-D (n, p); got shape {V.shape}")
    yf = np.asarray(y, dtype=np.float64).ravel()
    if yf.shape[0] != V.shape[0]:
        raise ValueError(f"y length {yf.shape[0]} != n_rows {V.shape[0]}")
    yf = np.nan_to_num(yf, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    return _abs_col_corr(V * V, yf) + _abs_col_corr(V, yf * yf)


def top_k_by_interaction_propensity(
    values: np.ndarray, y: np.ndarray, candidate_idx: Any, top_k: int,
) -> list[int]:
    """Rank ``candidate_idx`` (column indices into ``values``) by ``second_moment_propensity`` and return
    the top ``top_k`` as a SORTED list of indices (deterministic; ties broken by ascending index so the
    result is stable across runs). If ``top_k`` >= len(candidate_idx) all candidates are returned sorted.

    ``values`` is the full (n, n_cols) matrix; only the candidate columns are scored."""
    cand = sorted(int(i) for i in candidate_idx)
    if top_k >= len(cand):
        return cand
    if top_k <= 0:
        return []
    sub = values[:, cand]
    scores = second_moment_propensity(sub, y)
    # argsort descending by score, then ascending by position (stable) -> deterministic top_k.
    order = np.lexsort((np.arange(len(cand)), -scores))[:top_k]
    return sorted(cand[i] for i in order)


__all__ = ["second_moment_propensity", "top_k_by_interaction_propensity"]
