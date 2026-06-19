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


_NOMINAL_MAX_CLASSES = 64  # at/below this many distinct y values, treat y as discrete (one-hot, relabel-invariant)


def second_moment_propensity(values: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interaction-propensity score per column of ``values`` (n, p): higher moments of x leak when the
    linear marginal is flat.

    ``values`` may be raw floats or quantile bin-codes (a monotone transform preserves the even/odd
    higher-moment structure the score exploits). The treatment of ``y`` depends on its cardinality:

    * DISCRETE y (<= ``_NOMINAL_MAX_CLASSES`` distinct values -- the synergy-bootstrap site always passes the
      DISCRETISED target, so this is the live path): the score is RELABEL-INVARIANT -- one-hot y and sum
      ``|corr(x^2, 1[y=c])| + |corr(x, 1[y=c])|`` over classes. Squaring the raw integer class CODES would be
      meaningless for a NOMINAL multiclass target (the kept set would depend on the arbitrary label assigned
      to each class -- a real bug found 2026-06-19); one-hotting fixes it. For a BINARY target this reduces to
      twice the original ``|corr(x^2,y)|+|corr(x,y^2)|`` (the two class indicators are complementary), so the
      RANKING -- all that matters for top-k -- is identical to the benched binary contract.
    * CONTINUOUS y (> ``_NOMINAL_MAX_CLASSES`` distinct values -- a genuine regression target on raw values):
      the moment form ``|corr(x^2, y)| + |corr(x, y^2)|``.

    Works for EVERY target type: binary, nominal/ordinal multiclass, continuous regression, boolean, and
    NON-NUMERIC (string / object / pandas Categorical) class labels.

    NB on the MRMR synergy-bootstrap path the target arrives ALREADY ordinal-encoded -- ``categorize_dataset``
    (filters/discretization) factorises every column, target included, before the FE step -- so the wiring
    always passes integer codes and the non-numeric branch below is a NO-OP there. The defensive factorise is
    for DIRECT callers that score raw ``y`` before categorisation (e.g. the planned SIS front gate, which must
    rank raw columns before discretising a 100k-wide frame); it is not a re-implementation of categorize.

    O(K*n*p) for discrete (K = n_classes, small) / O(n*p) for continuous; fully vectorised."""
    V = np.ascontiguousarray(values, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"values must be 2-D (n, p); got shape {V.shape}")
    y_arr = np.asarray(y).ravel() if not hasattr(y, "to_numpy") else np.asarray(y.to_numpy()).ravel()
    if y_arr.shape[0] != V.shape[0]:
        raise ValueError(f"y length {y_arr.shape[0]} != n_rows {V.shape[0]}")
    # Non-numeric labels (str / object / bool / Categorical) -> classification: factorise to codes (the discrete
    # one-hot path below is relabel-invariant, so the integer assignment is irrelevant). Numeric y passes through.
    if y_arr.dtype.kind in "USO" or y_arr.dtype == bool:
        _, y_arr = np.unique(y_arr, return_inverse=True)
    yf = np.asarray(y_arr, dtype=np.float64)
    yf = np.nan_to_num(yf, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    V2 = V * V
    classes = np.unique(yf)
    if classes.size > _NOMINAL_MAX_CLASSES:
        return _abs_col_corr(V2, yf) + _abs_col_corr(V, yf * yf)
    # DISCRETE / nominal: relabel-invariant one-hot sum (correct for binary, nominal multiclass, binned y).
    # The per-column standardization of V / V2 is class-INDEPENDENT, so it is hoisted out of the (former)
    # K-class loop and done ONCE; the K classes then reduce to a single (p,n)@(n,K) GEMM per matrix.
    # compute_discrete_score dispatches that GEMM to the fastest backend (numpy/numba/cupy) by work size,
    # with the numpy GEMM as the bit-reference fallback. See _discrete_score_numpy_loop for the original.
    if classes.size <= 1:
        # degenerate single-class target: every indicator is constant -> corr undefined -> zero score
        # (matches the reference: a constant indicator has den==0 -> 0).
        return np.zeros(V.shape[1], dtype=np.float64)
    from mlframe.feature_selection.filters._fe_interaction_prerank_kernels import compute_discrete_score

    return compute_discrete_score(V, V2, yf, classes)


def _discrete_score_numpy_loop(V: np.ndarray, V2: np.ndarray, yf: np.ndarray,
                               classes: np.ndarray) -> np.ndarray:
    """Original per-class-loop reference for the discrete path (kept for parity tests / fallback).

    Recomputes V/V2 column stats once per class; semantically identical to the hoisted GEMM dispatcher
    but K x more BLAS calls. Retained as the named, dependency-free correctness baseline."""
    score = np.zeros(V.shape[1], dtype=np.float64)
    for c in classes:
        ind = (yf == c).astype(np.float64)
        score += _abs_col_corr(V2, ind) + _abs_col_corr(V, ind)
    return score


def gbm_split_propensity(values: np.ndarray, y: np.ndarray, num_boost_round: int = 100) -> np.ndarray:
    """LightGBM split-frequency importance per column -- a COMPLEMENTARY interaction-propensity signal.

    A pure-interaction operand carries ~0 marginal MI but a tree booster still SPLITS on it repeatedly
    once a partner operand has been split (the interaction surfaces in the conditional structure), so its
    split count is elevated even when its marginal correlation is flat. This is the criterion the H2 bench
    found scored highest in isolation (recall ~0.92 at L=0.1) but at ~18x the cost of the 2nd-moment score
    (one full booster fit). Used here only as one ingredient of the rank-fused criterion.

    Returns a length-p float array (split count per column); zeros if LightGBM is unavailable."""
    try:
        import lightgbm as lgb
    except Exception:
        return np.zeros(values.shape[1], dtype=np.float64)
    X = np.ascontiguousarray(values, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.asarray(y).ravel() if not hasattr(y, "to_numpy") else np.asarray(y.to_numpy()).ravel()
    if y_arr.dtype.kind in "USO" or y_arr.dtype == bool:
        _, y_arr = np.unique(y_arr, return_inverse=True)
    yf = np.nan_to_num(np.asarray(y_arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n_classes = int(np.unique(yf).size)
    if n_classes <= 1:
        return np.zeros(X.shape[1], dtype=np.float64)
    if n_classes <= _NOMINAL_MAX_CLASSES:
        codes = np.unique(yf, return_inverse=True)[1]
        if n_classes == 2:
            params = dict(objective="binary", num_leaves=31, learning_rate=0.1,
                          verbose=-1, min_child_samples=20, feature_fraction=1.0)
        else:
            params = dict(objective="multiclass", num_class=n_classes, num_leaves=31,
                          learning_rate=0.1, verbose=-1, min_child_samples=20, feature_fraction=1.0)
        label = codes.astype(np.float64)
    else:  # continuous regression target
        params = dict(objective="regression", num_leaves=31, learning_rate=0.1,
                      verbose=-1, min_child_samples=20, feature_fraction=1.0)
        label = yf
    try:
        ds = lgb.Dataset(X, label=label)
        booster = lgb.train(params, ds, num_boost_round=num_boost_round)
        return booster.feature_importance(importance_type="split").astype(np.float64)
    except Exception:
        return np.zeros(X.shape[1], dtype=np.float64)


def _rank_desc(scores: np.ndarray) -> np.ndarray:
    """Competition-free dense ranks (0 = best) for descending ``scores``; ties get the same average-free
    ordinal rank by stable argsort. Lower rank = more interesting -- the common scale for rank fusion."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.shape[0], dtype=np.float64)
    ranks[order] = np.arange(scores.shape[0], dtype=np.float64)
    return ranks


def fused_propensity(values: np.ndarray, y: np.ndarray, use_gbm: bool = True) -> np.ndarray:
    """Rank-fused interaction-propensity: combine the cheap 2nd-moment score with complementary signals so
    operands that EITHER leak in higher moments OR split frequently in a tree get surfaced.

    Fusion is by MIN-RANK (an operand kept high by ANY ingredient survives -- W3's prototype noted this
    helped over a single criterion): score = -(min over ingredients of the descending rank). Ingredients:
      * 2nd-moment propensity (always; the O(p*n) base),
      * marginal MI proxy via |corr(x, indicator)| summed over classes (cheap main-effect channel),
      * gbm split-frequency (when ``use_gbm`` and LightGBM is importable; one booster fit, the strong
        complementary interaction signal).
    Higher returned score = more interesting (so it plugs into the same descending-sort selection)."""
    sm = second_moment_propensity(values, y)
    # cheap main-effect channel: |corr(x, 1[y=c])| summed over classes (reuses the standardized machinery).
    V = np.nan_to_num(np.ascontiguousarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.asarray(y).ravel() if not hasattr(y, "to_numpy") else np.asarray(y.to_numpy()).ravel()
    if y_arr.dtype.kind in "USO" or y_arr.dtype == bool:
        _, y_arr = np.unique(y_arr, return_inverse=True)
    yf = np.nan_to_num(np.asarray(y_arr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    classes = np.unique(yf)
    main = np.zeros(V.shape[1], dtype=np.float64)
    if classes.size > 1 and classes.size <= _NOMINAL_MAX_CLASSES:
        for c in classes:
            main += _abs_col_corr(V, (yf == c).astype(np.float64))
    else:
        main = _abs_col_corr(V, yf)
    ingredients = [_rank_desc(sm), _rank_desc(main)]
    if use_gbm:
        gbm = gbm_split_propensity(values, y)
        if np.any(gbm > 0):
            ingredients.append(_rank_desc(gbm))
    min_rank = np.min(np.vstack(ingredients), axis=0)
    return -min_rank


_CRITERIA = {
    "second_moment": second_moment_propensity,
    "fused": fused_propensity,
    "gbm_splits": gbm_split_propensity,
}


def top_k_by_interaction_propensity(
    values: np.ndarray, y: np.ndarray, candidate_idx: Any, top_k: int,
    criterion: str = "second_moment",
) -> list[int]:
    """Rank ``candidate_idx`` (column indices into ``values``) by ``second_moment_propensity`` and return
    the top ``top_k`` as a SORTED list of indices (deterministic; ties broken by ascending index so the
    result is stable across runs). If ``top_k`` >= len(candidate_idx) all candidates are returned sorted.

    ``values`` is the full (n, n_cols) matrix; only the candidate columns are scored.

    ``criterion`` selects the ranking score: "second_moment" (default, O(p*n), the benched base),
    "fused" (rank-fusion of 2nd-moment + marginal + gbm split-frequency -- higher recall at L<=0.1 when
    LightGBM is available, ~one extra booster fit), or "gbm_splits" (split-frequency alone)."""
    cand = sorted(int(i) for i in candidate_idx)
    if top_k >= len(cand):
        return cand
    if top_k <= 0:
        return []
    sub = values[:, cand]
    score_fn = _CRITERIA.get(criterion, second_moment_propensity)
    scores = score_fn(sub, y)
    # argsort descending by score, then ascending by position (stable) -> deterministic top_k.
    order = np.lexsort((np.arange(len(cand)), -scores))[:top_k]
    return sorted(cand[i] for i in order)


__all__ = ["second_moment_propensity", "top_k_by_interaction_propensity",
           "fused_propensity", "gbm_split_propensity"]
