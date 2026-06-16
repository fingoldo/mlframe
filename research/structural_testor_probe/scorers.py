"""Pure-numpy scorers for the Structural-Testor go/no-go probe.

Throwaway research code (see research/Structural_Testor_Selector_MLFrame.md and the plan).
Two value questions:
  A) rare_pair_coverage  -- is it just (distance-weighted) ReliefF?  [kill-test]
  B) separation_profile  -- a NEW feature-redundancy metric for mRMR.  [lead bet]

Design notes:
  * At probe scale (n<=~2000, F<=~500) we materialize the pair-difference matrix
    D (P x F).  P=2e4, F=500 -> ~40MB float32 -- trivial.  Streaming/closed-form
    (O(F N log N)) is a *production* concern, not needed to decide the gate.
  * All normalizers are fit on the data passed in; the probe driver is responsible
    for keeping train/test separation when it evaluates downstream models.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# pair sampling + difference matrix                                           #
# --------------------------------------------------------------------------- #
def sample_cross_class_pairs(y: np.ndarray, max_pairs: int, rng: np.random.Generator):
    """Rejection-sample cross-class index pairs -- O(max_pairs), never the N^2/2 pool.

    Draws i,j uniformly and keeps pairs with y[i] != y[j].  For heavy imbalance the
    acceptance rate stays >= 2*p*(1-p); fine for the probe's balanced-ish data.
    """
    n = len(y)
    ia = np.empty(max_pairs, dtype=np.int64)
    ib = np.empty(max_pairs, dtype=np.int64)
    filled = 0
    while filled < max_pairs:
        need = max_pairs - filled
        # oversample to amortise rejections
        i = rng.integers(0, n, size=need * 2)
        j = rng.integers(0, n, size=need * 2)
        ok = y[i] != y[j]
        i, j = i[ok], j[ok]
        take = min(len(i), need)
        ia[filled:filled + take] = i[:take]
        ib[filled:filled + take] = j[:take]
        filled += take
    return ia, ib


def build_difference_matrix(X: np.ndarray, ia, ib, kinds, stds) -> np.ndarray:
    """D[p, f] = separation strength of feature f on pair p.

    kinds[f] == 'num'  -> |x_i - x_j| / std[f]   (std precomputed once)
    kinds[f] == 'cat'  -> 1.0 if codes differ else 0.0
    Returns float32 (P, F).
    """
    Xi = X[ia]
    Xj = X[ib]
    D = np.empty(Xi.shape, dtype=np.float32)
    num = np.asarray([k == "num" for k in kinds])
    if num.any():
        s = stds[num]
        s = np.where(s > 0, s, 1.0)
        D[:, num] = (np.abs(Xi[:, num] - Xj[:, num]) / s).astype(np.float32)
    if (~num).any():
        D[:, ~num] = (Xi[:, ~num] != Xj[:, ~num]).astype(np.float32)
    return D


def _rank_normalize_columns(D: np.ndarray) -> np.ndarray:
    """Map each column to its in-column rank in [0,1] so unbounded numeric |Δ|/std
    and {0,1} categorical contribute comparably to S_p.  Average-rank ties handling
    is unnecessary here (we only need scale comparability)."""
    P = D.shape[0]
    order = np.argsort(D, axis=0, kind="stable")
    ranks = np.empty_like(order)
    idx = np.arange(P)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(idx, order.shape), axis=0)
    return ranks.astype(np.float64) / max(P - 1, 1)


# --------------------------------------------------------------------------- #
# scorers                                                                     #
# --------------------------------------------------------------------------- #
def coverage(D: np.ndarray) -> np.ndarray:
    """Mean inter-class separation per feature (~ Fisher/point-biserial). Baseline."""
    return D.mean(axis=0, dtype=np.float64)


def exclusive_coverage(D: np.ndarray) -> np.ndarray:
    """Mean per-pair margin of feature f OVER the best *other* feature (top-2 trick)."""
    P, F = D.shape
    # max and argmax per row, plus second max
    order = np.argsort(D, axis=1)  # ascending
    top1 = order[:, -1]
    top2 = order[:, -2]
    rowmax = D[np.arange(P), top1]
    rowmax2 = D[np.arange(P), top2]
    best_other = np.broadcast_to(rowmax[:, None], D.shape).copy()
    # for the per-row argmax feature, best_other is the second max
    best_other[np.arange(P), top1] = rowmax2
    contrib = np.maximum(D - best_other, 0.0)
    return contrib.mean(axis=0, dtype=np.float64)


def rare_pair_coverage(D: np.ndarray, eps: float = 1e-3, winsor_q: float = 0.99):
    """Difficulty-weighted coverage. Returns (soft, tau_count) variants.

    soft:   weight(pair) = 1 / (eps + sum_f Dn[pair,f]),  Dn = rank-normalized D
            (rank-norm prevents high-variance numerics from dominating S_p and
             prevents near-zero S_p from turning this into a label-noise detector).
            Weight winsorized at the winsor_q quantile (bounds influence of a single
            near-duplicate cross-class pair).
    tau:    weight(pair) = 1 / (eps + #features separating the pair above median),
            the ТЗ's original integer form -- reported for comparison.
    """
    Dn = _rank_normalize_columns(D)
    S = Dn.sum(axis=1)
    w = 1.0 / (eps + S)
    cap = np.quantile(w, winsor_q)
    w = np.minimum(w, cap)
    soft = (D * w[:, None]).sum(axis=0) / w.sum()

    # tau-count variant: binarize each column at its own median
    thr = np.median(D, axis=0)
    sep = (D > thr[None, :]).sum(axis=1).astype(np.float64)
    wt = 1.0 / (eps + sep)
    wt = np.minimum(wt, np.quantile(wt, winsor_q))
    tau = (D * wt[:, None]).sum(axis=0) / wt.sum()
    return soft.astype(np.float64), tau.astype(np.float64)


def separation_profile_similarity(D: np.ndarray) -> np.ndarray:
    """(F x F) similarity: two features are similar if they separate the SAME pairs.

    sim(f,g) = corr(D[:,f], D[:,g]).  This is NOT corr(x_f, x_g): two features with
    low input correlation can have near-identical separation profiles (RSI14/MACD).
    """
    Dc = D - D.mean(axis=0, keepdims=True)
    norm = np.sqrt((Dc ** 2).sum(axis=0))
    norm = np.where(norm > 0, norm, 1.0)
    Dn = Dc / norm
    return (Dn.T @ Dn).astype(np.float64)


# --------------------------------------------------------------------------- #
# generic mRMR (Peng 2005) with a PLUGGABLE similarity matrix (Experiment B)   #
# --------------------------------------------------------------------------- #
def mrmr_select(relevance: np.ndarray, sim: np.ndarray, k: int) -> list[int]:
    """Greedy mRMR: pick argmax( relevance[f] - mean_{g in S} sim[f,g] ).

    Identical relevance + greedy loop across sim choices (corr / SU / separation_profile)
    => the redundancy metric is the ONLY moving part. sim is assumed in [0,1]-ish scale.
    """
    F = len(relevance)
    k = min(k, F)
    selected = [int(np.argmax(relevance))]
    remaining = set(range(F)) - set(selected)
    while len(selected) < k and remaining:
        best_f, best_score = None, -np.inf
        for f in remaining:
            red = np.mean([sim[f, g] for g in selected])
            score = relevance[f] - red
            if score > best_score:
                best_score, best_f = score, f
        selected.append(best_f)
        remaining.discard(best_f)
    return selected
