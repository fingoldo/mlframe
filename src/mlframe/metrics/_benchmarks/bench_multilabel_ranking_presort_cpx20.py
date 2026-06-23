"""CPX20 bench: O(n*K^2) -> O(n*K log K) for LRAP + label-ranking-loss kernels.

The OLD kernels (`_lrap_kernel`, `_ranking_loss_kernel`) rescan all K labels
for every true label (LRAP) / do an O(K^2) true-x-false pair double-loop
(ranking-loss). Both are quadratic in K. The NEW kernels presort each row's
scores ONCE (descending) and compute ranks / mis-ordered-pair counts in a
single forward pass with tie-group handling matching the 0.5 convention.

Run:  python -m mlframe.metrics._benchmarks.bench_multilabel_ranking_presort_cpx20

Both OLD (copied verbatim from _multilabel_extras.py) and NEW kernels are
defined inline so this is a real A/B against the actual prior code, not a
from-memory rewrite. Identity is checked exact (==) on random/tied/edge
inputs before timing.
"""
from __future__ import annotations

import time

import numpy as np
import numba

NJIT = dict(fastmath=False, cache=True, nogil=True)


# ===================== OLD kernels (verbatim baseline) =====================


@numba.njit(**NJIT)
def _lrap_old(y_true, scores):
    n, K = y_true.shape
    total = 0.0
    counted = 0
    for i in range(n):
        n_true = 0
        for k in range(K):
            if y_true[i, k] != 0:
                n_true += 1
        if n_true == 0 or n_true == K:
            continue
        counted += 1
        row_sum = 0.0
        for k in range(K):
            if y_true[i, k] == 0:
                continue
            sk = scores[i, k]
            rank = 0
            tp_rank = 0
            for j in range(K):
                if scores[i, j] >= sk:
                    rank += 1
                    if y_true[i, j] != 0:
                        tp_rank += 1
            row_sum += tp_rank / rank
        total += row_sum / n_true
    if counted == 0:
        return np.nan
    return total / counted


@numba.njit(**NJIT)
def _ranking_loss_old(y_true, scores):
    n, K = y_true.shape
    total = 0.0
    counted = 0
    for i in range(n):
        n_true = 0
        for k in range(K):
            if y_true[i, k] != 0:
                n_true += 1
        n_false = K - n_true
        if n_true == 0 or n_false == 0:
            continue
        counted += 1
        bad = 0.0
        for t in range(K):
            if y_true[i, t] == 0:
                continue
            for f in range(K):
                if y_true[i, f] != 0:
                    continue
                if scores[i, f] > scores[i, t]:
                    bad += 1.0
                elif scores[i, f] == scores[i, t]:
                    bad += 0.5
        total += bad / (n_true * n_false)
    if counted == 0:
        return np.nan
    return total / counted


# ===================== NEW kernels (presort, O(n*K log K)) =====================


@numba.njit(**NJIT)
def _lrap_new(y_true, scores):
    """LRAP via per-row descending-score sort + single forward pass.

    Sort the row by descending score. Walk it in tie-groups (runs of equal
    score): every label in a group shares the same overall rank (cumulative
    count incl. the whole group) and the same tp_rank (cumulative true count
    incl. the whole group) -- exactly what the OLD `score[j] >= sk` rescan
    computes. Each true label in the group then adds tp_rank/rank.
    """
    n, K = y_true.shape
    total = 0.0
    counted = 0
    for i in range(n):
        n_true = 0
        for k in range(K):
            if y_true[i, k] != 0:
                n_true += 1
        if n_true == 0 or n_true == K:
            continue
        counted += 1
        order = np.argsort(scores[i])[::-1]  # descending score
        row_sum = 0.0
        seen_total = 0
        seen_true = 0
        g = 0
        while g < K:
            # Extent of the tie-group [g, h) sharing this score.
            s_g = scores[i, order[g]]
            h = g + 1
            while h < K and scores[i, order[h]] == s_g:
                h += 1
            grp_true = 0
            for p in range(g, h):
                if y_true[i, order[p]] != 0:
                    grp_true += 1
            rank = seen_total + (h - g)
            tp_rank = seen_true + grp_true
            if grp_true > 0:
                row_sum += grp_true * (tp_rank / rank)
            seen_total = rank
            seen_true = tp_rank
            g = h
        total += row_sum / n_true
    if counted == 0:
        return np.nan
    return total / counted


@numba.njit(**NJIT)
def _ranking_loss_new(y_true, scores):
    """Label-ranking loss via per-row descending sort + single forward pass.

    Walk the row in descending-score tie-groups maintaining `false_above`
    (false labels strictly above the current group). A true label in a group
    is mis-ordered relative to: every false label strictly above it
    (`false_above`, counts 1.0 each) plus every false label tied with it
    (within-group, counts 0.5 each). Summing over the group's true labels:
        bad += grp_true*false_above + 0.5*grp_true*grp_false
    matching the OLD O(K^2) pair loop (> => 1.0, == => 0.5) bit-for-bit.
    """
    n, K = y_true.shape
    total = 0.0
    counted = 0
    for i in range(n):
        n_true = 0
        for k in range(K):
            if y_true[i, k] != 0:
                n_true += 1
        n_false = K - n_true
        if n_true == 0 or n_false == 0:
            continue
        counted += 1
        order = np.argsort(scores[i])[::-1]
        bad = 0.0
        false_above = 0
        g = 0
        while g < K:
            s_g = scores[i, order[g]]
            h = g + 1
            while h < K and scores[i, order[h]] == s_g:
                h += 1
            grp_true = 0
            grp_false = 0
            for p in range(g, h):
                if y_true[i, order[p]] != 0:
                    grp_true += 1
                else:
                    grp_false += 1
            bad += grp_true * false_above + 0.5 * grp_true * grp_false
            false_above += grp_false
            g = h
        total += bad / (n_true * n_false)
    if counted == 0:
        return np.nan
    return total / counted


# ===================== identity + timing =====================


def _make(n, K, rng, tie_level=0.0):
    yt = (rng.random((n, K)) < 0.3).astype(np.int64)
    if tie_level <= 0.0:
        sc = rng.random((n, K)).astype(np.float64)
    else:
        # Quantize scores to induce ties.
        levels = max(2, int(K * (1.0 - tie_level)))
        sc = (rng.integers(0, levels, size=(n, K))).astype(np.float64)
    return yt, sc


def _check_identity():
    rng = np.random.default_rng(0)
    cases = []
    # random no-tie
    cases.append(_make(200, 20, rng, 0.0))
    cases.append(_make(200, 50, rng, 0.0))
    # heavy ties
    cases.append(_make(200, 30, rng, 0.8))
    cases.append(_make(200, 30, rng, 1.0))  # all scores few-valued
    # edges: rows with n_true=0, n_true=K, all-equal scores
    yt = np.zeros((5, 8), dtype=np.int64)
    yt[1, :] = 1            # all true
    yt[2, 0] = 1            # one true
    yt[3, :4] = 1
    sc = np.ones((5, 8), dtype=np.float64)  # all equal scores
    cases.append((yt, sc))
    # single row, all-equal
    cases.append((np.array([[1, 0, 1, 0]], dtype=np.int64),
                  np.array([[2.0, 2.0, 2.0, 2.0]])))
    ok = True
    for idx, (yt, sc) in enumerate(cases):
        for name, old, new in (("lrap", _lrap_old, _lrap_new),
                               ("rloss", _ranking_loss_old, _ranking_loss_new)):
            o = old(yt, sc)
            v = new(yt, sc)
            same = (o == v) or (np.isnan(o) and np.isnan(v))
            if not same:
                ok = False
                print(f"  DIVERGE case={idx} {name}: old={o!r} new={v!r}")
    print(f"identity: {'EXACT' if ok else 'DIVERGED'}")
    return ok


def _bestof(fn, yt, sc, reps=7):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        fn(yt, sc)
        best = min(best, time.perf_counter() - t)
    return best


def main():
    rng = np.random.default_rng(123)
    # warm JIT
    wyt, wsc = _make(10, 5, rng)
    for f in (_lrap_old, _lrap_new, _ranking_loss_old, _ranking_loss_new):
        f(wyt, wsc)

    _check_identity()

    n = 2000
    print(f"\n{'shape':>14} {'kernel':>8} {'OLD ms':>10} {'NEW ms':>10} {'speedup':>8}")
    for K in (20, 50, 100):
        yt, sc = _make(n, K, rng, 0.0)
        for name, old, new in (("lrap", _lrap_old, _lrap_new),
                               ("rloss", _ranking_loss_old, _ranking_loss_new)):
            to = _bestof(old, yt, sc) * 1e3
            tn = _bestof(new, yt, sc) * 1e3
            print(f"  n={n} K={K:>3} {name:>8} {to:>10.3f} {tn:>10.3f} {to/tn:>7.2f}x")


if __name__ == "__main__":
    main()
