"""Unit + recall pins for the interaction-propensity pre-rank that chooses WHICH wide-frame columns
enter the capped O(p^2) synergy sweep (see filters/_fe_interaction_prerank.py).

Pins the three contract guarantees the bench established:
  * second_moment_propensity recovers PURE-interaction operands (zero marginal MI) that marginal MI misses,
    at realistic leakage -- on RAW values AND on quantile bin-codes (so the wide-frame wiring may use codes);
  * the L=0.0 perfectly-balanced case is irreducible (no lift over random) -- pinned so we never claim it;
  * top_k_by_interaction_propensity is deterministic, a no-op when top_k >= n_candidates, and O(n*p) cheap.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_interaction_prerank import (
    second_moment_propensity,
    top_k_by_interaction_propensity,
)

N = 8000
NBINS = 8
P = 1200
K = 6  # 12 planted pure-pair operand columns


def _make_frame(p, seed, leak):
    """K pure sign-product pair interactions; operands ~0 marginal signal; ``leak`` adds main effect."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, p))
    cols = rng.choice(p, size=2 * K, replace=False)
    operands, logit = [], np.zeros(N)
    for k in range(K):
        ia, ib = int(cols[2 * k]), int(cols[2 * k + 1])
        operands += [ia, ib]
        a, b = X[:, ia], X[:, ib]
        logit += 1.6 * np.sign(a) * np.sign(b)
        logit += leak * 1.6 * (a + b)
    p_y = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(N) < p_y).astype(np.int32)
    return X, y, set(operands)


def _bin_codes(X):
    out = np.empty_like(X, dtype=np.int16)
    for j in range(X.shape[1]):
        q = np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1])
        out[:, j] = np.searchsorted(q, X[:, j])
    return out


def _marginal_mi(X, y):
    n, p = X.shape
    fy = np.bincount(y, minlength=2) / n
    codes = _bin_codes(X)
    scores = np.empty(p)
    for j in range(p):
        c = codes[:, j]
        mi = 0.0
        for vx in range(NBINS):
            mask = c == vx
            nx = int(mask.sum())
            if nx == 0:
                continue
            px = nx / n
            yy = y[mask]
            for vy in range(2):
                jc = int((yy == vy).sum())
                if jc and fy[vy] > 0:
                    jf = jc / n
                    mi += jf * np.log(jf / (px * fy[vy]))
        scores[j] = mi
    return scores


def _recall(scores, operands, m):
    top = set(np.argsort(scores)[::-1][:m].tolist())
    return len(top & operands) / len(operands)


def test_second_moment_beats_marginal_on_leaky_pure_interaction_raw_and_codes():
    """At realistic leakage (L=0.1) the second-moment propensity recovers the planted pure-interaction
    operands into the top-250 FAR better than marginal MI -- on raw values AND on quantile bin-codes
    (the bin-code path is what the wide-frame wiring uses, so its recall must hold too)."""
    raw_2m, code_2m, marg = [], [], []
    for seed in (0, 1, 2):
        X, y, ops = _make_frame(P, seed, leak=0.1)
        codes = _bin_codes(X).astype(np.float64)
        raw_2m.append(_recall(second_moment_propensity(X, y), ops, 250))
        code_2m.append(_recall(second_moment_propensity(codes, y), ops, 250))
        marg.append(_recall(_marginal_mi(X, y), ops, 250))
    raw_2m, code_2m, marg = np.mean(raw_2m), np.mean(code_2m), np.mean(marg)
    # second-moment (both representations) clears a real recall bar and beats marginal MI by a wide margin.
    assert raw_2m >= 0.60, f"raw 2nd-moment recall too low: {raw_2m:.2f}"
    assert code_2m >= 0.55, f"bin-code 2nd-moment recall too low: {code_2m:.2f}"
    assert raw_2m >= marg + 0.10, f"2nd-moment ({raw_2m:.2f}) did not beat marginal MI ({marg:.2f})"
    # random baseline at top-250 of p=1200 is ~0.21; both must clear it decisively.
    assert code_2m >= 0.40


def test_perfectly_balanced_interaction_is_irreducible():
    """L=0.0 (exact sign product, zero higher-moment leakage): NO O(p) score beats the random baseline.
    Pinned so the pre-rank never CLAIMS to recover this measure-zero case (only the exhaustive sweep can)."""
    recalls = []
    for seed in (0, 1, 2):
        X, y, ops = _make_frame(P, seed, leak=0.0)
        recalls.append(_recall(second_moment_propensity(X, y), ops, 250))
    mean_recall = float(np.mean(recalls))
    random_base = 250 / P  # ~0.208
    # within noise of random -- explicitly NOT a recovery (allow a small slop above the baseline).
    assert mean_recall <= random_base + 0.15, (
        f"unexpected recovery of a perfectly-balanced interaction ({mean_recall:.2f} vs base {random_base:.2f}); "
        "the irreducibility assumption no longer holds -- revisit the pre-rank claims")


def test_top_k_deterministic_and_noop_when_k_exceeds_pool():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 40))
    y = (rng.random(500) < 0.5).astype(int)
    cand = [3, 7, 11, 19, 23, 31]
    a = top_k_by_interaction_propensity(X, y, cand, top_k=3)
    b = top_k_by_interaction_propensity(X, y, cand, top_k=3)
    assert a == b and len(a) == 3 and a == sorted(a)        # deterministic + sorted
    assert set(a).issubset(set(cand))
    # top_k >= pool size -> all candidates, sorted (a pure no-op selection)
    assert top_k_by_interaction_propensity(X, y, cand, top_k=10) == sorted(cand)
    assert top_k_by_interaction_propensity(X, y, cand, top_k=0) == []


def test_nominal_multiclass_is_relabel_invariant():
    """A nominal multiclass target must NOT have its arbitrary class CODES squared (that made the kept set
    depend on the integer assigned to each class -- critique bug #2, 2026-06-19). The one-hot relabel-
    invariant score gives the SAME ranking under any relabeling of the classes."""
    rng = np.random.default_rng(0)
    n, p = 6000, 400
    X = rng.standard_normal((n, p))
    ia, ib, ic, idd = 10, 200, 50, 300
    a, b, c, d = X[:, ia], X[:, ib], X[:, ic], X[:, idd]
    # 4-class NOMINAL target from two leaky interactions (sign product + a main-effect leak so the operands
    # carry recoverable higher-moment signal -- the realistic regime, not the irreducible pure-balanced case).
    s1 = np.sign(a) * np.sign(b) + 0.6 * (a + b)
    s2 = np.sign(c) * np.sign(d) + 0.6 * (c + d)
    cls = (2 * (s1 > 0) + (s2 > 0)).astype(int)  # 0..3, nominal
    operands = {ia, ib, ic, idd}

    # Recall of the operands must be STABLE across relabelings -- the bug (squaring class codes) made it swing
    # 0.12-0.88 (std 0.25) with the arbitrary integer per class. The one-hot score uses only the PARTITION
    # 1[y==c], never the label value, so recall is invariant (up to float summation-order noise on boundary
    # ties). recall = fraction of the 4 operands in the top-250.
    def _recall_top(y_):
        top = set(np.argsort(second_moment_propensity(X, y_))[::-1][:250])
        return len(operands & top) / len(operands)

    recalls = [_recall_top(cls)]
    for relabel in ([3, 1, 0, 2], [2, 3, 1, 0], [1, 0, 3, 2], [0, 3, 2, 1]):
        recalls.append(_recall_top(np.array([relabel[v] for v in cls])))
    assert np.std(recalls) <= 0.03, f"recall not relabel-invariant for nominal multiclass: {recalls}"
    assert np.mean(recalls) >= 0.5, f"multiclass operand recovery too low: {recalls}"


def _planted_interaction_X(p=400, seed=0, leak=0.2):
    """X with two leaky pure-pair interactions; returns X, the continuous driver s, and the operand set."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((4000, p))
    ia, ib, ic, idd = 7, 150, 33, 290
    a, b, c, d = X[:, ia], X[:, ib], X[:, ic], X[:, idd]
    s = np.sign(a) * np.sign(b) + np.sign(c) * np.sign(d) + leak * (a + b + c + d)
    return X, s, {ia, ib, ic, idd}


@pytest.mark.parametrize("target_kind", ["binary", "nominal_multiclass", "ordinal_multiclass",
                                         "regression_continuous", "regression_binned",
                                         "boolean", "string_labels"])
def test_all_target_types_score_finite_and_recover(target_kind):
    """second_moment_propensity must work for EVERY target type: produce finite scores AND recover the planted
    leaky-interaction operands above the random baseline, for binary / nominal / ordinal multiclass / continuous
    regression / binned regression / boolean / non-numeric string labels."""
    X, s, operands = _planted_interaction_X(p=400, seed=0, leak=0.2)
    n = X.shape[0]
    rng = np.random.default_rng(1)

    if target_kind == "binary":
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-s))).astype(int)
    elif target_kind == "nominal_multiclass":
        s2 = np.sign(X[:, 33]) * np.sign(X[:, 290])
        y = (2 * (s > 0) + (s2 > 0)).astype(int)            # 0..3, treated as nominal
    elif target_kind == "ordinal_multiclass":
        y = np.digitize(s, np.quantile(s, [0.25, 0.5, 0.75])).astype(int)   # 0..3 ordered
    elif target_kind == "regression_continuous":
        y = (s + 0.1 * rng.standard_normal(n)).astype(float)               # >64 unique -> moment path
    elif target_kind == "regression_binned":
        y = np.digitize(s, np.quantile(s, np.linspace(0, 1, 9)[1:-1]))      # 8 bins (the synergy-site form)
    elif target_kind == "boolean":
        y = (s > 0)                                                         # bool dtype
    else:  # string_labels (non-numeric nominal)
        lab = np.array(["lo", "mid", "hi"])
        y = lab[np.digitize(s, np.quantile(s, [0.33, 0.66]))]              # object/str array

    scores = second_moment_propensity(X, y)
    assert scores.shape == (X.shape[1],)
    assert np.isfinite(scores).all(), f"{target_kind}: non-finite scores"
    top = set(np.argsort(scores)[::-1][:100])
    recall = len(operands & top) / len(operands)
    assert recall >= 0.5, f"{target_kind}: operand recall {recall:.2f} at top-100 (random ~{100/400:.2f})"


def test_single_class_target_no_crash():
    """A degenerate constant target must not crash and must return finite (zero-information) scores."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 20))
    s = second_moment_propensity(X, np.zeros(300, dtype=int))
    assert np.isfinite(s).all()


def test_constant_column_scores_zero_no_nan():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 5))
    X[:, 2] = 4.0  # constant column -> undefined corr -> must score 0, not NaN
    y = (rng.random(300) < 0.5).astype(int)
    s = second_moment_propensity(X, y)
    assert np.isfinite(s).all()
    assert s[2] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
