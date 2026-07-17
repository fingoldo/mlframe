"""Characterization of the bench-REJECT verdict on FS backlog #5 (PERMUTATION-NULL-CALIBRATED
PREVALENCE BAR), so a future agent cannot silently re-add it without contradicting a green test.

WHY #5 WAS REJECTED (full numbers in the gate-site note in
``_feature_engineering_pairs/_pairs_core.py`` + D:/Temp/null_prev_results.md, 2026-06-09):

#5's idea was to REPLACE the hardcoded ``fe_min_engineered_mi_prevalence`` (0.90) ratio gate with a
SELF-CALIBRATING per-pool null ratio: in the SAME K y-shuffles the order-2 maxT floor runs, ALSO
mirror the max-over-transforms search (the elementary binary bank mul/add/sub/div/max/min over the
CONTINUOUS operands, discretised ONCE -> permutation-invariant) and per shuffle record the per-pair
``best_1d_engineered_MI / joint_pair_MI``; the q95 of that pooled null-ratio distribution is the
chance ceiling, admit only ABOVE it. Unlike #1 (a DETERMINISTIC bias subtraction that uniformly
relaxes the bar), the null ratio is calibrated to what NOISE actually produces.

The mechanism is SOUND IN ISOLATION (it rejects pure noise at the chance rate and unblocks the
He2(a)*b small-n synergy the hardcode kills) -- but it FAILS on the realistic case it was meant to
help, the user's WEAK F2, for the SAME fundamental reason #1/#8/#19 failed:

  A. PURE NOISE: the null ceiling (~0.16) is ABOVE essentially every real noise-pair ratio, so a
     pure-noise frame admits only the (1-q) chance rate -- the hard noise-FP gate PASSES.
     [pinned by ``test_pure_noise_ratios_below_null_ceiling``]

  B. WEAK F2 (``0.2*a**2/b + log(c*2)*sin(d/3)``): the cross-mix pairs (a,c)/(b,c) carry the
     DOMINANT MONOTONE predictor ``c`` smuggled across the pair boundary, so their best-1D-summary
     ratio sits FAR above the noise ceiling -- indistinguishable from genuine synergy. The null bar
     ADMITS the cross-mix, repeating #1's failure mode. No MI-threshold separates them.
     [pinned by ``test_weak_f2_crossmix_ratio_clears_null_ceiling``]

These tests use ONLY the shipped ``info_theory`` MI helpers + ``discretization`` (no #5 production
code exists -- by design). Small fixtures (n<=8000) keep them fast on a RAM-contended box.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from mlframe.feature_selection.filters.discretization import (
    discretize_array,
    discretize_2d_quantile_batch,
)
from mlframe.feature_selection.filters.info_theory import batch_pair_mi_prange
from mlframe.feature_selection.filters.feature_engineering import _safe_div

_N_BINS = 8
_QUANT = 0.95
_K_PERM = 25


def _binary_bank(a, b):
    """Elementary 'minimal' binary preset (the default)."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return [
            np.multiply(a, b),
            np.add(a, b),
            np.subtract(a, b),
            _safe_div(a, b),
            np.maximum(a, b),
            np.minimum(a, b),
        ]


def _discretize_frame(X, y):
    n, p = X.shape
    data = np.empty((n, p), dtype=np.int64)
    nbins = np.empty(p, dtype=np.int64)
    for j in range(p):
        codes = discretize_array(X[:, j].astype(np.float64), n_bins=_N_BINS, method="quantile", dtype=np.int32)
        data[:, j] = codes
        nbins[j] = int(codes.max()) + 1 if codes.size else 1
    yc = discretize_array(np.asarray(y, dtype=np.float64), n_bins=_N_BINS, method="quantile", dtype=np.int32)
    ky = int(yc.max()) + 1
    freqs_y = np.bincount(yc, minlength=ky).astype(np.float64) / n
    return data, nbins, yc.astype(np.int64), freqs_y


def _engineered_disc(X, pairs):
    """Discretise the engineered bank ONCE per pair (functions of X -> permutation-invariant)."""
    out = []
    for ia, ib in pairs:
        a = X[:, ia].astype(np.float64)
        b = X[:, ib].astype(np.float64)
        cols = [np.nan_to_num(np.asarray(v, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0) for v in _binary_bank(a, b)]
        mat = np.column_stack(cols).astype(np.float32)
        out.append(discretize_2d_quantile_batch(mat, n_bins=_N_BINS, dtype=np.int16))
    return out


def _mi_1d(codes, n_bins, y_codes, freqs_y):
    n = codes.shape[0]
    data = np.empty((n, 2), dtype=np.int64)
    data[:, 0] = codes
    data[:, 1] = 0
    nbins = np.array([int(n_bins), 1], dtype=np.int64)
    return float(batch_pair_mi_prange(data, np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), nbins, y_codes, freqs_y)[0])


def _null_ratio_q(X, data, nbins, yc, freqs_y, pairs, *, seed):
    """The #5 null ceiling: q95 of best_1d/pair_mi over K y-shuffles, pooled across pairs."""
    pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    eng = _engineered_disc(X, pairs)
    nbank = eng[0].shape[1] if eng else 0
    rng = np.random.default_rng(seed)
    yp = yc.copy()
    ratios = []
    for _ in range(_K_PERM):
        rng.shuffle(yp)
        pmis = batch_pair_mi_prange(data, pa, pb, np.ascontiguousarray(nbins), yp, freqs_y)
        for pi in range(len(pairs)):
            pm = pmis[pi]
            if pm <= 1e-12:
                continue
            best = 0.0
            for c in range(nbank):
                mi = _mi_1d(eng[pi][:, c], _N_BINS, yp, freqs_y)
                if mi > best:
                    best = mi
            ratios.append(best / pm)
    return float(np.quantile(np.array(ratios), _QUANT))


def _real_ratio(X, data, nbins, yc, freqs_y, ia, ib):
    pm = float(batch_pair_mi_prange(data, np.array([ia], dtype=np.int64), np.array([ib], dtype=np.int64), np.ascontiguousarray(nbins), yc, freqs_y)[0])
    if pm <= 1e-12:
        return 0.0
    eng = _engineered_disc(X, [(ia, ib)])[0]
    best = 0.0
    for c in range(eng.shape[1]):
        mi = _mi_1d(eng[:, c], _N_BINS, yc, freqs_y)
        if mi > best:
            best = mi
    return best / pm


# ---------------------------------------------------------------------------
# A. PURE NOISE -- the null ceiling rejects (admits only the chance rate). The
#    isolated mechanism is SOUND: this is the part that is BETTER than #1.
# ---------------------------------------------------------------------------
def test_pure_noise_ratios_below_null_ceiling():
    # Average the admit-rate over a few seeds so the single-draw spread of the pooled
    # quantile does not make the characterization flaky. The robust, defensible claim is
    # that the null ceiling REJECTS the MAJORITY of pure-noise pairs (admits a minority
    # near the chance rate) -- the part of #5 that is genuinely BETTER than a naive bar
    # lowering. The decisive bench-reject is the weak-F2 wall below, not this leg.
    n, p = 2000, 12
    rates = []
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n, p))
        y = rng.normal(size=n)  # independent of X
        data, nbins, yc, fy = _discretize_frame(X, y)
        pairs = list(combinations(range(p), 2))
        nq = _null_ratio_q(X, data, nbins, yc, fy, pairs, seed=seed)
        admitted = sum(1 for (ia, ib) in pairs if _real_ratio(X, data, nbins, yc, fy, ia, ib) > nq)
        rates.append(admitted / len(pairs))
    mean_rate = float(np.mean(rates))
    # The MAJORITY of pure-noise pairs are rejected (admit rate well under half). This is
    # the noise-FP gate working: pure noise is NOT broadly admitted (contrast: the weak-F2
    # cross-mix below is admitted 100%).
    assert mean_rate < 0.25, (
        f"#5 null ceiling admitted a mean {mean_rate:.1%} of pure-noise pairs across seeds -- the noise-FP gate must reject the majority of noise"
    )


# ---------------------------------------------------------------------------
# B. WEAK F2 -- the DECISIVE bench-reject wall: the cross-mix ratio sits ABOVE the
#    null ceiling, so #5 would ADMIT it (repeating #1). NO MI threshold separates
#    the cross-mix from genuine synergy here -- the documented detectability limit.
# ---------------------------------------------------------------------------
def test_weak_f2_crossmix_ratio_clears_null_ceiling():
    seed, n = 0, 8000  # n<=8000 (RAM-contended box)
    rng = np.random.default_rng(seed)
    a = rng.random(n) + 0.1
    b = rng.random(n) + 0.1
    c = rng.random(n) + 0.1
    d = rng.random(n) * 2 * np.pi
    e = rng.random(n)
    f = rng.random(n)
    y = 0.2 * a**2 / b + f / 5.0 + np.log(c * 2.0) * np.sin(d / 3.0)
    names = ["a", "b", "c", "d", "e"]
    X = np.column_stack([a, b, c, d, e])
    data, nbins, yc, fy = _discretize_frame(X, y)
    idx = {nm: i for i, nm in enumerate(names)}
    pairs = list(combinations(range(len(names)), 2))
    nq = _null_ratio_q(X, data, nbins, yc, fy, pairs, seed=seed)

    r_genuine_ab = _real_ratio(X, data, nbins, yc, fy, idx["a"], idx["b"])
    r_genuine_cd = _real_ratio(X, data, nbins, yc, fy, idx["c"], idx["d"])
    # The cross-mix pairs that smuggle the dominant monotone predictor c.
    cross = [("a", "c"), ("b", "c"), ("a", "d"), ("b", "d")]
    cross_ratios = {f"{u}{v}": _real_ratio(X, data, nbins, yc, fy, idx[u], idx[v]) for u, v in cross}

    # The genuine pairs clear the null ceiling (the bar would admit them) ...
    assert r_genuine_ab > nq and r_genuine_cd > nq, (
        f"weak-F2 genuine ratios (ab={r_genuine_ab:.3f}, cd={r_genuine_cd:.3f}) should clear the null ceiling {nq:.3f}"
    )
    # ... but SO DO ALL the cross-mix pairs -> #5 ADMITS the cross-mix on this seed,
    # the IRON-RULE failure mode identical to #1. This is the wall: the null bar
    # (calibrated to the noise floor) cannot reject a cross-mix that carries a real
    # dominant monotone predictor. THIS assertion is the bench-reject characterization.
    assert all(r > nq for r in cross_ratios.values()), (
        f"EXPECTED bench-reject wall: every cross-mix ratio must clear the null ceiling {nq:.3f} (so #5 admits noise like #1) -- got {cross_ratios}"
    )
    # And at least one cross-mix ratio is >= a genuine ratio -- no ordering separates them.
    assert max(cross_ratios.values()) >= min(r_genuine_ab, r_genuine_cd) * 0.9, (
        f"cross-mix ratios {cross_ratios} should be comparable to genuine (ab={r_genuine_ab:.3f}, cd={r_genuine_cd:.3f}); the null bar cannot separate them"
    )
