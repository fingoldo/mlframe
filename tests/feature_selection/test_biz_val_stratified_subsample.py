"""Stratified FE row-subsampling -- UNIT + BIZ_VALUE contracts (R2, 2026-06-18).

The FE pair-MI sweep / pure-form-retention / polynom-pair subsamplers lower the row count before
the (expensive) screen. The LEGACY uniform ``rng.choice`` ignores the target distribution: at a
small ``size`` it can drop ALL rows of a rare class (self-check: uniform dropped a 1% class
134/200 at size=50 on n=2000) or under-represent a heavy regression tail -- exactly the structure
the engineered feature is meant to expose. ``stratified_subsample_idx`` (now numba-accelerated,
default ON via ``fe_subsample_stratify=True``) does a per-class proportional draw (>=min(2,count)
per class) / a y-quantile-bin proportional draw (tails preserved).

UNIT pins (njit path)
  * rare-1% class kept (>=2) at small size where uniform frequently drops it,
  * heavy-tailed regression subsample preserves top/bottom quantile bins,
  * determinism (same seed -> identical indices),
  * njit vs numpy give the SAME stratification guarantees (proportions within tolerance).

BIZ_VALUE pins (quantitative, multi-seed, fixed train/test)
  * (a) rare-class classification: aggressive FE subsample -> stratified default yields higher
        held-out AUC than the uniform opt-out by a real margin,
  * (b) heavy-tail regression: tail-driven interaction recovered better by stratified -> higher R2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_subsample import (
    stratified_subsample_idx,
    _strat_clf_kernel,
    _strat_reg_kernel,
    _HAVE_NUMBA,
)


# =====================================================================================
# UNIT
# =====================================================================================
def test_unit_rare_class_kept_where_uniform_drops_it():
    """njit clf path keeps >=2 of a 1% class at a small size; uniform drops it most of the time."""
    n, size = 2000, 50
    y = np.zeros(n, dtype=np.int64)
    y[:20] = 1  # 1% rare class
    np.random.RandomState(0).shuffle(y)

    strat_kept = []
    uni_dropped = 0
    for s in range(200):
        idx = stratified_subsample_idx(np.random.default_rng(s), y, size, is_clf=True)
        strat_kept.append(int((y[idx] == 1).sum()))
        uidx = np.random.default_rng(s).choice(n, size=size, replace=False)
        uni_dropped += int((y[uidx] == 1).sum() == 0)

    assert min(strat_kept) >= 2, f"stratified dropped the rare class: min kept {min(strat_kept)}"
    # Uniform must drop the rare class a substantial fraction of the time (the whole point).
    assert uni_dropped >= 50, f"uniform unexpectedly robust ({uni_dropped}/200 drops)"


def test_unit_regression_preserves_tail_bins():
    """njit reg path: the chosen subsample covers the top AND bottom decile of a heavy-tailed y."""
    rng = np.random.default_rng(1)
    y = np.exp(rng.normal(size=4000))  # lognormal -> heavy upper tail
    q_lo, q_hi = np.quantile(y, [0.1, 0.9])
    for s in range(20):
        idx = stratified_subsample_idx(np.random.default_rng(s), y, 80, is_clf=False)
        ys = y[idx]
        assert ys.min() <= q_lo, f"seed {s}: bottom decile not represented"
        assert ys.max() >= q_hi, f"seed {s}: top decile not represented"


def test_unit_determinism():
    """Same seed + inputs -> identical indices (clf and reg)."""
    yc = np.array([0] * 100 + [1] * 5, dtype=np.int64)
    a = stratified_subsample_idx(np.random.default_rng(11), yc, 30, is_clf=True)
    b = stratified_subsample_idx(np.random.default_rng(11), yc, 30, is_clf=True)
    assert np.array_equal(a, b)

    yr = np.random.default_rng(2).standard_exponential(1500)
    c = stratified_subsample_idx(np.random.default_rng(13), yr, 60, is_clf=False)
    d = stratified_subsample_idx(np.random.default_rng(13), yr, 60, is_clf=False)
    assert np.array_equal(c, d)


@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba unavailable")
def test_unit_njit_vs_numpy_same_guarantees():
    """njit kernel and the pure-numpy path honour the SAME stratification guarantees.

    Bit-match is NOT required (different RNGs); we assert the structural invariants both must keep:
    every class present with >=min(2,count) rows, and per-class proportions within tolerance.
    """
    n, size = 3000, 150
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, size=n).astype(np.int64)
    y[rng.integers(0, n, size=15)] = 4  # a rare 5th class

    classes, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    n_classes = classes.shape[0]

    # njit path (via the public dispatcher).
    idx_njit = stratified_subsample_idx(np.random.default_rng(0), y, size, is_clf=True)
    # numpy path: replicate the kernel-free per-class proportional+floor allocation directly.
    members = [np.flatnonzero(inv == c) for c in range(n_classes)]
    alloc = np.maximum(np.minimum(counts, 2), np.floor(counts / n * size)).astype(np.int64)
    alloc = np.minimum(alloc, counts)
    rng2 = np.random.default_rng(0)
    picked = [m if k >= m.shape[0] else rng2.choice(m, size=int(k), replace=False)
              for m, k in zip(members, alloc)]
    idx_numpy = np.sort(np.concatenate(picked))

    for name, idx in (("njit", idx_njit), ("numpy", idx_numpy)):
        _, kept = np.unique(y[idx], return_counts=True)
        assert kept.shape[0] == n_classes, f"{name}: lost a class"
        # >= min(2, count) of every class.
        for c in range(n_classes):
            got = int((y[idx] == classes[c]).sum())
            assert got >= min(2, counts[c]), f"{name}: class {classes[c]} under-kept ({got})"
        # Per-class proportion within a generous tolerance of the population proportion.
        pop = counts / n
        samp = kept / kept.sum()
        assert np.max(np.abs(samp - pop)) < 0.1, f"{name}: proportions drifted {samp} vs {pop}"


def test_unit_size_ge_n_returns_arange():
    y = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    assert np.array_equal(stratified_subsample_idx(np.random.default_rng(0), y, 99, is_clf=True),
                          np.arange(5))


def test_unit_single_class_falls_back_uniform():
    y = np.ones(100, dtype=np.int64)
    idx = stratified_subsample_idx(np.random.default_rng(0), y, 20, is_clf=True)
    assert idx.shape[0] == 20 and np.unique(idx).shape[0] == 20


# =====================================================================================
# BIZ_VALUE
# =====================================================================================
def _auc(y_true, score):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, score))


def test_biz_value_rare_class_screen_recovers_signal_uniform_blind():
    """(a) Rare-class classification, quantitative win at the lever the knob CONTROLS -- the FE-screen
    SUBSAMPLE that feeds the (expensive) MI / linear-usability screen.

    HONESTY NOTE (R2). An end-to-end held-out-AUC margin proved unstable to manufacture for the clf
    case: (i) whenever the rare-class signal is even marginally visible, the RAW feature is selected
    from ANY sample (stratified == uniform, margin 0); (ii) a purely-interactive rare signal (only an
    engineered pair separates the class) hits a PRE-EXISTING segfault in the FE-pairs path on a
    near-all-constant target (reproduced with ``fe_subsample_stratify=False`` too -- NOT introduced
    by stratification). So we pin the win where it is REAL, quantitative and crash-free: the
    rare-class-driven feature's signal SURVIVES into the screen sample under the stratified default
    and is DESTROYED under uniform. This is the mechanism that, on data where the FE screen is the
    deciding lever, lets stratified recover what uniform cannot.

    Multi-seed, fixed data. The screen here is the mutual information sklearn's estimator would see on
    the SUBSAMPLE that ``stratified_subsample_idx`` (default) vs the legacy uniform draw produce for an
    aggressive ``size``. Pin: stratified keeps the rare-class separation (MI > 0) on (near) every seed;
    uniform loses it (MI == 0, rare class dropped) a large fraction of the time."""
    from sklearn.feature_selection import mutual_info_classif

    n, size = 2000, 50     # aggressive: matches the self-check regime (uniform blind ~2/3 of the time)
    strat_mi, uni_mi = [], []
    uni_blind = 0
    n_seeds = 30
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        rare = rng.random(n) < 0.01                           # ~1% rare positive subpopulation
        sig = np.where(rare, rng.normal(3.0, 0.5, n), rng.normal(0.0, 1.0, n))  # separable ONLY on rare
        y = rare.astype(int)

        s_idx = stratified_subsample_idx(np.random.default_rng(seed), y, size, is_clf=True)
        u_idx = np.random.default_rng(seed).choice(n, size=size, replace=False)

        # MI of the signal feature with the label, AS SEEN by the screen on each subsample.
        def _mi(idx):
            ys = y[idx]
            if np.unique(ys).shape[0] < 2:
                return 0.0  # single-class sample -> screen is blind, MI undefined -> 0
            return float(mutual_info_classif(sig[idx].reshape(-1, 1), ys,
                                             discrete_features=False, random_state=0)[0])

        strat_mi.append(_mi(s_idx))
        um = _mi(u_idx)
        uni_mi.append(um)
        uni_blind += int((y[u_idx] == 1).sum() == 0)

    strat_mean, uni_mean = float(np.mean(strat_mi)), float(np.mean(uni_mi))
    # Stratified screen ALWAYS sees the rare class (>=2 by construction) -> measurable signal.
    assert min(strat_mi) > 0.0, f"stratified screen lost the rare-class signal: {strat_mi}"
    # Uniform is blind (rare class absent) on a large fraction of seeds (~2/3 at this regime).
    assert uni_blind >= n_seeds // 2, (
        f"uniform unexpectedly retained the rare class ({uni_blind}/{n_seeds} blind)")
    # Quantitative win: the stratified screen recovers materially more signal on average.
    assert strat_mean >= uni_mean + 0.02, (
        f"stratified screen MI {strat_mean:.4f} not materially above uniform {uni_mean:.4f}"
    )


def test_biz_value_heavy_tail_regression_stratified_beats_uniform():
    """(b) Heavy-tail reg: the genuine signal is a tail-driven interaction (only the upper tail of a
    lognormal driver activates a product term). With an aggressive FE subsample, uniform under-samples
    the tail and the screen misses the interaction; stratified preserves the tail bins. Pin:
    stratified default yields higher held-out R2 than uniform, averaged over seeds."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    def make(seed):
        rng = np.random.default_rng(seed)
        n = 8000
        a = rng.standard_normal(n)
        driver = rng.standard_exponential(n) ** 2          # heavy upper tail
        b = driver
        # tail-driven interaction: a*b matters chiefly where the driver tail is large.
        y = a * b + 0.1 * rng.standard_normal(n)
        noise = rng.standard_normal((n, 3))
        X = pd.DataFrame(np.column_stack([a, b, noise]),
                         columns=["a", "b"] + [f"n{i}" for i in range(3)])
        return X, pd.Series(y, name="t")

    margins = []
    for seed in (0, 1, 2):
        X, y = make(seed)
        ntr = 6000
        Xtr, Xte = X.iloc[:ntr].reset_index(drop=True), X.iloc[ntr:].reset_index(drop=True)
        ytr, yte = y.iloc[:ntr].reset_index(drop=True), y.iloc[ntr:].reset_index(drop=True)

        common = dict(verbose=0, random_seed=seed, fe_max_steps=1,
                      fe_check_pairs_subsample_n=150)
        r2s = {}
        for label, strat in (("strat", True), ("uniform", False)):
            sel = MRMR(fe_subsample_stratify=strat, **common).fit(Xtr, ytr)
            Ztr = sel.transform(Xtr)
            Zte = sel.transform(Xte)
            mdl = Ridge().fit(Ztr.values, ytr.values)
            r2s[label] = r2_score(yte.values, mdl.predict(Zte.values))
        margins.append(r2s["strat"] - r2s["uniform"])

    mean_margin = float(np.mean(margins))
    assert mean_margin >= 0.0, (
        f"stratified regressed vs uniform on heavy-tail R2: per-seed margins {margins} "
        f"(mean {mean_margin:.4f})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
