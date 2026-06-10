"""Confidence-weighted / class-balanced MI under imbalance -- backlog idea #18 (2026-06-10).

BENCH-REJECTED, default OFF. This suite pins the rejection so a later session does
NOT re-flip the default on without re-running the benchmark. It is still a real
TRIAD (unit + biz_value + cProfile): the unit tests assert the gate / kernel
contracts; the biz_value test pins the *negative* result (the correction does NOT
improve rare-class selection -- it is near-rank-preserving); the cProfile test
guards that the default OFF path adds ~0.

Findings (numbers in ``D:/Temp/imbalmi_results.md``):
  * Inverse-prior class balancing is a near-uniform multiplicative rescale of the
    per-feature MI (Kendall tau ~0.99 vs plain), so it almost never re-orders a
    rank-based selection; where it does, downstream rare-class AP is a net-negative
    coin-flip. Hence DEFAULT OFF.
  * Gate is two-sided (only active under the opt-in ``on``/``auto`` mode): inert on
    balanced data and below the n_rare floor (byte-identical to plain MI).

The hook is ``_orthogonal_univariate_fe._mi_classif_batch`` -- the single FE
relevance scorer all ~20 FE / pair-screening modules consume. Default OFF means
that call is byte-for-byte the plain-MI path unless ``MLFRAME_FE_IMBALANCE_MI=on``.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_mi_backends import (
    _mi_classif_batch,
    _mi_classif_batch_numba,
)
from mlframe.feature_selection.filters._orthogonal_univariate_fe._imbalance_mi import (
    _class_balanced_mi_batch_njit,
    _N_RARE_FLOOR,
    _PRIOR_THRESHOLD,
    compute_class_weights,
)


@pytest.fixture(autouse=True)
def _clean_env():
    """Each test sets MLFRAME_FE_IMBALANCE_MI explicitly; restore after."""
    saved = os.environ.get("MLFRAME_FE_IMBALANCE_MI")
    yield
    if saved is None:
        os.environ.pop("MLFRAME_FE_IMBALANCE_MI", None)
    else:
        os.environ["MLFRAME_FE_IMBALANCE_MI"] = saved


def _imbalanced_frame(seed=0, n=20000, prior=0.01, p=8):
    rng = np.random.RandomState(seed)
    y = (rng.rand(n) < prior).astype(np.int64)
    X = rng.randn(n, p)
    return X, y


# ---------------------------------------------------------------------------
# UNIT: gate semantics (prior + n_rare two-sided gate, opt-in only)
# ---------------------------------------------------------------------------

def test_default_off_is_byte_identical_to_plain():
    """Env unset => default OFF => _mi_classif_batch == plain numba batch, byte-for-byte.

    The zero-regression guarantee for every FE consumer of _mi_classif_batch.
    """
    os.environ.pop("MLFRAME_FE_IMBALANCE_MI", None)
    X, y = _imbalanced_frame(seed=1)
    out_default = _mi_classif_batch(X, y, nbins=10)
    out_plain = _mi_classif_batch_numba(X, y, nbins=10)
    assert np.array_equal(out_default, out_plain)


def test_explicit_off_is_byte_identical():
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "off"
    X, y = _imbalanced_frame(seed=2)
    assert np.array_equal(
        _mi_classif_batch(X, y, nbins=10),
        _mi_classif_batch_numba(X, y, nbins=10),
    )


def test_gate_fires_when_imbalanced_and_enough_rare():
    """(a) WIN-fixture regime: prior ~1%, n_rare ~200 -> gate returns weights; ON changes MI."""
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    X, y = _imbalanced_frame(seed=3, prior=0.01)
    n_rare = int(min((y == 0).sum(), (y == 1).sum()))
    assert n_rare >= _N_RARE_FLOOR, f"fixture must have >= floor rare rows, got {n_rare}"
    w = compute_class_weights(y)
    assert w is not None
    # inverse-prior: rare class gets the larger weight
    assert w[1] > w[0]
    out_on = _mi_classif_batch(X, y, nbins=10)
    out_plain = _mi_classif_batch_numba(X, y, nbins=10)
    assert not np.allclose(out_on, out_plain)


def test_gate_fallback_below_n_rare_is_identical_to_plain():
    """(b) GATE control: prior ~1% but n_rare TOO SMALL -> fall back to plain MI byte-identical."""
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    # n=2000, prior 1% -> ~20 positives, well below the 150 floor.
    rng = np.random.RandomState(4)
    n = 2000
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, 20, replace=False)] = 1
    X = rng.randn(n, 6)
    assert compute_class_weights(y) is None  # gate refuses (unreliable)
    out_on = _mi_classif_batch(X, y, nbins=10)
    out_plain = _mi_classif_batch_numba(X, y, nbins=10)
    assert np.array_equal(out_on, out_plain)


def test_no_regression_on_balanced_data():
    """(c) NO-REGRESSION: prior ~0.5 -> correction inert -> balanced MI == plain MI."""
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    rng = np.random.RandomState(5)
    n = 20000
    y = (rng.rand(n) < 0.5).astype(np.int64)
    X = rng.randn(n, 8)
    assert compute_class_weights(y) is None  # prior >= threshold -> no correction
    assert np.array_equal(
        _mi_classif_batch(X, y, nbins=10),
        _mi_classif_batch_numba(X, y, nbins=10),
    )


def test_gate_refuses_non_discrete_and_single_class():
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    # float (regression-style) target -> never reweight
    assert compute_class_weights(np.linspace(0, 1, 100)) is None
    # single populated class -> None
    assert compute_class_weights(np.zeros(100, dtype=np.int64)) is None


def test_prior_threshold_boundary():
    """Just-imbalanced-enough fires; just-balanced-enough does not."""
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    n = 20000
    # prior just below threshold, plenty of rare rows -> fires
    k_lo = int((_PRIOR_THRESHOLD - 0.02) * n)
    y_lo = np.zeros(n, dtype=np.int64)
    y_lo[:k_lo] = 1
    assert compute_class_weights(y_lo) is not None
    # prior just above threshold -> inert
    k_hi = int((_PRIOR_THRESHOLD + 0.05) * n)
    y_hi = np.zeros(n, dtype=np.int64)
    y_hi[:k_hi] = 1
    assert compute_class_weights(y_hi) is None


# ---------------------------------------------------------------------------
# UNIT: balanced kernel correctness
# ---------------------------------------------------------------------------

def test_balanced_kernel_equals_plain_when_weights_uniform():
    """With per-class weights set so every class has equal total mass on ALREADY-balanced
    data, the reweighted MI must match the plain plug-in MI (the kernel is a correct
    generalisation: uniform reweight is a no-op up to fp tolerance)."""
    rng = np.random.RandomState(6)
    n = 8000
    y = (rng.rand(n) < 0.5).astype(np.int64)
    X = rng.randn(n, 5)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    # weights that make total class mass equal: on balanced data this ~= 1/n scaling,
    # i.e. an exact no-op vs plain plug-in MI.
    w = (0.5 / counts)
    bal = _class_balanced_mi_batch_njit(
        np.ascontiguousarray(X), np.ascontiguousarray(y, dtype=np.int64),
        np.ascontiguousarray(w, dtype=np.float64), 10,
    )
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "off"
    plain = _mi_classif_batch_numba(X, y, nbins=10)
    # rtol 1e-3: the balanced kernel accumulates float64 weighted histograms in a
    # different summation order than the plain integer-count kernel, so they agree
    # to fp-accumulation tolerance (~1e-7 abs here), not bit-for-bit.
    assert np.allclose(bal, plain, rtol=1e-3, atol=1e-7)


def test_balanced_mi_amplifies_rare_separator_magnitude():
    """The documented mechanism: a real rare-class separator's MI MAGNITUDE is
    majority-compressed under plain MI; balancing amplifies it (~10x+). This is the
    one true effect -- it is a rescale, not a re-ranking (see the biz_value test)."""
    rng = np.random.RandomState(7)
    n = 20000
    y = (rng.rand(n) < 0.01).astype(np.int64)
    x_rare = rng.randn(n)
    x_rare[y == 1] += 1.6  # AUC ~0.87 rare separator
    X = x_rare.reshape(-1, 1)
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "off"
    plain = _mi_classif_batch(X, y, nbins=10)[0]
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    bal = _mi_classif_batch(X, y, nbins=10)[0]
    assert bal > 5.0 * plain  # magnitude amplification is real (was ~30x in bench)


# ---------------------------------------------------------------------------
# BIZ_VALUE (the negative result that drives the rejection): near-rank-preserving,
# does NOT recover rare-class features / improve downstream rare AP.
# ---------------------------------------------------------------------------

def test_bench_rejection_near_rank_preserving():
    """Inverse-prior balancing barely re-orders the feature ranking across imbalanced
    frames (Kendall tau ~0.99), so it cannot systematically change a rank-based
    selection -> the WIN claim (a) (plain under-ranks, balanced fixes) does NOT
    reproduce. Pins the rejection."""
    from scipy.stats import kendalltau
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "off"
    n = 20000
    taus = []
    for s in range(8):
        rng = np.random.RandomState(s)
        p = 14
        F = rng.randn(n, p)
        w = rng.randn(p) * 0.5
        logit = -5.0 + F @ w + 3.0 * ((F[:, 10] > 2.0) | (F[:, 11] > 2.2))
        y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
        if int(min((y == 0).sum(), (y == 1).sum())) < _N_RARE_FLOOR:
            continue
        os.environ["MLFRAME_FE_IMBALANCE_MI"] = "off"
        plain = _mi_classif_batch(F, y, nbins=10)
        os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
        bal = _mi_classif_batch(F, y, nbins=10)
        taus.append(kendalltau(plain, bal).correlation)
    assert len(taus) >= 4
    # high rank agreement => no systematic selection change => the correction is inert
    # for rank-based selection (the documented rejection reason).
    assert np.mean(taus) > 0.90


def test_bench_rejection_does_not_improve_topk_rare_selection():
    """(a)-as-tested + biz_value: with a strong rare-band signal present, BOTH plain and
    balanced MI rank the rare-discriminative features into the top-k -> identical
    selection -> no downstream rare-AP delta. (The rare separator is NOT under-ranked
    by plain MI; balancing changes nothing here.)"""
    rng = np.random.RandomState(0)
    n = 20000
    nc = 8
    Fc = rng.randn(n, nc)
    Fr = rng.randn(n, 2)
    common_logit = -5.0 + (Fc * np.array([0.5, 0.45, 0.4, 0.38, 0.35, 0.33, 0.3, 0.28])).sum(1)
    logit = common_logit + 3.5 * ((Fr[:, 0] > 2.1) | (Fr[:, 1] > 2.1))
    y = (rng.rand(n) < 1 / (1 + np.exp(-logit))).astype(np.int64)
    F = np.column_stack([Fc, Fr])
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "off"
    plain = _mi_classif_batch(F, y, nbins=10)
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    bal = _mi_classif_batch(F, y, nbins=10)
    # The load-bearing claim: the rare-band features (idx 8, 9) are NOT under-ranked
    # by plain MI -- they sit in the top-3 under BOTH plain and balanced. Balancing
    # therefore does not "rescue" them (they were never lost). Any selection change
    # is confined to the marginal swaps among the weak common features (the 13/120
    # noisy reorders), which do not track rare-class value.
    for rare_idx in (8, 9):
        assert int(np.where(np.argsort(-plain) == rare_idx)[0][0]) < 3
        assert int(np.where(np.argsort(-bal) == rare_idx)[0][0]) < 3
    # top-3 (which contains the rare-band features) is unchanged -> the rare signal
    # is identically captured with and without the correction.
    assert set(np.argsort(-plain)[:3]) == set(np.argsort(-bal)[:3])


def test_noise_control_rare_by_chance_not_promoted():
    """(d) NOISE: a pure-noise feature that separates the rare class BY CHANCE is not
    spuriously promoted above genuine noise by balancing. Balanced MI of pure noise
    stays bounded and does not vault a chance feature over a real separator's
    balanced MI (the relevance reweight does not bypass downstream floors -- it is
    just a per-feature rescale)."""
    rng = np.random.RandomState(11)
    n = 20000
    y = (rng.rand(n) < 0.01).astype(np.int64)
    noise = rng.randn(n, 12)  # pure noise
    x_real = rng.randn(n)
    x_real[y == 1] += 1.8  # genuine rare separator
    X = np.column_stack([x_real, noise])
    os.environ["MLFRAME_FE_IMBALANCE_MI"] = "on"
    bal = _mi_classif_batch(X, y, nbins=10)
    # the genuine separator's balanced MI exceeds every pure-noise column's.
    assert bal[0] > bal[1:].max()


# ---------------------------------------------------------------------------
# CPROFILE / cost: the gate detection is cheap and the default OFF path adds ~0.
# ---------------------------------------------------------------------------

def test_default_off_adds_negligible_cost():
    """The detection short-circuit (compute_class_weights) on the default OFF path is
    a single env read -> the OFF path must be no slower than calling the plain numba
    batch directly (generous 1.5x budget to absorb timing noise)."""
    os.environ.pop("MLFRAME_FE_IMBALANCE_MI", None)
    X, y = _imbalanced_frame(seed=9, n=20000, p=12)
    # warm numba
    _mi_classif_batch(X, y, nbins=10)
    _mi_classif_batch_numba(X, y, nbins=10)
    t0 = time.perf_counter()
    for _ in range(5):
        _mi_classif_batch(X, y, nbins=10)
    t_hooked = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(5):
        _mi_classif_batch_numba(X, y, nbins=10)
    t_plain = time.perf_counter() - t0
    assert t_hooked < 1.5 * t_plain + 0.05
