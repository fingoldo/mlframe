"""Analytic large-n MI null (2026-06-16): equivalence to the permutation null + the speed win.

At n >= threshold, mi_direct(return_null_mean=...) replaces the permutation null with the analytic
Miller-Madow null mean + G-test p-value (see _analytic_mi_null). These tests pin: (1) the closed-form
matches the permutation kernel's null mean and is decision-equivalent on the p-value, (2) the off-switch
restores the permutation path, (3) below threshold the analytic path stays dormant (byte-identical),
(4) the analytic path is dramatically faster (the biz-value win that motivated it).
"""

from __future__ import annotations

import time

import numba
import numpy as np
import pytest

from mlframe.feature_selection.filters.permutation import mi_direct
from mlframe.feature_selection.filters._analytic_mi_null import (
    analytic_mi_null,
    analytic_batch_noise_gate,
    analytic_null_min_n,
    analytic_null_applicable,
)

NB = 10


def _binned(n, signal, seed):
    """Helper that binned."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, NB, n).astype(np.int32)
    if signal > 0:
        y = np.where(rng.random(n) < signal, x, rng.integers(0, NB, n)).astype(np.int32)
    else:
        y = rng.integers(0, NB, n).astype(np.int32)
    return np.column_stack([x, y]).astype(np.int32), np.array([NB, NB], dtype=np.int32)


def _md(data, nbins, **kw):
    """Call mi_direct on column pair (0,1) of data with the given nbins/kwargs, for the analytic-null gate tests."""
    return mi_direct(
        data,
        x=(0,),
        y=(1,),
        factors_nbins=nbins,
        npermutations=64,
        min_nonzero_confidence=0.0,
        parallelism="none",
        prefer_gpu=False,
        return_null_mean=True,
        **kw,
    )


def test_permutation_converges_to_analytic(monkeypatch):
    """EQUIVALENCE PROOF: the analytic null is the nperm->infinity limit of the permutation null.
    Both estimate the SAME independence null; the permutation is a Monte-Carlo estimate (error
    ~1/sqrt(nperm)). So as nperm grows the permutation null_mean and p converge to the analytic
    values -- the analytic is not an approximation, it is the exact limit (and more accurate than any
    finite-nperm run). Demonstrated here on both a noise and a signal case at large n."""
    n = max(int(analytic_null_min_n()), 80_000)
    for signal in (0.0, 0.04):
        data, nbins = _binned(n, signal, seed=2)
        monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
        mi_a, _c, nm_a, p_a = _md(data, nbins)  # analytic
        monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
        # high-nperm permutation -> should match the analytic value closely.
        _mi_lo, _c, nm_lo, _p_lo = mi_direct(
            data, x=(0,), y=(1,), factors_nbins=nbins, npermutations=64, min_nonzero_confidence=0.0, parallelism="none", prefer_gpu=False, return_null_mean=True
        )
        mi_hi, _c, nm_hi, p_hi = mi_direct(
            data,
            x=(0,),
            y=(1,),
            factors_nbins=nbins,
            npermutations=2048,
            min_nonzero_confidence=0.0,
            parallelism="none",
            prefer_gpu=False,
            return_null_mean=True,
        )
        # observed MI identical on both paths (same estimator).
        assert mi_a == pytest.approx(mi_hi, rel=1e-9)
        # high-nperm permutation null_mean within 2% of analytic; and CLOSER than low-nperm (convergence).
        assert abs(nm_hi - nm_a) / max(nm_a, 1e-9) < 0.02
        assert abs(nm_hi - nm_a) <= abs(nm_lo - nm_a) + 1e-6
        # p converges too: high-nperm p within 0.02 of analytic, and the keep/reject decision agrees.
        assert abs(p_hi - p_a) < 0.02
        assert (p_a < 0.05) == (p_hi < 0.05)


def test_occupancy_safe_condition():
    """The analytic null applies only when n >= threshold AND cells are not sparse (avg expected
    count N/(Bx*By) >= 5). A high-cardinality table with sparse cells must NOT be applicable even at
    large n -- the caller then falls back to the (sparsity-correct) permutation test."""
    n = max(int(analytic_null_min_n()), 60_000)
    # low cardinality (10x10 -> 100 cells -> ~600/cell at 60k): applicable.
    assert analytic_null_applicable(n, 10, 10)
    # high cardinality (300x300 -> 90k cells -> <1/cell at 60k): sparse -> NOT applicable.
    assert not analytic_null_applicable(n, 300, 300)
    # below the n threshold: never applicable regardless of occupancy.
    assert not analytic_null_applicable(int(analytic_null_min_n()) - 1, 10, 10)


def test_analytic_formula_miller_madow():
    # null_mean = (Bx-1)(By-1)/(2N); p in [0,1]; df<=0 / mi<=0 -> non-significant.
    """Analytic formula miller madow."""
    nm, p = analytic_mi_null(0.01, 100_000, 10, 10)
    assert nm == pytest.approx((9 * 9) / (2 * 100_000))
    assert 0.0 <= p <= 1.0
    assert analytic_mi_null(0.0, 100_000, 10, 10) == (pytest.approx(81 / 200_000), 1.0)
    assert analytic_mi_null(0.5, 1000, 1, 5) == (0.0, 1.0)  # df<=0 degenerate


@pytest.mark.parametrize("signal", [0.0, 0.05])
def test_analytic_matches_permutation_large_n(monkeypatch, signal):
    """Analytic matches permutation large n."""
    n = max(int(analytic_null_min_n()), 60_000)
    data, nbins = _binned(n, signal, seed=11)

    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
    mi_a, _conf_a, nm_a, p_a = _md(data, nbins)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    mi_p, _conf_p, nm_p, p_p = _md(data, nbins)

    # observed MI is computed the same way on both paths -> identical.
    assert mi_a == pytest.approx(mi_p, rel=1e-9)
    # analytic null mean matches the permutation null mean closely.
    assert nm_a == pytest.approx(nm_p, rel=0.20, abs=1e-4)
    # decision-equivalence at the canonical alpha=0.05: both agree significant / not.
    assert (p_a < 0.05) == (p_p < 0.05)
    if signal > 0:
        assert p_a < 0.05 and p_p < 0.05  # genuine signal -> significant on both


def test_below_threshold_is_dormant(monkeypatch):
    # n < threshold: toggling the analytic flag must NOT change the result (permutation path both).
    """Below threshold is dormant."""
    n = max(1, int(analytic_null_min_n()) // 10)
    data, nbins = _binned(n, 0.05, seed=7)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
    r_on = _md(data, nbins)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    r_off = _md(data, nbins)
    assert r_on == pytest.approx(r_off, rel=1e-9, abs=1e-12)


def test_analytic_is_faster_large_n(monkeypatch):
    """Analytic is faster large n."""
    n = max(int(analytic_null_min_n()), 120_000)
    data, nbins = _binned(n, 0.05, seed=5)
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "1")
    _md(data, nbins)  # warmup
    t = time.perf_counter()
    for _ in range(10):
        _md(data, nbins)
    t_a = time.perf_counter() - t
    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL", "0")
    _md(data, nbins)
    t = time.perf_counter()
    for _ in range(10):
        _md(data, nbins)
    t_p = time.perf_counter() - t
    # analytic skips the permutation shuffles entirely -> materially faster (conservative bar: 3x).
    assert t_a < t_p / 3.0, f"analytic not faster: analytic={t_a:.3f}s permutation={t_p:.3f}s"


def _mi_nats(x, y, n, nb):
    """Mi nats."""
    j = np.zeros((nb, nb))
    np.add.at(j, (x, y), 1)
    j /= n
    px = j.sum(1, keepdims=True)
    py = j.sum(0, keepdims=True)
    m = j > 0
    return float((j[m] * np.log(j[m] / (px @ py)[m])).sum())


def test_batch_noise_gate_keeps_signal_rejects_noise():
    """The analytic batch gate keeps every genuine-signal candidate and rejects pure noise at large n
    -- the asymptotic-exact decision the per-candidate permutation gate approximates (without its
    Monte-Carlo false positives)."""
    nb = NB
    n = max(int(analytic_null_min_n()), 80_000)
    K = 30
    rng = np.random.default_rng(4)
    y = rng.integers(0, nb, n).astype(np.int64)
    cols, signal = [], set()
    for k in range(K):
        if k % 3 == 0:
            signal.add(k)
            cols.append(np.where(rng.random(n) < 0.06, y, rng.integers(0, nb, n)).astype(np.int64))
        else:
            cols.append(rng.integers(0, nb, n).astype(np.int64))
    disc = np.column_stack(cols)
    observed = np.array([_mi_nats(disc[:, k], y, n, nb) for k in range(K)])

    fe = analytic_batch_noise_gate(disc, observed, y, n, min_nonzero_confidence=0.95)
    kept = set(np.where(fe > 0)[0].tolist())

    # all genuine signal kept (no false negatives); kept candidates keep their observed MI unchanged.
    assert signal <= kept, f"dropped genuine signal: missing {sorted(signal - kept)}"
    for k in kept:
        assert fe[k] == pytest.approx(observed[k])
    # noise rejection: a significance gate at alpha = 1 - 0.95 = 0.05 admits ~alpha * n_noise nulls by
    # construction (here 20 noise cols -> ~1 expected). Allow a small false-positive budget; the point
    # is the gate rejects the BULK of noise, not that it is infallible.
    n_noise = K - len(signal)
    false_pos = kept - signal
    assert len(false_pos) <= max(3, int(0.10 * n_noise)), f"too many noise candidates admitted: {sorted(false_pos)} (n_noise={n_noise})"


def test_analytic_batch_noise_gate_vectorised_matches_scalar_loop():
    """The 2026-07-03 vectorisation of ``analytic_batch_noise_gate`` (one array ``chi2.sf`` instead of
    ~K scalar calls) must be BIT-IDENTICAL to the prior per-column scalar loop over the exact
    ``analytic_null_applicable`` + ``analytic_mi_null`` decision -- across n below/above the min-n gate,
    sparse/dense cells, zero and positive observed MI, and single-occupied-bin (df=0) candidates."""
    from mlframe.feature_selection.filters._analytic_mi_null import (
        analytic_null_applicable,
        analytic_mi_null as _scalar_null,
    )

    def _scalar_gate(observed, by, n_rows, min_conf, bx_per_col):
        """Scalar gate."""
        fe = np.asarray(observed, dtype=np.float64).copy()
        alpha = 1.0 - float(min_conf)
        for k in range(fe.shape[0]):
            mi = float(fe[k])
            if mi <= 0.0:
                fe[k] = 0.0
                continue
            bx = int(bx_per_col[k])
            if not analytic_null_applicable(int(n_rows), bx, by):
                continue
            _nm, p = _scalar_null(mi, int(n_rows), bx, by)
            if p >= alpha:
                fe[k] = 0.0
        return fe

    rng = np.random.default_rng(20260703)
    min_conf = 0.95
    for _ in range(400):
        K = int(rng.integers(1, 80))
        observed = rng.uniform(-0.02, 0.6, K)
        observed[rng.random(K) < 0.25] = 0.0  # some non-positive MI
        by = int(rng.integers(2, 10))
        n = int(rng.choice([50, 100, 5_000, 30_000, 50_000, 300_000]))
        bx = rng.integers(1, 15, K)  # 1 -> df=0 edge
        got = analytic_batch_noise_gate(None, observed.copy(), np.arange(by), n, min_conf, bx_per_col=bx.copy())
        exp = _scalar_gate(observed, by, n, min_conf, bx)
        assert np.array_equal(got, exp), f"vectorised gate diverged (K={K}, n={n}, by={by})"


def test_occupied_bins_per_col_rowchunk_parallel_matches_np_unique():
    # The row-chunk parallel _occupied_bins_per_col must equal the per-column np.unique count
    # (non-negative codes only) bit-for-bit across shapes, including masked (negative) sentinels.
    """Occupied bins per col rowchunk parallel matches np unique."""
    import numpy as np
    from mlframe.feature_selection.filters._analytic_mi_null import _occupied_bins_per_col

    rng = np.random.default_rng(17)
    for n, K, nb in [(3000, 5, 10), (20000, 40, 12), (50000, 130, 8)]:
        disc = np.ascontiguousarray(rng.integers(0, nb, size=(n, K)).astype(np.int16))
        disc[rng.random((n, K)) < 0.03] = -1  # masked -> ignored by both
        got = _occupied_bins_per_col(disc, numba.get_num_threads())
        ref = np.array([np.unique(disc[disc[:, k] >= 0, k]).size for k in range(K)], dtype=np.int64)
        assert np.array_equal(got, ref), f"K={K}"
