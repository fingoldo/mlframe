"""Regression guard for backlog #2 / frontier-idea-3 (leave-candidate-out / trimmed MI noise floor) -- BENCH-REJECTED.

The hypothesis was that the MRMR FE candidate-pool noise floor
``median(pool) + 3.5 * 1.4826 * MAD(pool)`` self-gates a LONE strong signal
(the signal pooled into med/MAD lifts the floor above itself), fixable by
computing med/MAD leave-candidate-out or upper-trimmed. A 300k-draw Monte
Carlo falsified it: median/MAD is already robust to a single outlier, so the
miss is rare (~10% of signal pools, driven by WIDE NOISE, not self-pooling),
and removing the top/held-out value REGRESSES the all-noise case by lowering
the median while barely moving MAD -- dropping the floor below the remaining
noise band and admitting noise (all-noise leak ~2-3x worse).

These tests pin the two load-bearing facts so the lever is not re-shipped:
1. The pooled median+MAD floor is k=1-robust: a lone outlier does NOT lift the
   floor above itself in the realistic tight-noise regime.
2. Upper-trimming the pool (the proposed "fix") LOWERS the all-noise floor and
   leaks noise -- i.e. it would fail the all-noise ship gate.
"""
from __future__ import annotations

import numpy as np


def _classic_floor(pool: np.ndarray, sigma: float = 3.5) -> float:
    a = np.asarray(pool, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return med + sigma * 1.4826 * mad


def _upper_trimmed_floor(pool: np.ndarray, sigma: float = 3.5, trim_frac: float = 0.10) -> float:
    a = np.sort(np.asarray(pool, dtype=np.float64))
    n = a.size
    n_trim = max(1, min(int(np.ceil(trim_frac * n)), n - 1))
    ref = a[: n - n_trim]
    med = float(np.median(ref))
    mad = float(np.median(np.abs(ref - med)))
    return med + sigma * 1.4826 * mad


def test_pooled_floor_is_k1_robust_admits_lone_signal_tight_noise():
    # Realistic MI pool: 15 tight-noise MIs ~0.010 + one genuine signal 0.040.
    noise = 0.010 + np.array([-.001, .001, -.0005, .0005, 0, .0008, -.0008, .0003,
                              -.0003, .0006, -.0006, .0002, -.0002, .0009, -.0009])
    sig = 0.040
    pool = np.concatenate([[sig], noise])
    floor = _classic_floor(pool)
    # The lone signal is ADMITTED -- median/MAD does NOT self-gate it. The
    # backlog premise ("lone signal drags up med+MAD and rejects itself") does
    # not hold in the realistic regime.
    assert sig >= floor, f"pooled floor {floor:.5f} wrongly rejected lone signal {sig}"


def test_upper_trim_leaks_noise_on_at_least_one_all_noise_pool():
    # The per-pool effect of trimming is seed-dependent (on a right-skewed draw
    # removing the top value can even RAISE the floor), so we do NOT assert a
    # single pool. Instead we confirm the failure mode EXISTS: there is an
    # all-noise pool the classic floor rejects entirely but the trimmed "fix"
    # leaks. Its existence is why the lever fails the all-noise ship gate.
    rng = np.random.default_rng(7)
    found = False
    for _ in range(2000):
        n = int(rng.integers(4, 20))
        base = float(rng.uniform(0.005, 0.05))
        spread = float(rng.uniform(0.0, base))
        noise = np.abs(base + rng.normal(0.0, spread, n))
        if (noise >= _classic_floor(noise)).sum() == 0 and (noise >= _upper_trimmed_floor(noise)).any():
            found = True
            break
    assert found, "expected at least one all-noise pool where trim leaks but classic does not"


def test_monte_carlo_trim_regresses_all_noise_leak():
    # Aggregate evidence (small, deterministic): across many all-noise pools the
    # upper-trimmed floor admits strictly MORE noise pools than the classic floor.
    rng = np.random.default_rng(7)
    classic_leaks = 0
    trim_leaks = 0
    for _ in range(3000):
        n = int(rng.integers(4, 20))
        base = float(rng.uniform(0.005, 0.05))
        spread = float(rng.uniform(0.0, base))
        noise = np.abs(base + rng.normal(0.0, spread, n))
        if (noise >= _classic_floor(noise)).any():
            classic_leaks += 1
        if (noise >= _upper_trimmed_floor(noise)).any():
            trim_leaks += 1
    # The trimmed "fix" admits noise on materially more all-noise pools -- the
    # bench-rejection reason. (~2-3x in the 300k run; require a clear margin here.)
    assert trim_leaks > classic_leaks * 1.5, (
        f"trim leak {trim_leaks} not materially worse than classic {classic_leaks}"
    )
