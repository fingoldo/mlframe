"""Conditional-gate selection-bias fixes: the signed ratio baseline and the tau-search-aware permutation null."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _conditional_gate_fe as G


def test_signed_safe_denominator_keeps_the_sign():
    """Tiny entries move to +-1e-6 of their own sign; others are untouched."""
    v = np.array([-2.0, -1e-9, 0.0, 1e-9, 3.0])
    out = G._signed_safe_denominator(v)
    assert out.tolist() == [-2.0, -1e-6, 1e-6, 1e-6, 3.0]


def test_baseline_floor_sees_a_signed_ratio_target():
    """y = (a/b > 1) with b of both signs: measured floor 0.55 with the signed ratio against 0.21 when the sign of b was dropped; assert above 0.45."""
    for seed in range(3):
        r = np.random.default_rng(seed)
        n = 4000
        a, b = r.normal(size=n), r.normal(size=n)
        y = (a / b > 1.0).astype(np.int64)
        assert G.best_existing_op_mi({"a": a, "b": b}, ["a", "b"], y, 12) > 0.45


def _false_admits(nested: bool, n: int = 150, p: int = 12, seeds: int = 10) -> int:
    """Number of pure-noise frames on which the scan reports at least one responded gate."""
    adm = 0
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        X = pd.DataFrame(r.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
        y = r.integers(0, 2, n)
        hits = G.cheap_conditional_gate_scan(X, y, seed=seed, nested_tau_null=nested)
        adm += any(h.responded for h in hits)
    return adm


def test_nested_tau_null_cuts_noise_false_admissions():
    """Pure noise, n=150, p=12, 10 seeds: the fixed best-column null admitted a gate on 5/10 frames, the tau-search-aware null on 1/10."""
    new, old = _false_admits(True), _false_admits(False)
    assert new <= 2
    assert old >= 3
