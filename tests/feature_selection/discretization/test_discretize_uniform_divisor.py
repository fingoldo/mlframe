"""Wave 9.1 loop-iter-33 regression: ``discretize_uniform`` divisor
must be the canonical ``(max - min)``, not the buggy
``(max - min + min/2)``.

Pre-fix at ``discretization.py:607``::

    rev_bin_width = n_bins / (max_value - min_value + min_value / 2)

This formula silently miscoded any positive-shifted input. Concrete
demos at n=1000, n_bins=10:
- ``linspace(1000, 1100)``: collapsed to 2 bins {0: 600, 1: 400}
  instead of 10 evenly populated bins.
- ``linspace(10, 110)``: bin 9 underfilled (55 vs ~100 expected).
- ``linspace(-100, 100)``: bin 9 overcrowded (325 vs ~100 expected).
- ``linspace(-100, -50)``: div-by-zero RuntimeWarning, everything ->
  bin 0.

Effect: every downstream MI / SU / MRMR score on uniform-discretized
positive-shifted features was poisoned. Affects prices, distances,
counts, epoch timestamps - any feature with mean significantly
nonzero. Total bin collapse on negative ranges.

Severity: P0 silent.

Fix (mirrored CPU + CUDA paths):
- Canonical formula ``rev = n_bins / (max - min)``.
- Constant column (``max == min``) returns all-zero codes honestly
  instead of dividing by zero.
- CUDA mirror at line 850 fixed identically (cross-backend
  bit-comparability preserved by both using the canonical formula).
"""

from __future__ import annotations

import numpy as np


def _assert_uniformly_binned(arr, n_bins=10):
    """Each bin (0..n_bins-1) must be populated and counts roughly equal."""
    from mlframe.feature_selection.filters.discretization import discretize_uniform

    out = discretize_uniform(
        arr=arr,
        n_bins=n_bins,
        min_value=float(arr.min()),
        max_value=float(arr.max()),
        dtype=np.int16,
    )
    codes_used = sorted(set(int(c) for c in out))
    assert int(out.max()) <= n_bins - 1
    assert int(out.min()) >= 0
    return codes_used, out


def test_positive_shifted_linspace_uses_all_bins():
    """Pre-fix collapsed to 2 bins. Post-fix uses all 10."""
    arr = np.linspace(1000, 1100, 1000)
    codes, _ = _assert_uniformly_binned(arr, n_bins=10)
    assert codes == list(range(10)), f"positive-shifted linspace lost bins: used={codes}"


def test_small_positive_linspace_uses_all_bins():
    """Pre-fix bin 9 was underfilled (55 of 1000). Post-fix evenly spread."""
    arr = np.linspace(10, 110, 1000)
    codes, out = _assert_uniformly_binned(arr, n_bins=10)
    assert codes == list(range(10))
    # Last bin should have ~100 (n_total / n_bins) not 55.
    _, counts = np.unique(out, return_counts=True)
    assert counts.min() >= 90, f"bin imbalance still present; counts={counts.tolist()}"


def test_crossing_zero_linspace_uses_all_bins():
    """Pre-fix bin 9 had 325 (4x overshoot). Post-fix balanced."""
    arr = np.linspace(-100, 100, 1000)
    codes, _out = _assert_uniformly_binned(arr, n_bins=10)
    assert codes == list(range(10))


def test_negative_only_linspace_no_div_by_zero():
    """Pre-fix div-by-zero -> everything bin 0. Post-fix all 10 bins used."""
    arr = np.linspace(-100, -50, 1000)
    codes, _ = _assert_uniformly_binned(arr, n_bins=10)
    assert codes == list(range(10))


def test_constant_column_yields_single_bin():
    """Zero-range input must produce a single honest bin 0, not crash."""
    from mlframe.feature_selection.filters.discretization import discretize_uniform

    arr = np.full(500, 7.0)
    out = discretize_uniform(arr=arr, n_bins=10, min_value=7.0, max_value=7.0, dtype=np.int16)
    assert (out == 0).all()


def test_bin_count_invariant():
    """For monotone input, count per bin must be ~n_total / n_bins."""
    arr = np.linspace(50, 200, 2000)
    out = _assert_uniformly_binned(arr, n_bins=10)[1]
    _, counts = np.unique(out, return_counts=True)
    expected = 2000 / 10
    assert (counts >= expected * 0.9).all() and (counts <= expected * 1.1).all()


def test_e2e_via_discretize_array():
    """End-to-end through the public ``discretize_array(method='uniform')``
    entry point.
    """
    from mlframe.feature_selection.filters.discretization import discretize_array

    arr = np.linspace(1000, 1100, 1000)
    out = discretize_array(arr=arr, n_bins=10, method="uniform", dtype=np.int16)
    codes = sorted(set(int(c) for c in out))
    assert codes == list(range(10))


def test_discretize_array_uniform_large_n_routes_to_parallel_twin_bit_identical(monkeypatch):
    """Size-gated parallel twin: ``discretize_array(method='uniform')`` on a large array MUST dispatch to the prange
    twin ``discretize_uniform_parallel`` (17.9x @10M) and produce a result byte-identical to the serial kernel.

    Pre-fix the uniform branch called ``discretize_uniform`` (serial) unconditionally and the parallel twin did not
    exist; this test fails pre-fix (ImportError on the twin) and the spy proves the large-n dispatch routes correctly.
    """
    from mlframe.feature_selection.filters import discretization as D

    n = 200_000
    rng = np.random.default_rng(7)
    arr = rng.standard_normal(n).astype(np.float64)

    calls = {"par": 0}
    real_par = D.discretize_uniform_parallel

    def _spy(*a, **k):
        calls["par"] += 1
        return real_par(*a, **k)

    monkeypatch.setattr(D, "discretize_uniform_parallel", _spy)
    out = D.discretize_array(arr=arr, n_bins=10, method="uniform", dtype=np.int8)
    assert calls["par"] == 1, "large-n uniform path must route to the parallel twin"

    mn, mx = float(arr.min()), float(arr.max())
    serial = D.discretize_uniform(arr, 10, mn, mx, dtype=np.int8)
    assert np.array_equal(out, serial), "parallel twin must be byte-identical to the serial kernel"


def test_discretize_array_uniform_small_n_stays_serial(monkeypatch):
    """Below the crossover the serial kernel wins; the parallel twin must NOT be invoked for small arrays."""
    from mlframe.feature_selection.filters import discretization as D

    arr = np.linspace(0.0, 100.0, 1000)
    calls = {"par": 0}
    real_par = D.discretize_uniform_parallel
    monkeypatch.setattr(D, "discretize_uniform_parallel", lambda *a, **k: (calls.__setitem__("par", calls["par"] + 1), real_par(*a, **k))[1])
    D.discretize_array(arr=arr, n_bins=10, method="uniform")
    assert calls["par"] == 0, "small-n uniform path must stay on the serial kernel"
