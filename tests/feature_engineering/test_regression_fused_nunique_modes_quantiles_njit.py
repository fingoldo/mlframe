"""Regression: njit-fused nunique/modes/quantiles core stays bit-identical to the numpy reference.

Pins the perf optimization in feature_engineering/numerical.py::_fused_nunique_modes_quantiles
(its body is now `_fused_nunique_modes_quantiles_kernel`, a single @njit pass). The fused kernel
must reproduce the prior numpy stack (boundary / nonzero / diff / append + np.lexsort mode pick +
Hyndman-Fan type-8 quantile interp) EXACTLY -- including the tie-break and all-tie / low-cardinality
edge cases -- so feature selection downstream is unaffected.

Verified to FAIL on pre-fix code if the kernel ever diverges (e.g. wrong mode tie-break or quantile
index) by asserting exact agreement against the inlined numpy reference below.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.feature_engineering.numerical as N


def _numpy_reference(arr: np.ndarray, q: np.ndarray, max_modes: int) -> tuple:
    """The pre-fix numpy-only body of _fused_nunique_modes_quantiles."""
    s = np.sort(arr)
    n = s.size
    boundary = np.empty(n, dtype=bool)
    boundary[0] = True
    np.not_equal(s[1:], s[:-1], out=boundary[1:])
    idx = np.nonzero(boundary)[0]
    vals = s[idx]
    counts = np.diff(np.append(idx, n))
    mm = min(max_modes, len(counts))
    modes_indices = np.lexsort((vals, -counts))[:mm]
    fmc = counts[modes_indices[0]]
    if fmc == 1:
        modes_min = modes_max = modes_mean = modes_qty = np.nan
    else:
        best = [vals[modes_indices[0]]]
        for i in range(1, mm):
            ni = modes_indices[i]
            if counts[ni] < fmc:
                break
            best.append(vals[ni])
        best = np.asarray(best)
        modes_min, modes_max, modes_mean, modes_qty = best.min(), best.max(), best.mean(), len(best)
    res = (len(vals), modes_min, modes_max, modes_mean, modes_qty)
    h = (n + 1.0 / 3.0) * q + 1.0 / 3.0
    np.clip(h, 1.0, float(n), out=h)
    lo = np.floor(h).astype(np.intp) - 1
    hi = np.minimum(lo + 1, n - 1)
    g = h - np.floor(h)
    quantiles = s[lo] * (1.0 - g) + s[hi] * g
    res = res + tuple(quantiles)
    res = res + tuple(N.compute_ncrossings(arr=arr, marks=quantiles))
    return res


_Q = np.array([0.1, 0.25, 0.5, 0.75, 0.9], dtype=np.float64)


def _gen(rng, kind, n):
    if kind == 0:
        return rng.standard_normal(n)
    if kind == 1:  # low-cardinality / many ties (exercises the mode tie-break)
        return rng.integers(0, 5, n).astype(np.float64)
    if kind == 2:  # all identical (single mode, all-tie)
        return np.full(n, float(rng.standard_normal()))
    return np.round(rng.standard_normal(n), 1)  # rounded -> mixed ties


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
def test_fused_kernel_bit_identical_to_numpy_reference(kind):
    rng = np.random.default_rng(1234 + kind)
    for _ in range(400):
        n = int(rng.integers(2, 250))
        arr = _gen(rng, kind, n)
        ref = np.asarray(_numpy_reference(arr, _Q, 10), dtype=np.float64)
        got = np.asarray(N._fused_nunique_modes_quantiles(arr, _Q, "median_unbiased", 10), dtype=np.float64)
        # Exact agreement: integer-valued fields (nunique/modes_qty/ncrossings) and modes/quantiles.
        assert ref.shape == got.shape
        assert np.allclose(ref, got, rtol=0.0, atol=0.0, equal_nan=True), (
            f"kind={kind} n={n}\nref={ref}\ngot={got}"
        )


def test_fused_kernel_matches_low_cardinality_multimode():
    # Two values share the global-max count -> modes_min/max span them, modes_qty == 2.
    arr = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0], dtype=np.float64)
    ref = np.asarray(_numpy_reference(arr, _Q, 10), dtype=np.float64)
    got = np.asarray(N._fused_nunique_modes_quantiles(arr, _Q, "median_unbiased", 10), dtype=np.float64)
    assert np.allclose(ref, got, rtol=0.0, atol=0.0, equal_nan=True)


def test_fused_kernel_all_distinct_modes_are_nan():
    arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    got = N._fused_nunique_modes_quantiles(arr, _Q, "median_unbiased", 10)
    # nuniques == 5, modes all NaN (max count == 1).
    assert got[0] == 5.0
    assert all(np.isnan(x) for x in got[1:5])
