"""GPU-vs-CPU parity for discretize_2d_array_cuda (P3-3: the CPU/GPU-equivalency agent found ZERO such
tests; the tune-time sweep only ran Gaussian data -- no n_bins>128, no on-edge ties).

Findings, verified empirically (2026-06-22):
* P1-3 (REAL, fixed): int8 request at n_bins>128 wrapped negative on the GPU while the CPU path widened to
  int16 (via _safe_code_dtype). Fixed by widening in discretize_2d_array_cuda too.
* P1-2 (NOT a divergence -- agent mis-trace): NaN routes to the TOP bin on both backends for quantile (the
  rawkernel binary search drives lo->n_cuts for NaN), so no fix needed.
* On clean inputs both methods are BIT-IDENTICAL; on-edge ties differ by <=1 bin (the documented P2-3
  edge-ULP, selection-equivalent, NOT bit-identical). NaN-contaminated columns are the caller's contract to
  scrub (discretize_*_array does not apply _handle_missing) -- both backends are undefined there and degrade
  differently, so this test does not feed raw NaN to the bin-equality assertions.

cupy required.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization import (
    discretize_2d_array,
    discretize_2d_array_cuda,
)

pytest.importorskip("cupy")
try:
    from pyutilz.core.pythonlib import is_cuda_available

    if not is_cuda_available():
        pytest.skip("CUDA not available", allow_module_level=True)
except ImportError:
    pass


@pytest.mark.parametrize("method", ["quantile", "uniform"])
@pytest.mark.parametrize("n_cols", [5, 1200])  # narrow (per-col searchsorted) + wide (rawkernel)
def test_cuda_matches_cpu_clean_inputs(method, n_cols):
    """Clean (finite, no exact-tie plateau) inputs must be BIT-IDENTICAL across backends."""
    rng = np.random.default_rng(7)
    g = discretize_2d_array_cuda(rng.normal(size=(2000, n_cols)), n_bins=10, method=method, dtype=np.int32)
    rng = np.random.default_rng(7)
    c = discretize_2d_array(rng.normal(size=(2000, n_cols)), n_bins=10, method=method, dtype=np.int32)
    np.testing.assert_array_equal(g, c, err_msg=f"GPU/CPU discretize diverged on clean input ({method}, n_cols={n_cols})")


def test_cuda_quantile_edge_ties_within_one_bin():
    """On-edge tie plateaus: GPU vs CPU may differ by <=1 bin (P2-3 edge-ULP, selection-equivalent)."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2000, 5))
    a[a > 1.0] = 1.0  # a dense tie plateau exactly on a quantile edge
    g = discretize_2d_array_cuda(a, n_bins=10, method="quantile", dtype=np.int32).astype(int)
    c = discretize_2d_array(a, n_bins=10, method="quantile", dtype=np.int32).astype(int)
    assert np.abs(g - c).max() <= 1, "quantile edge-tie divergence exceeded the +-1 ULP tolerance"


def test_cuda_quantile_nan_routes_to_top_like_cpu():
    """NaN routes to the top bin on BOTH backends for quantile (P1-2 was a mis-trace)."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2000, 5))
    a[7, :] = np.nan
    g = discretize_2d_array_cuda(a, n_bins=10, method="quantile", dtype=np.int32)
    c = discretize_2d_array(a, n_bins=10, method="quantile", dtype=np.int32)
    assert np.all(g[7] == 9) and np.all(c[7] == 9)  # top bin (n_bins-1)


def test_cuda_int8_widens_like_cpu_at_large_nbins():
    """P1-3: int8 at n_bins>128 must widen (not wrap negative), matching the CPU path."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=(2000, 4))
    g = discretize_2d_array_cuda(a, n_bins=200, method="quantile", dtype=np.int8)
    c = discretize_2d_array(a, n_bins=200, method="quantile", dtype=np.int8)
    assert g.min() >= 0, f"GPU int8 wrapped negative at n_bins=200: min={g.min()}"
    assert g.dtype == c.dtype
    np.testing.assert_array_equal(g, c)


def test_discretize_quantile_rawkernel_built_once_across_calls():
    """The fused per-column searchsorted ``cp.RawKernel`` (used at n_cols>=1000) must be compiled ONCE
    (module-level singleton, ``_get_searchsorted_right_2d_kernel``) and REUSED across every call -- not
    rebuilt from CUDA source text every time ``_discretize_quantile_rawkernel`` runs. Proven two ways:
    (1) ``cp.RawKernel`` itself is constructed at most once across TWO separate wide-n_cols discretize
    calls; (2) the singleton getter returns the IDENTICAL kernel object both times."""
    import cupy as cp

    from mlframe.feature_selection.filters import discretization as disc_mod

    disc_mod._searchsorted_right_2d_cuda = None  # force a fresh build for this test, regardless of import order
    k1 = disc_mod._get_searchsorted_right_2d_kernel()
    k2 = disc_mod._get_searchsorted_right_2d_kernel()
    assert k1 is k2, "the RawKernel singleton getter built a NEW kernel on the second call"

    calls = {"n": 0}
    orig_rawkernel = cp.RawKernel

    def _counting_rawkernel(*a, **kw):
        """Wrap cp.RawKernel to count construction calls, proving the singleton getter never rebuilds on repeat calls."""
        calls["n"] += 1
        return orig_rawkernel(*a, **kw)

    cp.RawKernel = _counting_rawkernel
    try:
        rng = np.random.default_rng(9)
        a1 = rng.normal(size=(2000, 1200))
        a2 = rng.normal(size=(3000, 1200))
        discretize_2d_array_cuda(a1, n_bins=10, method="quantile", dtype=np.int32)
        discretize_2d_array_cuda(a2, n_bins=10, method="quantile", dtype=np.int32)
    finally:
        cp.RawKernel = orig_rawkernel

    assert calls["n"] == 0, (
        f"cp.RawKernel was (re)constructed {calls['n']} time(s) by real discretize calls after the "
        "singleton was already built -- the kernel should have been reused, not recompiled"
    )
    assert disc_mod._get_searchsorted_right_2d_kernel() is k1, "the module-level kernel singleton object changed"
