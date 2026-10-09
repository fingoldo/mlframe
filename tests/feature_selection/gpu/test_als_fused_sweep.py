"""The fused ALS sweep returns the original sweep's coefficients and falls back on a singular system."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.hermite_fe import _hermite_prewarp_gpu_resident as pw
from mlframe.feature_selection.filters.hermite_fe._als_fused_gpu import als_sweep_fused


def _designs(basis, deg, n, seed):
    """Device bases of two standardised columns and a centred product-like target."""
    rng = np.random.default_rng(seed)
    za = rng.normal(size=n)
    zb = rng.normal(size=n)
    y = (za**2 - 0.5) * np.tanh(zb) + 0.1 * rng.normal(size=n)
    y -= y.mean()
    return pw._build_basis_matrix_gpu(cp, basis, cp.asarray(za), deg), pw._build_basis_matrix_gpu(cp, basis, cp.asarray(zb), deg), cp.asarray(y)


def _original(Ba, Bb, yc, iters):
    """The multi-kernel sweep, forced via the switch."""
    import os

    old = os.environ.get("MLFRAME_FE_ALS_FUSED")
    os.environ["MLFRAME_FE_ALS_FUSED"] = "0"
    try:
        return pw._als_sweep_gpu(cp, Ba, Bb, yc, iters)
    finally:
        if old is None:
            del os.environ["MLFRAME_FE_ALS_FUSED"]
        else:
            os.environ["MLFRAME_FE_ALS_FUSED"] = old


def _qr_reference(Ba, Bb, yc, iters):
    """The same alternating sweep with every solve done by QR least squares on the host (no normal equations), the accuracy yardstick."""
    Ba, Bb, y = cp.asnumpy(Ba), cp.asnumpy(Bb), cp.asnumpy(yc)

    def scaled(v):
        """The guarded unit-std scaling of a weight vector."""
        sd = v.std()
        return v / (sd if sd > 32 * np.finfo(np.float64).eps * np.abs(v).max() else 1.0)

    cb = np.linalg.lstsq(Bb, y, rcond=None)[0]
    g = Bb @ cb
    for _ in range(iters):
        ca = np.linalg.lstsq(Ba * scaled(g)[:, None], y, rcond=None)[0]
        f = Ba @ ca
        cb = np.linalg.lstsq(Bb * scaled(f)[:, None], y, rcond=None)[0]
        g = Bb @ cb
    return ca, cb


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre"])
@pytest.mark.parametrize("deg,n", [(3, 5_000), (6, 50_000), (8, 200_000)])
def test_fused_is_as_accurate_as_the_original_sweep(basis, deg, n):
    """Against a QR least-squares reference the fused coefficients are no worse than the original sweep's (up to a factor of two and a 1e-4 relative floor): high-degree Laguerre has
    cond(A'A) ~ 1e13, where two normal-equation solvers legitimately differ from each other by more than from the truth."""
    Ba, Bb, yc = _designs(basis, deg, n, seed=deg)
    want = _original(Ba, Bb, yc, 3)
    got = als_sweep_fused(cp, Ba, Bb, yc, 3)
    ref = _qr_reference(Ba, Bb, yc, 3)
    assert got is not None and want[0] is not None
    for g, w, r in zip(got, want, ref):
        scale = max(1.0, float(np.abs(r).max()))
        assert np.abs(g - r).max() <= max(2.0 * np.abs(w - r).max(), 1e-4 * scale)


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev"])
def test_well_conditioned_designs_agree_with_the_original_closely(basis):
    """Where the normal equations are well conditioned the two sweeps agree to solver round-off."""
    Ba, Bb, yc = _designs(basis, 4, 20_000, seed=11)
    want = _original(Ba, Bb, yc, 3)
    got = als_sweep_fused(cp, Ba, Bb, yc, 3)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-8, atol=1e-10)


def test_the_public_sweep_uses_the_fused_path_by_default(monkeypatch):
    """_als_sweep_gpu goes through the fused kernels unless MLFRAME_FE_ALS_FUSED=0."""
    Ba, Bb, yc = _designs("hermite", 4, 2_000, seed=1)
    calls = []
    import mlframe.feature_selection.filters.hermite_fe._als_fused_gpu as fused

    real = fused.als_sweep_fused
    monkeypatch.setattr(fused, "als_sweep_fused", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.delenv("MLFRAME_FE_ALS_FUSED", raising=False)
    pw._als_sweep_gpu(cp, Ba, Bb, yc, 2)
    assert calls == [1]
    monkeypatch.setenv("MLFRAME_FE_ALS_FUSED", "0")
    pw._als_sweep_gpu(cp, Ba, Bb, yc, 2)
    assert calls == [1]


def test_a_singular_system_falls_back_to_the_original_path():
    """A duplicated design column makes the normal matrix singular: the fused kernel reports NaN, the caller's fallback still returns finite coefficients."""
    Ba, Bb, yc = _designs("hermite", 4, 3_000, seed=2)
    Ba[:, 3] = Ba[:, 2]
    assert als_sweep_fused(cp, Ba, Bb, yc, 2) is None
    ca, _ = pw._als_sweep_gpu(cp, Ba, Bb, yc, 2)
    assert ca is None or np.all(np.isfinite(ca))


def test_a_constant_weight_column_does_not_blow_up():
    """A target the first factor cannot move (constant f) keeps the guarded scale at 1 instead of dividing by ~0."""
    Ba, Bb, yc = _designs("legendre", 3, 4_000, seed=3)
    got = als_sweep_fused(cp, Ba, Bb, cp.zeros_like(yc) + 1e-30, 2)
    assert got is None or all(np.all(np.isfinite(v)) for v in got)
