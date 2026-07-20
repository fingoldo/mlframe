"""Direct unit coverage for ``_usability_gpu.py`` (mrmr_audit_2026-07-20 test_coverage.md #14) -- the
gated cupy scoring primitives for usability-aware selection / pure-form retention, previously only
exercised transitively via full MRMR fits with ``MLFRAME_FE_GPU_USABILITY=1``. Pins the no-cupy /
env-var-off / globally-disabled fallback gate directly (via monkeypatching the module's own
``_CUPY_AVAIL`` flag rather than uninstalling cupy), plus GPU-vs-host correctness parity for the
corr and additive-basis-residual primitives."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _usability_gpu as ug


def _host_abscorr(u, v):
    """Reference |corr(u, v)| with the same std<1e-12 zero-guard as _abscorr/gpu_abscorr."""
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if u.std() < 1e-12 or v.std() < 1e-12:
        return 0.0
    um, vm = u - u.mean(), v - v.mean()
    denom = np.sqrt((um * um).sum() * (vm * vm).sum())
    if denom <= 0.0:
        return 0.0
    return abs(float((um * vm).sum()) / denom)


class TestFeGpuUsabilityEnabledGate:
    """The gate must be OFF by default and must degrade cleanly with no cupy / no env var / global off."""

    def test_off_by_default_when_env_var_unset(self, monkeypatch):
        """Even with cupy importable, the env var must be explicitly truthy to enable the path."""
        monkeypatch.delenv("MLFRAME_FE_GPU_USABILITY", raising=False)
        assert ug.fe_gpu_usability_enabled() is False

    def test_off_when_cupy_unavailable_regardless_of_env_var(self, monkeypatch):
        """Simulates the no-cupy-installed environment by monkeypatching the module's own import-time
        flag (never uninstalling/reloading cupy itself)."""
        monkeypatch.setattr(ug, "_CUPY_AVAIL", False)
        monkeypatch.setenv("MLFRAME_FE_GPU_USABILITY", "1")
        assert ug.fe_gpu_usability_enabled() is False

    def test_off_for_falsy_env_var_values(self, monkeypatch):
        """Every recognized falsy string value must leave the gate off."""
        for val in ("0", "false", "off", "no", ""):
            monkeypatch.setenv("MLFRAME_FE_GPU_USABILITY", val)
            assert ug.fe_gpu_usability_enabled() is False, f"env var {val!r} must not enable the gate"

    def test_on_when_cupy_available_env_var_set_and_not_globally_disabled(self, monkeypatch):
        """The one path that actually enables the gate -- requires real cupy, so this test is meaningful
        only on a host where cupy is installed and MLFRAME_DISABLE_GPU is not set."""
        pytest.importorskip("cupy")
        monkeypatch.setattr(ug, "_CUPY_AVAIL", True)
        monkeypatch.setenv("MLFRAME_FE_GPU_USABILITY", "1")
        monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert ug.fe_gpu_usability_enabled() is True


class TestGpuAbscorrParity:
    """gpu_abscorr / gpu_abscorr_batch must match the host |corr| reference bit-for-bit-close."""

    def test_gpu_abscorr_matches_host_reference(self):
        """A genuinely correlated pair's GPU |corr| must match the host reference to near-fp precision."""
        pytest.importorskip("cupy")
        rng = np.random.default_rng(0)
        u = rng.standard_normal(500)
        v = 0.6 * u + 0.4 * rng.standard_normal(500)
        assert ug.gpu_abscorr(u, v) == pytest.approx(_host_abscorr(u, v), abs=1e-9)

    def test_gpu_abscorr_constant_input_returns_zero(self):
        """A zero-variance input hits the std<1e-12 zero-guard rather than dividing by zero."""
        pytest.importorskip("cupy")
        u = np.full(100, 3.0)
        v = np.arange(100, dtype=np.float64)
        assert ug.gpu_abscorr(u, v) == 0.0

    def test_gpu_abscorr_empty_input_returns_zero(self):
        """An empty input hits the explicit size==0 early-return."""
        pytest.importorskip("cupy")
        assert ug.gpu_abscorr(np.array([]), np.array([])) == 0.0

    def test_gpu_abscorr_batch_matches_per_column_host_reference(self):
        """The batched multi-column kernel must match looping the scalar host reference per column."""
        pytest.importorskip("cupy")
        rng = np.random.default_rng(1)
        n, k = 400, 5
        cols = rng.standard_normal((n, k))
        v = cols[:, 0] * 0.7 + rng.standard_normal(n) * 0.3
        batch = ug.gpu_abscorr_batch(cols, v)
        expected = np.array([_host_abscorr(cols[:, j], v) for j in range(k)])
        np.testing.assert_allclose(batch, expected, atol=1e-9)

    def test_gpu_abscorr_batch_empty_columns_returns_empty(self):
        """K=0 columns hits the explicit K==0 early-return, not a zero-size cupy reduction crash."""
        pytest.importorskip("cupy")
        out = ug.gpu_abscorr_batch(np.empty((10, 0)), np.arange(10, dtype=np.float64))
        assert out.shape == (0,)


class TestGpuAdditiveBasisResidualParity:
    """gpu_additive_basis_residual must match the documented StandardScaler+LinearRegression equivalence."""

    def test_residual_matches_sklearn_reference(self):
        """The mean-centered-OLS residual must match the CPU StandardScaler+LinearRegression residual
        to near-fp precision, per the module's own documented equivalence."""
        pytest.importorskip("cupy")
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(2)
        n = 600
        xa = rng.standard_normal(n)
        xb = rng.uniform(0.5, 3.0, n)
        fv = xa**2 / xb + 0.05 * rng.standard_normal(n)

        def _host_basis(x):
            """The exact 6-function additive single-operand basis, matching the GPU kernel's own."""
            xs = (x - x.mean()) / (x.std() + 1e-12)
            return [xs, xs * xs, xs**3, np.sign(xs) * np.sqrt(np.abs(xs)), np.sign(xs) * np.log1p(np.abs(xs)), 1.0 / (np.abs(xs) + 1.0)]

        Xr = np.column_stack(_host_basis(xa) + _host_basis(xb))
        Xr_scaled = StandardScaler().fit_transform(Xr)
        lr = LinearRegression().fit(Xr_scaled, fv)
        expected_resid = fv - lr.predict(Xr_scaled)

        gpu_resid = ug.gpu_additive_basis_residual(fv, xa, xb)
        np.testing.assert_allclose(gpu_resid, expected_resid, atol=1e-6)
