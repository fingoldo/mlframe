"""Unit coverage for ``_usability_gpu.py``'s gated cupy usability-scoring primitives.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this module had zero test references anywhere in
the suite. Pins the gate contract (``fe_gpu_usability_enabled``) fully offline, plus numeric parity of
the cupy primitives against their numpy equivalents on a real CUDA device when available.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mlframe.feature_selection.filters import _usability_gpu


def _need_cuda() -> bool:
    """Whether a usable CUDA device is present this process."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


def teardown_function():
    """Never leak the usability GPU gate env var into another test."""
    os.environ.pop("MLFRAME_FE_GPU_USABILITY", None)


def _abscorr_numpy(u, v):
    """Reference numpy implementation mirroring the CPU ``_abscorr`` this module twins."""
    u = np.asarray(u, dtype=np.float64).ravel()
    v = np.asarray(v, dtype=np.float64).ravel()
    if u.std() < 1e-12 or v.std() < 1e-12:
        return 0.0
    um, vm = u - u.mean(), v - v.mean()
    ssu, ssv = float((um * um).sum()), float((vm * vm).sum())
    if ssu <= 0.0 or ssv <= 0.0:
        return 0.0
    r = float((um * vm).sum()) / float(np.sqrt(ssu * ssv))
    return abs(r) if np.isfinite(r) else 0.0


def test_disabled_when_cupy_unavailable(monkeypatch):
    """Even with the env flag on, an absent cupy install must keep the gate OFF -- it's the primary
    'no GPU path at all' guard."""
    monkeypatch.setattr(_usability_gpu, "_CUPY_AVAIL", False)
    os.environ["MLFRAME_FE_GPU_USABILITY"] = "1"
    assert _usability_gpu.fe_gpu_usability_enabled() is False


def test_disabled_by_default_without_env_flag(monkeypatch):
    """The gate defaults OFF: even with cupy importable, no ``MLFRAME_FE_GPU_USABILITY`` set means the
    proven CPU numpy/sklearn path stays in charge."""
    monkeypatch.setattr(_usability_gpu, "_CUPY_AVAIL", True)
    os.environ.pop("MLFRAME_FE_GPU_USABILITY", None)
    assert _usability_gpu.fe_gpu_usability_enabled() is False


@pytest.mark.parametrize("falsey", ["0", "false", "off", "no", ""])
def test_disabled_for_falsey_env_values(monkeypatch, falsey):
    """Common falsey spellings of the env flag must all resolve to disabled, not just an empty string."""
    monkeypatch.setattr(_usability_gpu, "_CUPY_AVAIL", True)
    os.environ["MLFRAME_FE_GPU_USABILITY"] = falsey
    assert _usability_gpu.fe_gpu_usability_enabled() is False


@pytest.mark.parametrize("truthy", ["1", "true", "True", "on", "ON", "yes"])
def test_enabled_for_truthy_env_values_when_not_globally_disabled(monkeypatch, truthy):
    """Common truthy spellings all enable the gate when cupy is present and the global GPU off-switch is
    not engaged."""
    monkeypatch.setattr(_usability_gpu, "_CUPY_AVAIL", True)
    os.environ["MLFRAME_FE_GPU_USABILITY"] = truthy
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: False)
    assert _usability_gpu.fe_gpu_usability_enabled() is True


def test_disabled_when_globally_disabled(monkeypatch):
    """The global GPU off-switch (``gpu_globally_disabled``) must veto this gate even when the local env
    flag is set and cupy is importable -- a single master kill-switch for all GPU FE paths."""
    monkeypatch.setattr(_usability_gpu, "_CUPY_AVAIL", True)
    os.environ["MLFRAME_FE_GPU_USABILITY"] = "1"
    monkeypatch.setattr("mlframe.feature_selection.filters._gpu_policy.gpu_globally_disabled", lambda: True)
    assert _usability_gpu.fe_gpu_usability_enabled() is False


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_gpu_abscorr_matches_numpy_reference():
    """``gpu_abscorr`` must match the CPU ``_abscorr`` reference to float64 precision on informative,
    correlated data."""
    pytest.importorskip("cupy")
    rng = np.random.default_rng(0)
    u = rng.normal(size=2000)
    v = 0.7 * u + 0.3 * rng.normal(size=2000)
    got = _usability_gpu.gpu_abscorr(u, v)
    want = _abscorr_numpy(u, v)
    assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_gpu_abscorr_zero_std_column_returns_zero():
    """A constant column has std < 1e-12 -- the guard must return exactly 0.0, matching the CPU path's
    degenerate-input contract (not NaN, not an exception)."""
    pytest.importorskip("cupy")
    u = np.full(500, 3.0)
    v = np.random.default_rng(1).normal(size=500)
    assert _usability_gpu.gpu_abscorr(u, v) == 0.0


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_gpu_abscorr_batch_matches_per_column_numpy_reference():
    """``gpu_abscorr_batch`` scored against a multi-column pool must match the per-column numpy
    reference for every column, including a mix of informative and constant (degenerate) columns."""
    pytest.importorskip("cupy")
    rng = np.random.default_rng(2)
    n = 1500
    v = rng.normal(size=n)
    informative = 0.6 * v + 0.4 * rng.normal(size=n)
    noise = rng.normal(size=n)
    constant = np.full(n, 5.0)
    cols = np.column_stack([informative, noise, constant])
    got = _usability_gpu.gpu_abscorr_batch(cols, v)
    want = np.array([_abscorr_numpy(cols[:, j], v) for j in range(cols.shape[1])])
    assert got.shape == (3,)
    np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_gpu_abscorr_batch_empty_pool_returns_empty_array():
    """A zero-column pool is a well-defined degenerate case: an empty ``(0,)`` result, not a crash."""
    pytest.importorskip("cupy")
    n = 200
    v = np.random.default_rng(3).normal(size=n)
    cols = np.zeros((n, 0))
    result = _usability_gpu.gpu_abscorr_batch(cols, v)
    assert result.shape == (0,)


@pytest.mark.gpu
@pytest.mark.skipif(not _need_cuda(), reason="no CUDA")
def test_gpu_additive_basis_residual_matches_sklearn_reference():
    """``gpu_additive_basis_residual`` (mean-centered-OLS) must match an equivalent
    StandardScaler+LinearRegression residual on the CPU to float64 precision -- the documented
    equivalence this module's docstring relies on."""
    pytest.importorskip("cupy")
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(4)
    n = 1000
    xa = rng.normal(size=n)
    xb = rng.normal(size=n)
    fv = 0.5 * xa + 0.2 * xb**2 + 0.1 * rng.normal(size=n)

    def _basis_np(x):
        """Numpy reference twin of the module's cupy ``_basis`` inner helper."""
        xs = (x - x.mean()) / (x.std() + 1e-12)
        return [xs, xs**2, xs**3, np.sign(xs) * np.sqrt(np.abs(xs)), np.sign(xs) * np.log1p(np.abs(xs)), 1.0 / (np.abs(xs) + 1.0)]

    X = np.column_stack(_basis_np(xa) + _basis_np(xb))
    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(X, fv)
    want_resid = fv - model.predict(X)

    got_resid = _usability_gpu.gpu_additive_basis_residual(fv, xa, xb)
    assert got_resid.shape == (n,)
    np.testing.assert_allclose(got_resid, want_resid, atol=1e-6)
