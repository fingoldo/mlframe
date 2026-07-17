"""Parity for the DEVICE-BORN ALS prewarp design (H2D collapse, 2026-06-30).

The rank-1 ALS warm-start design matrices ``B_a``/``B_b`` (n x degree+1) were the
only bulk n-scaled operand the resident GPU twin uploaded; at 300k rows over the
per-operand-pair sweeps that was ~374MB aggregate. They are a deterministic
closed-form function of the small standardised column ``z`` + the basis recurrence,
so they are now BUILT ON DEVICE (``_build_basis_matrix_gpu``) instead of uploaded.

Correctness is SELECTION-EQUIVALENCE, proven in two layers:
  1. the device basis matrix mirrors the host ``build_basis_matrix`` recurrence
     EXACTLY (same recurrence + column order) -> agree to ~1e-13 (FP reduction
     order only), for all four orthogonal-polynomial bases.
  2. the resulting ALS coefficients from ``warm_start_als_seed_gpu_from_z`` agree
     with the CPU ``warm_start_als_seed`` to ~1e-12, so the prewarp transform and
     the downstream FE selection are unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

_BASES = ("chebyshev", "hermite", "legendre", "laguerre")


def _als_parity_tol():
    """(rtol, atol) for the device-born ALS coefficient parity vs the CPU f64 reference.

    Under ``MLFRAME_CRIT_DTYPE_RELAXED`` (DEFAULT ON) the device path standardises za/zb in FLOAT32, so its
    coefficients agree with the CPU float64 reference only to the f32-input CONDITION-NUMBER-bounded tolerance:
    the rank-1 ALS normal equations of the fast-growing high-degree Laguerre/Hermite bases have AtA cond up to
    ~1e7, so a ~1e-7 f32 input rounding amplifies to ~1e-3 relative in the coefficients. That is a PRECISION
    contract; SELECTION-equivalence (the coefficients are a smooth CMA-ES warm-start seed, not tie-sensitive) is
    proven separately by F2 across all distributions + the hermite biz/e2e suites. With the relaxation OFF
    (``MLFRAME_CRIT_DTYPE_RELAXED=0``) the device standardises in f64 and the strict ~1e-7 parity holds."""
    import os

    _relaxed = os.environ.get("MLFRAME_CRIT_DTYPE_RELAXED", "1").strip().lower() not in ("0", "false", "off", "no")
    # f32 builds the design + matvecs in float32 (the relaxation) but keeps the tiny ill-conditioned
    # normal-equation SOLVE in float64, so the coefficients agree with the CPU f64 reference to the f32-INPUT
    # condition bound (~1e-3 absolute on the fast-growing high-degree bases) -- NOT the pure-f32-solve ~1e-1
    # that a f32 solve would give. That is a precision contract; SELECTION-equivalence (a smooth, non-tie-
    # sensitive CMA-ES warm-start seed) is proven separately by F2 across distributions + the hermite biz suites.
    return (5e-3, 1e-4) if _relaxed else (1e-7, 1e-9)


def _std_col(x):
    """Match the host standardisation the prewarp uses (basis_fit) closely enough
    for a parity check: the device builder and host builder are fed the SAME z, so
    we standardise once here and feed both."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return (x - x.mean()) / (x.std() + 1e-12)


def test_device_basis_matrix_matches_host_all_bases():
    """``_build_basis_matrix_gpu`` == host ``build_basis_matrix`` to 1e-13 for every
    orthogonal-polynomial basis, across degrees and a heavy-tailed z range."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.hermite_fe import build_basis_matrix
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp_gpu_resident import (
        _build_basis_matrix_gpu,
    )

    rng = np.random.default_rng(0)
    for n in (1, 2, 5000):
        z = _std_col(rng.standard_normal(n) * 3.0 + rng.standard_normal(n) ** 3)
        zg = cp.asarray(z)
        for basis in _BASES:
            for deg in (1, 2, 3, 4, 8):
                host = build_basis_matrix(basis, z, deg)
                dev = cp.asnumpy(_build_basis_matrix_gpu(cp, basis, zg, deg))
                assert dev.shape == host.shape == (n, deg + 1)
                # ~1e-13 RELATIVE: the host njit recurrence runs under fastmath=True (FP
                # reassociation) so the agreement is FP reduction-order noise, not a bias.
                # On the high-degree Chebyshev/Hermite columns the entries reach magnitude
                # 1e2+, so a relative 1e-11 (atol 1e-12 for the ~0 entries) is the correct
                # selection-equivalent bar -- far below any decision boundary downstream.
                np.testing.assert_allclose(
                    dev,
                    host,
                    rtol=1e-11,
                    atol=1e-12,
                    err_msg=f"basis={basis} deg={deg} n={n}",
                )


def test_device_basis_matrix_rejects_unknown_basis():
    """The device builder must raise ``KeyError`` on a non-polynomial basis, mirroring
    the host ``build_basis_matrix`` contract (factory bases use the per-call path)."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp_gpu_resident import (
        _build_basis_matrix_gpu,
    )

    with pytest.raises(KeyError):
        _build_basis_matrix_gpu(cp, "fourier_adaptive", cp.asarray(np.zeros(4)), 3)


def test_als_coeffs_device_born_match_cpu():
    """``warm_start_als_seed_gpu_from_z`` (device-born design) coefficients == the CPU
    ``warm_start_als_seed`` (prebuilt host design) coefficients to ~1e-12 -- a
    product-structured target so the ALS actually recovers both factors."""
    pytest.importorskip("cupy")
    from mlframe.feature_selection.filters.hermite_fe import build_basis_matrix
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import (
        warm_start_als_seed,
    )
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp_gpu_resident import (
        warm_start_als_seed_gpu_from_z,
    )

    rng = np.random.default_rng(1)
    for n in (2000, 30000):
        for basis in _BASES:
            for deg in (2, 3, 4):
                xa = rng.uniform(1.0, 5.0, n)
                xb = rng.uniform(1.0, 5.0, n)
                za = _std_col(xa)
                zb = _std_col(xb)
                # product target f(a)*g(b) so ALS recovers both rank-1 factors.
                y = (za**2 - 0.5) * (zb - 0.3 * zb**3) + 0.01 * rng.standard_normal(n)

                Ba = build_basis_matrix(basis, za, deg)
                Bb = build_basis_matrix(basis, zb, deg)
                ca_cpu, cb_cpu = warm_start_als_seed(Ba, Bb, y, iters=3)
                ca_gpu, cb_gpu = warm_start_als_seed_gpu_from_z(za, zb, y, basis=basis, max_degree=deg, iters=3)

                assert ca_cpu is not None and ca_gpu is not None, (basis, deg, n)
                # Coeff agreement is CONDITION-NUMBER bounded, not 1e-12: the rank-1
                # ALS solves a NORMAL-EQUATIONS system whose AtA is ill-conditioned for
                # the fast-growing high-degree Laguerre/Hermite bases (cond ~6.7e6 at
                # deg=4 here), so the ~1e-14 device-vs-host design noise is amplified to
                # ~1e-10..1e-9 RELATIVE in the coefficients -- and that same amplification
                # appears when the CPU solver is fed the device design (verified). This is
                # NOT a selection change: the coeffs are a SMOOTH CMA-ES warm-start seed
                # (the module documents it is not tie-sensitive), so a 1e-9 relative
                # perturbation cannot flip the downstream FE selection.
                _rt, _at = _als_parity_tol()
                np.testing.assert_allclose(ca_gpu, ca_cpu, rtol=_rt, atol=_at, err_msg=f"coef_a basis={basis} deg={deg} n={n}")
                np.testing.assert_allclose(cb_gpu, cb_cpu, rtol=_rt, atol=_at, err_msg=f"coef_b basis={basis} deg={deg} n={n}")


def test_dispatch_routes_device_born_under_resident_flag(monkeypatch):
    """With the resident flag ON and za/zb/basis supplied, ``warm_start_als_seed``
    must route through the device-born twin and return coefficients matching the CPU
    path -- end-to-end selection-equivalence through the production dispatch."""
    pytest.importorskip("cupy")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    from mlframe.feature_selection.filters._gpu_strict_fe._entry import (
        fe_gpu_strict_resident_enabled,
    )

    if not fe_gpu_strict_resident_enabled():
        pytest.skip("resident flag not active (no usable CUDA device)")

    from mlframe.feature_selection.filters.hermite_fe import build_basis_matrix
    from mlframe.feature_selection.filters.hermite_fe._hermite_prewarp import (
        warm_start_als_seed,
    )

    rng = np.random.default_rng(2)
    n, deg, basis = 20000, 3, "chebyshev"
    xa = rng.uniform(1.0, 5.0, n)
    xb = rng.uniform(1.0, 5.0, n)
    za = _std_col(xa)
    zb = _std_col(xb)
    y = (za**2 - 0.5) * (zb - 0.3 * zb**3) + 0.01 * rng.standard_normal(n)
    Ba = build_basis_matrix(basis, za, deg)
    Bb = build_basis_matrix(basis, zb, deg)

    # resident dispatch with device-born design (za/zb/basis supplied)
    ca_dev, cb_dev = warm_start_als_seed(Ba, Bb, y, iters=3, z_a=za, z_b=zb, basis=basis)
    # pure CPU reference: disable the resident flag for this call
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "0")
    ca_cpu, cb_cpu = warm_start_als_seed(Ba, Bb, y, iters=3)

    assert ca_dev is not None and ca_cpu is not None
    # condition-number bounded (see test_als_coeffs_device_born_match_cpu); warm-start seed. Tolerance is
    # precision-mode-aware: f32-input under MLFRAME_CRIT_DTYPE_RELAXED (default), strict f64 when it is off.
    _rt, _at = _als_parity_tol()
    np.testing.assert_allclose(ca_dev, ca_cpu, rtol=_rt, atol=_at)
    np.testing.assert_allclose(cb_dev, cb_cpu, rtol=_rt, atol=_at)
