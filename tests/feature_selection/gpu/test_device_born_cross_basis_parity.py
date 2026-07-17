"""DEVICE-BORN orthogonal cross-basis FE parity + selection-equivalence (2026-06-30).

The pair / triplet / quadruplet / adaptive-arity cross-basis FE families build their engineered
``h_a * h_b [* h_c [* h_d]]`` product matrix on the HOST and, under STRICT residency, upload it at
``_orth_mi_backends.py:311``. ``_gpu_resident_cross_basis.build_leg_product_matrix_gpu`` rebuilds that matrix
DEVICE-BORN (per-leg basis columns via the resident batched Clenshaw evaluator + cupy elementwise products),
and the per-family scorers route both it and the raw / lower-arity baseline through the resident plug-in MI.

These tests pin the SELECTION-EQUIVALENCE hard gate per family:

* device product matrix == host product matrix within ~5e-6 under the default MLFRAME_CRIT_DTYPE_RELAXED=1
  (2026-07-13): both host (_orth_pair_cross_fe.py / _orthogonal_{triplet,quadruplet}_fe.py) and device
  (_gpu_resident_cross_basis.py) now operate on the operand at ``_crit_np_dtype()`` -- f32 by default -- so
  the polynomial recurrence runs at the SAME precision on both sides instead of the device silently
  upcasting to float64 underneath a nominally-relaxed host. Measured worst case across pair/triplet/
  quadruplet x {hermite,auto,legendre,chebyshev} x degree {1,2}: 2.52e-6 (quadruplet, degree 2 -- more
  multiplied legs compounds the per-leg f32 rounding). 5e-6 gives margin above that without masking a real
  regression. Under a non-relaxed MLFRAME_CRIT_DTYPE_RELAXED=0 host, both sides run in float64 and the gap
  collapses to the algorithm-only ~1e-12 (device backward-Clenshaw vs host forward recurrence; laguerre is
  forward on both so bit-consistent); AND
* the resident-scored selection (top winner + the ranking the family consumes) == the host selection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Need cuda."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _strict(monkeypatch):
    """Helper that strict."""
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")


# ---------------------------------------------------------------------------
# Device product matrix == host product matrix within ~5e-6 under relaxed dtype (all four families)
# ---------------------------------------------------------------------------


def _host_pair_matrix(X, pairs, max_degree, basis):
    """Host pair matrix."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
        generate_pair_cross_basis_features,
    )

    return generate_pair_cross_basis_features(X, pairs, max_degree=max_degree, basis=basis)


@pytest.mark.parametrize("basis", ["hermite", "auto", "legendre", "chebyshev"])
@pytest.mark.parametrize("max_degree", [1, 2])
def test_pair_device_matrix_matches_host(basis, max_degree, monkeypatch):
    """Pair device matrix matches host."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
        _pair_device_col_specs,
    )
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._gpu_resident_cross_basis import (
        build_leg_product_matrix_gpu,
    )

    rng = np.random.default_rng(20260630)
    n = 4000
    X = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.uniform(-3, 3, n),
            "c": rng.gamma(2.0, 1.0, n),
        }
    )
    pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    host = _host_pair_matrix(X, pairs, max_degree, basis)
    assert host.shape[1] > 0
    specs = _pair_device_col_specs(host.columns, list(X.columns), nbins=10)
    assert specs is not None
    mat = build_leg_product_matrix_gpu(cp, X, specs, basis=basis)
    dev = cp.asnumpy(mat)
    h = host.to_numpy(dtype=np.float64)
    assert dev.shape == h.shape
    # relative tolerance ~5e-6 under the default relaxed (f32) dtype -- see module docstring for the
    # measured worst case and why (the products span several orders of magnitude at degree 2).
    denom = np.maximum(np.abs(h), 1.0)
    maxrel = float(np.max(np.abs(dev - h) / denom))
    assert maxrel < 5e-6, f"pair device/host product reldiff {maxrel:.3e} exceeds 5e-6 (basis={basis}, deg={max_degree})"


@pytest.mark.parametrize("basis", ["hermite", "auto"])
def test_triplet_device_matrix_matches_host(basis, monkeypatch):
    """Triplet device matrix matches host."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._orthogonal_triplet_fe import (
        generate_triplet_cross_basis_features,
        _triplet_device_col_specs,
    )
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._gpu_resident_cross_basis import (
        build_leg_product_matrix_gpu,
    )

    rng = np.random.default_rng(7)
    n = 4000
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
            "x4": rng.standard_normal(n),
        }
    )
    host = generate_triplet_cross_basis_features(
        X,
        [("x1", "x2", "x3"), ("x1", "x2", "x4")],
        max_degree=2,
        basis=basis,
    )
    specs = _triplet_device_col_specs(host.columns, list(X.columns))
    assert specs is not None
    dev = cp.asnumpy(build_leg_product_matrix_gpu(cp, X, specs, basis=basis))
    h = host.to_numpy(dtype=np.float64)
    denom = np.maximum(np.abs(h), 1.0)
    maxrel = float(np.max(np.abs(dev - h) / denom))
    assert maxrel < 5e-6, f"triplet device/host reldiff {maxrel:.3e} exceeds 5e-6 (basis={basis})"


@pytest.mark.parametrize("basis", ["hermite", "auto"])
def test_quadruplet_device_matrix_matches_host(basis, monkeypatch):
    """Quadruplet device matrix matches host."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._orthogonal_quadruplet_fe import (
        generate_quadruplet_cross_basis_features,
        _quadruplet_device_col_specs,
    )
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._gpu_resident_cross_basis import (
        build_leg_product_matrix_gpu,
    )

    rng = np.random.default_rng(13)
    n = 4000
    X = pd.DataFrame(
        {
            "x1": rng.standard_normal(n),
            "x2": rng.standard_normal(n),
            "x3": rng.standard_normal(n),
            "x4": rng.standard_normal(n),
            "x5": rng.standard_normal(n),
        }
    )
    host = generate_quadruplet_cross_basis_features(
        X,
        [("x1", "x2", "x3", "x4"), ("x1", "x2", "x3", "x5")],
        max_degree=1,
        basis=basis,
    )
    specs = _quadruplet_device_col_specs(host.columns, list(X.columns))
    assert specs is not None
    dev = cp.asnumpy(build_leg_product_matrix_gpu(cp, X, specs, basis=basis))
    h = host.to_numpy(dtype=np.float64)
    denom = np.maximum(np.abs(h), 1.0)
    maxrel = float(np.max(np.abs(dev - h) / denom))
    assert maxrel < 5e-6, f"quadruplet device/host reldiff {maxrel:.3e} exceeds 5e-6 (basis={basis})"


# ---------------------------------------------------------------------------
# Selection-equivalence: device-born scorer top winner + ranking == host
# ---------------------------------------------------------------------------


def _build_xor(seed, n=4000):
    """Build xor."""
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.standard_normal(n)})
    y = ((x1 * x2) + 0.02 * rng.standard_normal(n) > 0).astype(int)
    return X, y


def test_pair_scorer_selection_equivalent(monkeypatch):
    """Pair scorer selection equivalent."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_pair_cross_fe import (
        generate_pair_cross_basis_features,
        score_pair_cross_basis_by_mi_uplift,
    )

    X, y = _build_xor(42)
    eng = generate_pair_cross_basis_features(X, [("x1", "x2")], max_degree=2, basis="hermite")

    # Host path (device-born OFF).
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_CROSSBASIS", "0")
    host_sc = score_pair_cross_basis_by_mi_uplift(X[["x1", "x2"]], eng, y, basis="hermite")

    # Device-born path (STRICT-resident ON).
    _strict(monkeypatch)
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_CROSSBASIS", "1")
    dev_sc = score_pair_cross_basis_by_mi_uplift(X[["x1", "x2"]], eng, y, basis="hermite")

    assert list(dev_sc["engineered_col"]) == list(host_sc["engineered_col"]), "device-born pair ranking diverged from host"
    assert dev_sc.iloc[0]["engineered_col"] == "x1*x2__He1_He1"
    # engineered_mi agrees to selection precision (both via the resident plug-in vs host njit -> ~1e-3 drift ok)
    hm = dict(zip(host_sc["engineered_col"], host_sc["engineered_mi"]))
    dm = dict(zip(dev_sc["engineered_col"], dev_sc["engineered_mi"]))
    for c in hm:
        assert abs(hm[c] - dm[c]) < 5e-2, f"engineered_mi for {c}: host {hm[c]:.4f} dev {dm[c]:.4f}"
