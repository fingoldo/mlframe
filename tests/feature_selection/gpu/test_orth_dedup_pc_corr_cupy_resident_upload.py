"""RESIDENT UPLOAD (wave 10): ``_pc_corr_cupy`` must upload ``Q``/``R`` (the partial-NaN candidate block and
its dense/partial comparison block from ``_dedup_collinear_source_cols``) via ``resident_operand`` instead
of a raw ``cp.asarray`` every call, so identical content recurring across the <=6 dedup calls/fit (each
independent FE family -- orth/extra-basis/gpu-resident/wavelet/hinge -- deduping its own candidate columns
against the SAME underlying source frame) uploads ONCE. LOW-tier, capped, numpy-is-default-backend finding
(cupy is force-selectable only via ``MLFRAME_FE_DEDUP_CORR_BACKEND=cupy``), so this test targets
``_pc_corr_cupy`` directly rather than the whole dedup pipeline.

Critically: Q/R must keep their CALLER dtype (f32 under ``MLFRAME_CRIT_DTYPE_RELAXED``) -- resident_operand
must NOT force float64, since that would change matmul precision, not just cache the upload.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_dedup import (
    _pc_corr_cupy,
    _pc_corr_numpy,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _blocks(q=5, r=6, n=400, seed=0, dtype=np.float64):
    """Helper that blocks."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(q, n)).astype(dtype)
    R = rng.normal(size=(r, n)).astype(dtype)
    # sprinkle some NaNs, mirroring the partial-NaN candidate block this backend is used for
    Q[0, ::17] = np.nan
    R[1, ::23] = np.nan
    return Q, R


def test_pc_corr_cupy_uploads_q_and_r_once_across_calls():
    """Two calls with the SAME (Q, R) content but as independent host objects (mirrors two FE families'
    dedup passes over the SAME source-column subset) must upload the Q-shaped and R-shaped arrays via
    cp.asarray only ONCE each."""
    import cupy as cp

    Q, R = _blocks(seed=1)
    Q2 = Q.copy()
    R2 = R.copy()

    upload_calls = {"q": 0, "r": 0}
    orig_asarray = cp.asarray

    def _counting(arr, *a, **kw):
        """Helper that counting."""
        shp = getattr(arr, "shape", None)
        if shp == Q.shape:
            upload_calls["q"] += 1
        elif shp == R.shape:
            upload_calls["r"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting
    try:
        out1 = _pc_corr_cupy(Q, R)
        out2 = _pc_corr_cupy(Q2, R2)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["q"] == 1, f"Q-shaped cp.asarray called {upload_calls['q']} times across 2 calls (expected 1)"
    assert upload_calls["r"] == 1, f"R-shaped cp.asarray called {upload_calls['r']} times across 2 calls (expected 1)"
    np.testing.assert_array_equal(out1, out2)


def test_pc_corr_cupy_different_content_is_a_plain_cache_miss():
    """When Q/R genuinely differ per call (the common case -- each FE family's candidate set differs), the
    fix must still upload fresh every time (a correctness-neutral cache miss, not a stale hit)."""
    import cupy as cp

    Q1, R1 = _blocks(seed=2)
    Q2, R2 = _blocks(seed=3)  # different content entirely

    upload_calls = {"q": 0}
    orig_asarray = cp.asarray

    def _counting(arr, *a, **kw):
        """Helper that counting."""
        if getattr(arr, "shape", None) == Q1.shape:
            upload_calls["q"] += 1
        return orig_asarray(arr, *a, **kw)

    cp.asarray = _counting
    try:
        out1 = _pc_corr_cupy(Q1, R1)
        out2 = _pc_corr_cupy(Q2, R2)
    finally:
        cp.asarray = orig_asarray

    assert upload_calls["q"] == 2, f"different-content Q must upload every call (got {upload_calls['q']}, expected 2)"
    assert not np.allclose(np.nan_to_num(out1), np.nan_to_num(out2))


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_pc_corr_cupy_bit_identical_to_prefix_raw_path(dtype):
    """The resident-upload path must be bit-identical to the pre-fix raw cp.asarray(Q)/cp.asarray(R) path,
    AND must preserve the caller's dtype (f32 under MLFRAME_CRIT_DTYPE_RELAXED) rather than forcing f64."""
    import cupy as cp

    Q, R = _blocks(seed=4, dtype=dtype)
    new_result = _pc_corr_cupy(Q, R)

    clear_fe_resident_operands()
    Qd = cp.asarray(Q)
    Rd = cp.asarray(R)
    assert Qd.dtype == dtype, f"resident upload changed Q dtype: expected {dtype}, got {Qd.dtype}"
    Qm = cp.isfinite(Qd)
    Rm = cp.isfinite(Rd)
    Q0 = cp.where(Qm, Qd, 0.0)
    R0 = cp.where(Rm, Rd, 0.0)
    Qmf = Qm.astype(cp.float64)
    Rmf = Rm.astype(cp.float64)
    n = Qmf @ Rmf.T
    Sx = Q0 @ Rmf.T
    Sy = Qmf @ R0.T
    Sxx = (Q0 * Q0) @ Rmf.T
    Syy = Qmf @ (R0 * R0).T
    Sxy = Q0 @ R0.T
    cov = Sxy - Sx * Sy / n
    vx = Sxx - Sx * Sx / n
    vy = Syy - Sy * Sy / n
    corr = cp.abs(cov / cp.sqrt(vx * vy))
    corr[(n < 8) | (vx <= 1e-24) | (vy <= 1e-24)] = cp.nan
    old_result = np.asarray(cp.asnumpy(corr))

    np.testing.assert_array_equal(np.nan_to_num(new_result, nan=-1.0), np.nan_to_num(old_result, nan=-1.0))


def test_pc_corr_cupy_matches_numpy_backend():
    """Sanity cross-check: the resident-cached cupy backend still agrees with the numpy(BLAS) backend."""
    Q, R = _blocks(seed=5)
    gpu = _pc_corr_cupy(Q, R)
    cpu = _pc_corr_numpy(Q, R)
    np.testing.assert_allclose(np.nan_to_num(gpu, nan=-1.0), np.nan_to_num(cpu, nan=-1.0), atol=1e-9, rtol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])


def test_resolve_pc_backend_prefers_cupy_under_strict(monkeypatch):
    """STRICT GPU mode routes the dedup correlation to the cupy backend (its 2.0-2.7x solo win is the
    point of strict mode's "carry FE compute on the device" contract); non-strict keeps the measured
    numpy default. The strict gate's own per-call work floor (p >= 64, n*p >= 1M) keeps tiny dedups on CPU."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_dedup as od

    monkeypatch.delenv("MLFRAME_FE_DEDUP_CORR_BACKEND", raising=False)
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    # Big enough to clear the strict per-call work floor (p=max(q,r)=500 >= 64; n*p = 25M >= 1M).
    assert od._resolve_pc_backend(500, 300, 50_000) == "cupy"
    # Tiny call: below the work floor -> falls through to the KTC/numpy default, NOT forced to cupy.
    assert od._resolve_pc_backend(4, 3, 100) != "cupy"
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "0")
    assert od._resolve_pc_backend(500, 300, 50_000) == "numpy"
    # Env force still wins over strict.
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_DEDUP_CORR_BACKEND", "numpy")
    assert od._resolve_pc_backend(500, 300, 50_000) == "numpy"
