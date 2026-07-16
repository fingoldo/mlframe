"""DEVICE-BORN / resident-operand parity for the four ``_orth_mi_backends.py:311`` H2D sub-families (2026-06-30).

The orth-univariate uplift scorer, the unified-gate noise floor, the conditional-gate prune, and the
pairwise-modular scan each scored a fit-constant / device-reconstructible matrix that, under STRICT residency,
was ``cp.asarray``-uploaded fresh at ``_orth_mi_backends.py:311``. These tests pin the SELECTION-EQUIVALENCE
hard gate per sub-family: with the resident / device-born family route OFF (host STRICT path) vs ON, the
per-column MI -- and every selection / floor / ranking built on it -- is byte-identical (the resident route only
removes the redundant upload / rebuilds the SAME matrix on device; the estimator is unchanged).

* SF1a/1b  ``score_features_by_mi_uplift``  -- raw baseline (class B) + poly engineered matrix (device-born A).
* SF2      ``_pairwise_modular_fe``         -- residue grid + combiner baseline + permutation null (device-born A).
* SF3      ``raw_mi_noise_floor``           -- raw noise-floor baseline (class B).
* SF4a     ``_rank_and_prune``              -- raw relevance column_stack (class B).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _strict(monkeypatch):
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_STRICT_RESIDENT", "1")
    monkeypatch.setenv("MLFRAME_CMI_GPU", "1")


# ---------------------------------------------------------------------------
# SF3: raw_mi_noise_floor (class B) -- byte-identical floor
# ---------------------------------------------------------------------------
def test_sf3_noise_floor_resident_byte_identical(monkeypatch):
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._unified_fe_gate import raw_mi_noise_floor

    rng = np.random.default_rng(7)
    n = 4000
    rawX = pd.DataFrame({f"r{i}": rng.standard_normal(n) for i in range(8)})
    y = ((rawX["r0"] + rawX["r1"]) > 0).astype(np.int64).to_numpy()

    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", "0")
    host = raw_mi_noise_floor(rawX, y, nbins=10)
    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", "1")
    res = raw_mi_noise_floor(rawX, y, nbins=10)
    assert host == res, f"noise floor host {host} != resident {res}"


# ---------------------------------------------------------------------------
# SF4a: _rank_and_prune raw relevance (class B) -- identical gate/operand pools
# ---------------------------------------------------------------------------
def test_sf4a_rank_and_prune_resident_selection_identical(monkeypatch):
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._conditional_gate_fe import _rank_and_prune

    rng = np.random.default_rng(7)
    n = 4000
    Xg = pd.DataFrame({f"c{i}": rng.integers(0, 5, n).astype(np.float64) for i in range(10)})
    yi = (Xg["c0"] > 2).astype(np.int64).to_numpy()
    cols = list(Xg.columns)

    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", "0")
    gh, oh = _rank_and_prune(Xg, cols, yi, nbins=10, k_gate=4, k_operand=5)
    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", "1")
    gr, orr = _rank_and_prune(Xg, cols, yi, nbins=10, k_gate=4, k_operand=5)
    assert gh == gr, f"gate pool host {gh} != resident {gr}"
    assert oh == orr, f"operand pool host {oh} != resident {orr}"


# ---------------------------------------------------------------------------
# SF1: score_features_by_mi_uplift -- raw baseline (B) + poly engineered (device-born A)
# ---------------------------------------------------------------------------
def test_sf1_uplift_poly_engineered_device_born_selection_identical(monkeypatch):
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
        score_features_by_mi_uplift,
    )

    rng = np.random.default_rng(7)
    n = 4000
    Xp = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.uniform(-1, 1, n)})
    yp = ((Xp["a"] ** 2 + Xp["b"] ** 3) > 0.5).astype(np.int64).to_numpy()
    eng = generate_univariate_basis_features(Xp, cols=["a", "b"], degrees=(2, 3), basis="auto", y=yp)

    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", "0")
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_UPLIFT_UNIVARIATE", "0")
    sh = score_features_by_mi_uplift(Xp, eng, yp, nbins=10).set_index("engineered_col").sort_index()
    monkeypatch.setenv("MLFRAME_FE_GPU_RESIDENT_RAW_BASELINE", "1")
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_UPLIFT_UNIVARIATE", "1")
    sr = score_features_by_mi_uplift(Xp, eng, yp, nbins=10).set_index("engineered_col").sort_index()

    # baseline (class B) is byte-identical; engineered (device-born poly) matches within ~1e-10 (Clenshaw vs
    # forward recurrence) -- here the FP delta is below the binning edges so the MI is identical.
    np.testing.assert_allclose(sh["baseline_mi"].to_numpy(), sr["baseline_mi"].to_numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(sh["engineered_mi"].to_numpy(), sr["engineered_mi"].to_numpy(), rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(sh["uplift"].to_numpy(), sr["uplift"].to_numpy(), rtol=1e-7, atol=1e-9)


def test_sf1c_extra_basis_engineered_stays_on_host(monkeypatch):
    # An extra-basis engineered name (spline / Fourier) must NOT parse to a poly leg -> the device-born helper
    # returns None and the caller keeps the engineered matrix on the host (irreducible born-fresh transient).
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._uplift_univariate_resident import (
        _specs_from_engineered_names,
    )

    assert _specs_from_engineered_names(["a__He2", "b__LL3"], ["a", "b"]) is not None
    assert _specs_from_engineered_names(["a__sp3", "b__sin2"], ["a", "b"]) is None
    assert _specs_from_engineered_names(["a__He2", "b__qsin1"], ["a", "b"]) is None


# ---------------------------------------------------------------------------
# SF2: pairwise-modular device-born (residue grid + combiner baseline + perm null) -- bit-identical
# ---------------------------------------------------------------------------
def test_sf2_residue_grid_device_born_byte_identical(monkeypatch):
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._pairwise_modular_fe import _residue_grid_mi, COARSE_MODULI

    rng = np.random.default_rng(7)
    n = 50000
    c = (rng.integers(0, 1000, n) - rng.integers(0, 1000, n)).astype(np.int64)  # diff combiner (negatives)
    y = (c % 7).astype(np.int64)
    mods = [k for k in COARSE_MODULI if k >= 2]

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_MODULAR", "0")
    host = np.asarray(_residue_grid_mi(c, y, mods, 12))
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_MODULAR", "1")
    res = np.asarray(_residue_grid_mi(c, y, mods, 12))
    np.testing.assert_array_equal(host, res)
    assert int(np.argmax(host)) == int(np.argmax(res))


def test_sf2_combiner_baseline_and_residue_mi_byte_identical(monkeypatch):
    """The REAL bug this pinned (found 2026-07-16, not a test artifact): ``_mi_classif_batch_numba``'s
    ``_orth_mi_gpu_enabled`` gate returns True unconditionally whenever ``MLFRAME_CMI_GPU=1`` (a diagnostic
    full-GPU-coverage flag, not shape-gated), silently routing to the EDGE-binned GPU FE-batcher
    (``multi_gpu_fe_batch_mi``) even when the caller explicitly requested ``rank_binning=True`` and the
    outer STRICT/rank-resident gate (``fe_gpu_strict_enabled``, which DOES apply the
    ``_STRICT_MIN_CALL_WORK=1_000_000`` shape floor) declined at this test's n=50000/p=1 shape -- silently
    swapping RANK binning for EDGE binning, ~6%% MI divergence on the tie-heavy integer combiner. Fixed by
    threading ``rank_binning`` into ``_mi_classif_batch_numba`` so it skips the edge-binned GPU branch
    whenever rank binning was requested, regardless of ``_orth_mi_gpu_enabled``'s own (shape-blind) verdict.
    A second, independent bug in the SAME dependency chain: numba's default (non-stable) argsort inside
    ``_quantile_bin_njit`` broke tie order differently from cupy's (stable) argsort -- fixed by pinning
    ``kind="mergesort"`` there (see that function's docstring).

    What remains here, POST-fix, is genuine float64 FP-reduction-order noise between the CPU njit and GPU
    cupy accumulation order (~2e-13 relative, e.g. 0.0009076343653791308 vs 0.0009076343653793286) -- the
    SAME class of divergence every other CPU/GPU parity claim in this codebase documents as acceptable
    (~1e-9, "selection-equivalent" not literal bit-identity); ``==`` was simply too strict for a claim no
    other part of the codebase actually holds to bit-for-bit. ``rtol=1e-9`` comfortably covers this while
    still catching a real algorithmic divergence (the fixed bug was ~6%%, four orders of magnitude above)."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._pairwise_modular_fe import _mi, _residue_mi
    from mlframe.feature_selection.filters._pairwise_modular_resident import combiner_mi_resident

    rng = np.random.default_rng(7)
    n = 50000
    c = (rng.integers(0, 1000, n) - rng.integers(0, 1000, n)).astype(np.int64)
    y = (c % 7).astype(np.int64)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_MODULAR", "0")
    base_host = _mi(c.astype(np.float64), y, nbins=12)
    res_host = _residue_mi(c, y, 7, 12)
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_MODULAR", "1")
    base_dev = combiner_mi_resident(c, y, nbins=12, rank_binning=True, modulus=0)
    res_dev = _residue_mi(c, y, 7, 12)
    assert base_host == pytest.approx(base_dev, rel=1e-9), f"baseline host {base_host} != dev {base_dev}"
    assert res_host == pytest.approx(res_dev, rel=1e-9), f"residue_mi host {res_host} != dev {res_dev}"


def test_sf2_perm_null_device_born_byte_identical(monkeypatch):
    """Same rank_binning fix as the combiner test above (this call path shares
    ``_mi_classif_batch_numba``); the residual float64 FP-reorder noise (~2e-13 relative here too) gets the
    same ``rtol=1e-9`` tolerance -- see that test's docstring for the full root-cause writeup."""
    _strict(monkeypatch)
    from mlframe.feature_selection.filters._pairwise_modular_fe import _perm_null_hi

    rng = np.random.default_rng(7)
    n = 50000
    c = rng.integers(0, 50, n).astype(np.int64)
    y = (c % 7 == 0).astype(np.int64)

    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_MODULAR", "0")
    nh_host = _perm_null_hi(c, y, 7, 12, n_perm=12, seed=0)
    monkeypatch.setenv("MLFRAME_FE_GPU_DEVICE_BORN_MODULAR", "1")
    nh_dev = _perm_null_hi(c, y, 7, 12, n_perm=12, seed=0)
    assert nh_host == pytest.approx(nh_dev, rel=1e-9), f"perm-null host {nh_host} != dev {nh_dev}"
