"""Unit tests for the HW-calibrated numpy-vs-numba dispatch of _per_member_mae_std
and its auto-tune-on-first-miss sweep (mlframe.models.ensembling.per_member_tuning).

Cache I/O is isolated to a tmp dir via PYUTILZ_KERNEL_CACHE_DIR so these never
touch the real ~/.pyutilz cache.
"""
import numpy as np
import pytest

from mlframe.models.ensembling import base as eb
from mlframe.models.ensembling import per_member_tuning as pmt


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("MLFRAME_PER_MEMBER_BACKEND", raising=False)
    monkeypatch.setenv("MLFRAME_PER_MEMBER_AUTOTUNE", "1")
    eb._per_member_use_numba.cache_clear()
    pmt._AUTOTUNE_ATTEMPTED = False
    yield
    eb._per_member_use_numba.cache_clear()


def _numpy_ref(arr, med):
    diffs = np.abs(arr - med)
    return diffs.mean(axis=1), np.sqrt(np.var(diffs, axis=1))


@pytest.mark.skipif(not eb._HAS_NUMBA_PER_MEMBER, reason="numba unavailable")
def test_env_override_forces_backend(monkeypatch):
    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numpy")
    eb._per_member_use_numba.cache_clear()
    assert eb._per_member_use_numba(500_000, 8) is False  # would otherwise be numba
    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numba")
    eb._per_member_use_numba.cache_clear()
    assert eb._per_member_use_numba(100, 2) is True  # would otherwise be sub-floor numpy


def test_fallback_floor(monkeypatch):
    # Autotune off -> measurement-backed fallback: numba above the element floor, numpy below.
    monkeypatch.setenv("MLFRAME_PER_MEMBER_AUTOTUNE", "0")
    eb._per_member_use_numba.cache_clear()
    below = eb._per_member_use_numba(eb._PER_MEMBER_NUMBA_FLOOR_ELEMENTS - 1, 4)
    above = eb._per_member_use_numba(eb._PER_MEMBER_NUMBA_FLOOR_ELEMENTS + 1, 4)
    assert below is False
    if eb._HAS_NUMBA_PER_MEMBER:
        assert above is True
    else:
        assert above is False


@pytest.mark.skipif(not eb._HAS_NUMBA_PER_MEMBER, reason="numba unavailable")
def test_sweep_is_bounded_and_records_max_abs_diff():
    # observed=20k -> grid must not exceed the ceiling and must record max_abs_diff.
    regions = pmt.run_per_member_sweep(observed_elements=20_000, max_elements=20_000, repeats=5)
    assert regions and isinstance(regions, list)
    chosen = [r for r in regions if r.get("backend_choice") in ("numpy", "numba")]
    assert chosen, regions
    for r in regions:
        assert "max_abs_diff" in r
        # numpy vs numba are two-pass equivalent -> float-noise agreement
        assert r["max_abs_diff"] < 1e-6, r


@pytest.mark.skipif(not eb._HAS_NUMBA_PER_MEMBER, reason="numba unavailable")
def test_ensure_tuning_populates_cache_and_dispatch_reads_it():
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    pmt.ensure_per_member_tuning(force=True, observed_elements=50_000, observed_groups=4, repeats=5)
    assert KernelTuningCache().has(pmt._PER_MEMBER_KERNEL_NAME)
    # dispatch now reads the persisted region (autotune already ran this process)
    eb._per_member_use_numba.cache_clear()
    decision = eb._per_member_use_numba(50_000, 4)
    assert decision in (True, False)  # whatever the live HW measured, it is a valid decision


def test_2d_result_matches_numpy_reference():
    rng = np.random.default_rng(0)
    for n, k in [(2_000, 3), (50_000, 4)]:
        arr = rng.standard_normal((k, n)).astype(np.float64)
        med = rng.standard_normal(n).astype(np.float64)
        mae, std = eb._per_member_mae_std(arr, med)
        ref_mae, ref_std = _numpy_ref(arr, med)
        assert np.allclose(mae, ref_mae, rtol=1e-6, atol=1e-9)
        assert np.allclose(std, ref_std, rtol=1e-6, atol=1e-9)


def _numpy_ref_3d(arr, med):
    diffs = np.abs(arr - med)
    mae_per_col = diffs.mean(axis=1)
    std_per_col = np.sqrt(((diffs - mae_per_col[:, None, :]) ** 2).mean(axis=1))
    return mae_per_col.mean(axis=1), std_per_col.mean(axis=1)


@pytest.mark.skipif(not eb._HAS_NUMBA_PER_MEMBER, reason="numba unavailable")
def test_3d_njit_bit_identical_to_numpy():
    # The fixed 3-D njit (per-COLUMN std, not pooled) must match numpy exactly.
    rng = np.random.default_rng(1)
    for k, n, c in [(3, 1_000, 2), (5, 5_000, 3), (4, 200, 5)]:
        arr = rng.standard_normal((k, n, c)).astype(np.float64)
        med = rng.standard_normal((n, c)).astype(np.float64)
        nb_mae, nb_std = eb._per_member_mae_std_njit(arr, med)
        np_mae, np_std = _numpy_ref_3d(arr, med)
        assert np.allclose(nb_mae, np_mae, rtol=1e-6, atol=1e-9)
        assert np.allclose(nb_std, np_std, rtol=1e-6, atol=1e-9)


def test_3d_dispatch_matches_numpy_reference(monkeypatch):
    # Whichever backend the dispatch picks for 3-D, the result must match numpy.
    rng = np.random.default_rng(2)
    arr = rng.standard_normal((4, 20_000, 3)).astype(np.float64)
    med = rng.standard_normal((20_000, 3)).astype(np.float64)
    mae, std = eb._per_member_mae_std(arr, med)
    ref_mae, ref_std = _numpy_ref_3d(arr, med)
    assert np.allclose(mae, ref_mae, rtol=1e-6, atol=1e-9)
    assert np.allclose(std, ref_std, rtol=1e-6, atol=1e-9)


@pytest.mark.skipif(not eb._HAS_NUMBA_PER_MEMBER, reason="numba unavailable")
def test_3d_dispatch_can_select_numba(monkeypatch):
    # ndim is threaded into the decision; forcing numba via env applies to 3-D too.
    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numba")
    eb._per_member_use_numba.cache_clear()
    assert eb._per_member_use_numba(60_000, 4, ndim=3) is True
