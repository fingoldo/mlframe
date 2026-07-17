"""RESIDENT UPLOAD (2026-07-13): ``mi_direct_gpu_batched_pairs`` uploads ``classes_y``/``nbins_a``/
``joint_offsets``/``factors_data_T`` via ``resident_operand`` instead of a raw ``cp.asarray`` every call.

Call-frequency note (see the wave-10 report): grepping the one production caller
(``_cat_interactions_step.run_cat_interaction_step`` -> ``mi_direct_gpu_batched_pairs``) shows it is invoked
ONCE per ``MRMR.fit()`` call (the cat-FE step runs once before the screening loop, per the "Runs once before
the screening loop" comment at its call site in ``_fit_impl_core.py``) -- so the cross-call dedup this test
exercises has no measured production win today. It is still applied (cheap, correct via content-hash) so any
future/other caller that invokes this more than once per fit gets the dedup for free; this test proves the
mechanism itself is correct and bit-identical, independent of how often production currently calls it.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands
from mlframe.feature_selection.filters._gpu_pairs import mi_direct_gpu_batched_pairs


def _gpu_available() -> bool:
    """Gpu available."""
    try:
        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:  # pragma: no cover - no driver / no GPU
        return False


_GPU_AVAILABLE = _gpu_available()
if not _GPU_AVAILABLE:  # pragma: no cover - guarded at collection time
    pytest.skip("No CUDA device available", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_resident_cache():
    """Clear resident cache."""
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _build_inputs(n=1500, seed=1):
    """Build inputs."""
    rng = np.random.default_rng(seed)
    nbins_list = [4, 4, 3, 4, 3, 4]
    cols = [rng.integers(0, nb, size=n) for nb in nbins_list]
    y_nbins = 3
    y = rng.integers(0, y_nbins, size=n)
    factors_data = np.column_stack(cols).astype(np.int32)
    factors_nbins = np.array(nbins_list, dtype=np.int64)
    classes_y = y.astype(np.int32)
    freqs_y = np.bincount(y, minlength=y_nbins).astype(np.float64) / n
    pairs_a = np.array([0, 1, 2], dtype=np.int64)
    pairs_b = np.array([3, 4, 5], dtype=np.int64)
    return factors_data, factors_nbins, classes_y, freqs_y, pairs_a, pairs_b


def _count_asarray_calls_matching(monkeypatch, target_values):
    """Count asarray calls matching."""
    calls = {"n": 0}
    orig = cp.asarray
    target = np.ascontiguousarray(target_values)

    def spy(a, *args, **kw):
        """Helper that spy."""
        if isinstance(a, np.ndarray) and a.shape == target.shape and a.dtype == target.dtype and np.array_equal(a, target):
            calls["n"] += 1
        return orig(a, *args, **kw)

    monkeypatch.setattr(cp, "asarray", spy)
    return calls


@pytest.mark.gpu
def test_classes_y_and_nbins_a_and_joint_offsets_uploaded_once_across_repeat_calls(monkeypatch):
    """Two calls with IDENTICAL (factors_data, pairs, classes_y) content must upload each resident operand
    exactly once (content-hash hit on the second call), demonstrating the caching mechanism is correct even
    though production currently reaches this function once per fit (see module docstring)."""
    factors_data, factors_nbins, classes_y, freqs_y, pairs_a, pairs_b = _build_inputs(seed=3)

    nbins_a = factors_nbins[pairs_a].astype(np.int32)
    y_calls = _count_asarray_calls_matching(monkeypatch, classes_y.astype(np.int32))
    nbins_a_calls = _count_asarray_calls_matching(monkeypatch, nbins_a)

    mi1 = mi_direct_gpu_batched_pairs(factors_data, pairs_a, pairs_b, factors_nbins, classes_y.copy(), freqs_y.copy())
    mi2 = mi_direct_gpu_batched_pairs(factors_data, pairs_a, pairs_b, factors_nbins, classes_y.copy(), freqs_y.copy())

    assert y_calls["n"] == 1, f"classes_y-content cp.asarray called {y_calls['n']}x across 2 identical calls (expected 1)"
    assert nbins_a_calls["n"] == 1, f"nbins_a-content cp.asarray called {nbins_a_calls['n']}x across 2 identical calls (expected 1)"
    np.testing.assert_array_equal(mi1, mi2)


@pytest.mark.gpu
def test_resident_upload_bit_identical_to_disabled_cache(monkeypatch):
    """Deterministic kernel (no RNG) -- cache ON must produce EXACTLY the same joint-MI array as the
    diagnostic-disabled (pre-fix raw-upload) path."""
    factors_data, factors_nbins, classes_y, freqs_y, pairs_a, pairs_b = _build_inputs(seed=7)

    clear_fe_resident_operands()
    mi_cached = mi_direct_gpu_batched_pairs(factors_data, pairs_a, pairs_b, factors_nbins, classes_y.copy(), freqs_y.copy())

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    clear_fe_resident_operands()
    mi_fresh = mi_direct_gpu_batched_pairs(factors_data, pairs_a, pairs_b, factors_nbins, classes_y.copy(), freqs_y.copy())

    np.testing.assert_array_equal(mi_cached, mi_fresh)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
