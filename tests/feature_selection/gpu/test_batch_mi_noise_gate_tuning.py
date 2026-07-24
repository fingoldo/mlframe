"""Unit coverage for ``_batch_mi_noise_gate_tuning.py``'s backend-choice / dispatch glue.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this module had zero test references anywhere in
the suite. Pins the pure fallback-heuristic logic, the synthetic-input generator's shape contract, and
the ``get_or_tune``-orchestrated backend choice (mocked cache, no real GPU sweep needed) -- the heavy
sweep/tuning path itself (``_run_batch_mi_noise_gate_sweep``) needs a real GPU host and stays out of
scope for closing this coverage gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _batch_mi_noise_gate_tuning as tuning


def test_make_inputs_shapes_match_requested_dims():
    """The synthetic input tuple's discretized matrix and label vector must match the requested
    (n_rows, n_cols) dims exactly -- this feeds both the sweep and the equivalence harness."""
    dims = {"n_rows": 50, "n_cols": 7}
    disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y, nperm, _seed, _mnc, use_su, dtype = tuning._make_batch_mi_noise_gate_inputs(dims)
    assert disc_2d.shape == (50, 7)
    assert factors_nbins.shape == (7,)
    assert classes_y.shape == (50,)
    assert classes_y_safe.shape == (50,)
    assert freqs_y.shape == (tuning._BMING_SWEEP_N_CLASSES_Y,)
    assert nperm == tuning._BMING_SWEEP_NPERM
    assert use_su is False
    assert dtype is np.int32


def test_make_inputs_discretized_values_within_nbins_range():
    """Every discretized value must be a valid bin index in ``[0, nbins)`` -- the kernel's contract."""
    disc_2d, *_ = tuning._make_batch_mi_noise_gate_inputs({"n_rows": 200, "n_cols": 4})
    assert disc_2d.min() >= 0
    assert disc_2d.max() < tuning._BMING_SWEEP_NBINS


def test_make_inputs_freqs_y_sums_to_one():
    """``freqs_y`` is a class-frequency distribution over the label space -- must sum to ~1."""
    _, _, _, _, freqs_y, *_ = tuning._make_batch_mi_noise_gate_inputs({"n_rows": 500, "n_cols": 2})
    assert freqs_y.sum() == pytest.approx(1.0)


def test_fallback_choice_cpu_below_both_thresholds():
    """Below both size thresholds, the fallback heuristic must pick CPU regardless of GPU availability."""
    assert tuning._batch_mi_noise_gate_fallback_choice(n_rows=100, n_cols=10) == "cpu"


def test_fallback_choice_cpu_when_rows_large_but_cols_small():
    """Large n_rows alone (K below GPU_MIN_COLS) must still route to CPU -- both thresholds gate together,
    not either one alone."""
    assert tuning._batch_mi_noise_gate_fallback_choice(n_rows=1_000_000, n_cols=10) == "cpu"


def test_fallback_choice_prefers_cupy_over_cuda(monkeypatch):
    """When both backends are large enough to route to GPU and both cupy/cuda are available, cupy is
    preferred (single batched bincount per shuffle vs per-column block launch)."""
    monkeypatch.setattr(tuning, "_CUPY_AVAIL", True)
    monkeypatch.setattr(tuning, "_CUDA_AVAIL", True)
    assert tuning._batch_mi_noise_gate_fallback_choice(n_rows=tuning.GPU_MIN_ROWS, n_cols=tuning.GPU_MIN_COLS) == "cupy"


def test_fallback_choice_falls_back_to_cuda_without_cupy(monkeypatch):
    """When cupy is unavailable but cuda (numba) is, and the size thresholds are met, cuda is chosen."""
    monkeypatch.setattr(tuning, "_CUPY_AVAIL", False)
    monkeypatch.setattr(tuning, "_CUDA_AVAIL", True)
    assert tuning._batch_mi_noise_gate_fallback_choice(n_rows=tuning.GPU_MIN_ROWS, n_cols=tuning.GPU_MIN_COLS) == "cuda"


def test_fallback_choice_cpu_when_no_gpu_backend_available(monkeypatch):
    """Even when the size thresholds are met, with neither cupy nor cuda available the fallback must be
    CPU (no GPU backend to route to)."""
    monkeypatch.setattr(tuning, "_CUPY_AVAIL", False)
    monkeypatch.setattr(tuning, "_CUDA_AVAIL", False)
    assert tuning._batch_mi_noise_gate_fallback_choice(n_rows=tuning.GPU_MIN_ROWS, n_cols=tuning.GPU_MIN_COLS) == "cpu"


class _FakeCache:
    """Minimal stand-in for ``KernelTuningCache`` returning a fixed ``get_or_tune`` result."""

    def __init__(self, result):
        self._result = result

    def get_or_tune(self, name, dims, tuner, axes, fallback, code_version):
        """Ignore the real tuner/cache mechanics and hand back the pre-baked result."""
        return self._result

    def get_regions(self, name):
        """No cached regions in this fake -- forces the caller down the tuning path."""
        return None

    def update(self, *args, **kwargs):
        """No-op persistence for the fake cache."""


def test_backend_choice_uses_get_or_tune_result(monkeypatch):
    """When the tuning cache's ``get_or_tune`` returns a concrete backend region, that value is used
    verbatim rather than falling back to the heuristic."""
    monkeypatch.setattr(
        "pyutilz.performance.kernel_tuning.cache.KernelTuningCache.load_or_create",
        classmethod(lambda cls: _FakeCache({"backend_choice": "cuda"})),
    )
    assert tuning._batch_mi_noise_gate_backend_choice(100, 4096) == "cuda"


def test_backend_choice_resolves_legacy_gpu_region(monkeypatch):
    """A stale pre-cupy/cuda-split 'gpu' region must resolve to whichever concrete GPU backend is
    actually available on this host, not be returned verbatim (it isn't one of the 3 valid dispatch
    values)."""
    monkeypatch.setattr(tuning, "_CUPY_AVAIL", True)
    monkeypatch.setattr(tuning, "_CUDA_AVAIL", False)
    monkeypatch.setattr(
        "pyutilz.performance.kernel_tuning.cache.KernelTuningCache.load_or_create",
        classmethod(lambda cls: _FakeCache({"backend_choice": "gpu"})),
    )
    assert tuning._batch_mi_noise_gate_backend_choice(100, 4096) == "cupy"


def test_backend_choice_falls_back_on_cache_exception(monkeypatch):
    """If the tuning-cache machinery raises for any reason (missing dependency, corrupt cache file), the
    call must degrade to the pure size-heuristic rather than propagate the exception."""

    def _raise_import(*args, **kwargs):
        """Simulate the tuning-cache machinery raising for any reason."""
        raise RuntimeError("cache unavailable")

    monkeypatch.setattr(
        "pyutilz.performance.kernel_tuning.cache.KernelTuningCache.load_or_create",
        classmethod(_raise_import),
    )
    monkeypatch.setattr(tuning, "_CUPY_AVAIL", False)
    monkeypatch.setattr(tuning, "_CUDA_AVAIL", False)
    assert tuning._batch_mi_noise_gate_backend_choice(10, 10) == "cpu"


def test_ensure_tuning_returns_cached_regions_without_force(monkeypatch):
    """With ``force=False`` and existing cached regions, ``ensure_batch_mi_noise_gate_tuning`` must return
    them immediately without re-running the (expensive) sweep."""

    class _CachedRegionsCache(_FakeCache):
        """Fake cache pre-loaded with a cached region, to verify no re-sweep happens."""

        def get_regions(self, name):
            """Return the pre-baked cached region for any name."""
            return [{"backend_choice": "cpu"}]

    monkeypatch.setattr(
        "pyutilz.performance.kernel_tuning.cache.KernelTuningCache.load_or_create",
        classmethod(lambda cls: _CachedRegionsCache(None)),
    )
    regions = tuning.ensure_batch_mi_noise_gate_tuning(force=False)
    assert regions == [{"backend_choice": "cpu"}]


def test_ensure_tuning_returns_none_when_sweep_fails(monkeypatch):
    """A sweep failure (e.g. no GPU on this host) must degrade to ``None``, not raise -- the CLI entry
    point reports a skip rather than crashing."""
    monkeypatch.setattr(
        "pyutilz.performance.kernel_tuning.cache.KernelTuningCache.load_or_create",
        classmethod(lambda cls: _FakeCache(None)),
    )
    monkeypatch.setattr(tuning, "_run_batch_mi_noise_gate_sweep", lambda: (_ for _ in ()).throw(RuntimeError("no GPU")))
    assert tuning.ensure_batch_mi_noise_gate_tuning(force=True) is None
