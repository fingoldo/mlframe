"""Critic-3 follow-up tests: cover the gaps the third audit flagged.

Targets:
* A1: ``_measure_single_region`` unit test.
* A2/D2/D3: streamed variant uses N>=2 streams + per-stream RNG independence
  (proxy: stack trace + monkeypatch counter).
* A3/A4: ``_FALLBACK_BY_CC`` returns the right entry for cc 7/8/9 under
  mocked ``gpu_capability_summary``.
* A5: ``ensure_joint_hist_tuning`` post-sweep JSON shape.
* A7: new RawKernel attributes wired by ``_ensure_kernels_inited``.

All tests gate on CUDA via ``pytest.importorskip("cupy")`` + ``is_cuda_available()``.
"""
from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda():
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _need_cuda(), reason="no CUDA")


# --------------------------------------------------------------------------
# A1: _measure_single_region unit test
# --------------------------------------------------------------------------

def test_measure_single_region_returns_well_formed_dict():
    """The new online-relearn helper must return a dict with all the
    expected axis_max keys, kernel_variant, block_size, and wall_ms."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import auto_tune as at

    region = at._measure_single_region(
        n_samples=50_000, joint_size=25, n_iters=2,
    )
    assert region is not None
    for key in ("n_samples_max", "joint_size_max", "nbins_x_max", "nbins_y_max",
                "kernel_variant", "block_size", "wall_ms"):
        assert key in region, f"missing {key} in {region}"
    assert region["kernel_variant"] in ("shared", "global")
    assert region["block_size"] in (256, 512, 1024)
    assert region["wall_ms"] > 0


# --------------------------------------------------------------------------
# A2 + D2 + D3: streamed variant uses N>=2 streams + per-stream RNG independence
# --------------------------------------------------------------------------

def test_streamed_uses_multiple_streams_and_independent_rngs():
    """Verify the bug-fix paths in mi_direct_gpu_batched_streamed are
    actually exercised: at least 2 cp.cuda.Stream instances created AND
    at least 2 cp.random.default_rng generators created."""
    from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_streamed

    n_streams_created = []
    n_generators_created = []
    real_stream = cp.cuda.Stream
    real_default_rng = cp.random.default_rng

    def _track_stream(*a, **kw):
        s = real_stream(*a, **kw)
        n_streams_created.append(s)
        return s

    def _track_rng(*a, **kw):
        g = real_default_rng(*a, **kw)
        n_generators_created.append(g)
        return g

    rng = np.random.default_rng(31)
    data = np.column_stack([
        rng.integers(0, 3, size=2000).astype(np.int32),
        rng.integers(0, 3, size=2000).astype(np.int32),
    ])
    nbins = np.array([3, 3], dtype=np.int32)

    with mock.patch.object(cp.cuda, "Stream", side_effect=_track_stream), \
         mock.patch.object(cp.random, "default_rng", side_effect=_track_rng):
        mi_direct_gpu_batched_streamed(
            data, (0,), (1,), nbins, npermutations=64, batch_size=32,
        )

    assert len(n_streams_created) >= 2, (
        f"streamed variant should create >=2 streams; got {len(n_streams_created)}"
    )
    assert len(n_generators_created) >= 2, (
        f"streamed variant should create >=2 per-stream RNGs; "
        f"got {len(n_generators_created)} (regression -- per-stream "
        f"RNG protection may have been reverted)"
    )


# --------------------------------------------------------------------------
# A3 / A4: _FALLBACK_BY_CC routes correctly per cc_major + cache-miss path
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cc_major,expected_block_size", [
    (5, 512),  # Maxwell
    (6, 512),  # Pascal
    (7, 256),  # Volta/Turing
    (8, 256),  # Ampere
    (9, 1024),  # Hopper -> global
])
def test_hw_aware_fallback_routes_per_cc_major(cc_major, expected_block_size):
    """Fallback hits the cc-major table when the kernel_tuning_cache is empty."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with mock.patch.object(
        dispatch, "_get_cache", return_value=False,
    ):
        with mock.patch(
            "pyutilz.system.gpu_dispatch.gpu_capability_summary",
            return_value={
                "cc_major": cc_major, "cc_minor": 0, "name": "MockGPU",
            },
        ):
            r = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=25)
    assert r["block_size"] == expected_block_size, (
        f"cc {cc_major} expected block_size={expected_block_size}, got {r}"
    )


def test_hw_aware_fallback_large_joint_routes_to_global():
    """At joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK the fallback prefers
    the global kernel regardless of cc_major."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with mock.patch.object(dispatch, "_get_cache", return_value=False):
        with mock.patch(
            "pyutilz.system.gpu_dispatch.gpu_capability_summary",
            return_value={"cc_major": 6, "cc_minor": 1, "name": "M"},
        ):
            r = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=8192)
    assert r["kernel_variant"] == "global"


# --------------------------------------------------------------------------
# A5: ensure_joint_hist_tuning saves a schema-valid JSON
# --------------------------------------------------------------------------

def test_ensure_joint_hist_tuning_saves_expected_schema(tmp_path, monkeypatch):
    """After ``ensure_joint_hist_tuning(force=True)`` the on-disk cache
    JSON must contain a ``joint_hist_batched`` kernel entry with the
    expected axes + region keys."""
    import json
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    from pyutilz.system.kernel_tuning_cache import hw_fingerprint
    hw_fingerprint.cache_clear()

    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import auto_tune as at

    # Trim sweep axes for speed -- 1 combo only.
    monkeypatch.setattr(at, "_N_SAMPLES_AXIS", (50_000,))
    monkeypatch.setattr(at, "_NBINS_AXIS", ((3, 3),))
    monkeypatch.setattr(at, "_BLOCK_SIZE_AXIS", (256,))

    regions = at.ensure_joint_hist_tuning(force=True)
    assert regions, "sweep returned no regions"

    # Read the persisted JSON.
    from pyutilz.system.kernel_tuning_cache import cache_path
    path = cache_path()
    assert os.path.isfile(path), f"sweep did not persist {path}"
    with open(path, "r") as f:
        data = json.load(f)
    assert data["schema_version"] == 1
    assert "joint_hist_batched" in data["kernels"]
    entry = data["kernels"]["joint_hist_batched"]
    assert entry["axes"] == ["n_samples", "joint_size"]
    for r in entry["regions"]:
        # Each region has the standard keys.
        for k in ("n_samples_max", "joint_size_max", "kernel_variant", "block_size"):
            assert k in r, f"region missing {k}: {r}"


# --------------------------------------------------------------------------
# A7: new RawKernel attributes wired by _ensure_kernels_inited
# --------------------------------------------------------------------------

def test_ensure_kernels_inited_populates_new_shared_kernels():
    """The init guard must populate the new shared-mem RawKernel attrs."""
    from mlframe.feature_selection.filters import gpu as g
    g._ensure_kernels_inited()
    # Pre-existing kernels.
    assert g.compute_joint_hist_cuda is not None
    assert g.compute_joint_hist_batched_cuda is not None
    # New kernels from WAVE 2.
    assert g.compute_joint_hist_batched_shared_cuda is not None
    assert g.compute_joint_hist_multi_pair_shared_cuda is not None


# --------------------------------------------------------------------------
# A6: KernelTuningCache.update concurrent-process preserves kernels
# --------------------------------------------------------------------------

def _proc_update(cache_dir: str, kernel_name: str, payload: dict) -> None:
    """Worker for the concurrent test. Runs in a fresh process via
    ``multiprocessing.Process``."""
    import os
    os.environ["PYUTILZ_KERNEL_CACHE_DIR"] = cache_dir
    from pyutilz.system.kernel_tuning_cache import (
        KernelTuningCache, hw_fingerprint,
    )
    hw_fingerprint.cache_clear()
    cache = KernelTuningCache()
    cache.update(kernel_name, axes=["n"],
                 regions=[{"n_max": None, **payload}])


def test_concurrent_update_preserves_kernels(tmp_path):
    """Two processes calling ``update`` on different kernel names must
    both land in the final on-disk cache (file-lock + merge-on-write)."""
    import multiprocessing
    procs = [
        multiprocessing.Process(target=_proc_update, args=(str(tmp_path), "k_a", {"v": "a"})),
        multiprocessing.Process(target=_proc_update, args=(str(tmp_path), "k_b", {"v": "b"})),
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, f"worker {p.name} exited {p.exitcode}"

    # Read directly via a fresh KTC instance.
    os.environ["PYUTILZ_KERNEL_CACHE_DIR"] = str(tmp_path)
    from pyutilz.system.kernel_tuning_cache import (
        KernelTuningCache, hw_fingerprint,
    )
    hw_fingerprint.cache_clear()
    cache = KernelTuningCache()
    # Both kernels must be present even though they came from different
    # processes racing on _save -- the file-lock + merge-on-write fix.
    assert cache.has("k_a"), "k_a lost in concurrent save (file-lock failed)"
    assert cache.has("k_b"), "k_b lost in concurrent save (file-lock failed)"
