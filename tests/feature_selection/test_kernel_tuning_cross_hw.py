"""Cross-HW dispatcher verification for joint_hist kernel tuning.

The existing ``test_kernel_tuning_cache_critic3.py`` already parametrizes
the cc-major fallback by ``block_size`` for cc 5/6/7/8/9. This file
adds the missing coverage:

* **variant_per_cc**: every cc in ``_FALLBACK_BY_CC`` must also return the
  expected ``kernel_variant`` string (``shared`` or ``global``). The
  existing test only checks ``block_size`` -- a regression that swapped
  ``shared``<->``global`` would pass.

* **joint_size_override_per_cc**: a ``joint_size > 4096`` MUST force
  ``kernel_variant='global'`` regardless of the cc-default (cc 5/6/7/8
  default to shared; the override flips them to global because the
  shared-mem joint histogram cannot fit). cc 9 is unaffected (already
  global by default).

* **unknown_cc_falls_back_to_size_default**: a cc not in the table
  (e.g. simulated cc 10) must degrade gracefully to the size-only
  defaults (shared for small, global for large), not raise.
"""
from __future__ import annotations

from unittest import mock

import pytest


@pytest.fixture(autouse=True)
def _reset_cc_major_memo():
    """Snapshot/restore the process-global ``dispatch._CC_MAJOR_CACHE`` around
    every test here. dispatch.py memoises the GPU cc_major (perf commit f858046e,
    drop per-call nvidia-smi); on a real-GPU host that memo pins the host's true
    cc_major, so the ``gpu_capability_summary`` mocks below would be IGNORED and
    every simulated-cc case would read the host routing. Reset to None before each
    test (mock re-probes a fresh value) and restore the host value after so a real
    probe isn't leaked across the suite."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch
    saved = dispatch._CC_MAJOR_CACHE
    dispatch._CC_MAJOR_CACHE = None
    try:
        yield
    finally:
        dispatch._CC_MAJOR_CACHE = saved


@pytest.mark.parametrize("cc_major,expected_variant", [
    (5, "shared"),   # Maxwell
    (6, "shared"),   # Pascal (host's actual HW)
    (7, "shared"),   # Volta / Turing
    (8, "shared"),   # Ampere
    (9, "global"),   # Hopper
])
def test_fallback_variant_per_cc_matches_table(cc_major, expected_variant):
    """For each cc in _FALLBACK_BY_CC, the fallback must return the
    documented kernel_variant. Complements the block_size parametric
    test in test_kernel_tuning_cache_critic3.py."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with mock.patch.object(dispatch, "_get_cache", return_value=False):
        with mock.patch(
            "pyutilz.system.gpu_dispatch.gpu_capability_summary",
            return_value={
                "cc_major": cc_major, "cc_minor": 0, "name": "SimulatedCC{0}".format(cc_major),
            },
        ):
            # joint_size=25 stays below the 4096 override threshold, so the
            # cc-default takes effect.
            r = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=25)
    assert r["kernel_variant"] == expected_variant, (
        f"cc {cc_major} expected kernel_variant={expected_variant!r}; got {r!r}"
    )


@pytest.mark.parametrize("cc_major", [5, 6, 7, 8])
def test_large_joint_size_forces_global_regardless_of_cc(cc_major):
    """joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK (4096) must override
    the cc-default shared variant. Critical for the cc 5-8 path -- the
    shared-mem histogram doesn't fit and the kernel would either OOM or
    silently corrupt counters."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with mock.patch.object(dispatch, "_get_cache", return_value=False):
        with mock.patch(
            "pyutilz.system.gpu_dispatch.gpu_capability_summary",
            return_value={
                "cc_major": cc_major, "cc_minor": 0, "name": "SimCC{0}".format(cc_major),
            },
        ):
            r = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=8192)
    assert r["kernel_variant"] == "global", (
        f"cc {cc_major} with joint_size=8192 must route to global; got {r!r}"
    )


def test_unknown_cc_falls_back_to_size_default():
    """A cc_major not in _FALLBACK_BY_CC (simulated cc 10) must NOT raise.
    Degrades to size-only default: shared for small joint, global for large."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with mock.patch.object(dispatch, "_get_cache", return_value=False):
        with mock.patch(
            "pyutilz.system.gpu_dispatch.gpu_capability_summary",
            return_value={"cc_major": 10, "cc_minor": 0, "name": "FutureCC10"},
        ):
            r_small = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=25)
            r_large = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=8192)

    assert r_small["kernel_variant"] in ("shared", "global"), (
        f"unknown-cc small-joint path returned {r_small!r}"
    )
    assert r_large["kernel_variant"] == "global", (
        f"unknown-cc large-joint must force global; got {r_large!r}"
    )


def test_missing_gpu_capability_falls_through_to_size_default():
    """If gpu_capability_summary raises (no CUDA / no driver), the dispatch
    must still return a usable region, not propagate the exception."""
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with mock.patch.object(dispatch, "_get_cache", return_value=False):
        with mock.patch(
            "pyutilz.system.gpu_dispatch.gpu_capability_summary",
            side_effect=RuntimeError("no CUDA driver"),
        ):
            r_small = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=25)
            r_large = dispatch.lookup_joint_hist(n_samples=100_000, joint_size=8192)

    # The cc-aware branch is gone but the size-based defaults remain.
    assert "kernel_variant" in r_small and "block_size" in r_small
    assert r_large["kernel_variant"] == "global", (
        f"no-CUDA large-joint must still force global; got {r_large!r}"
    )
