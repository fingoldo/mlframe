"""Direct unit coverage for ``_gpu_hw_launch`` (mrmr_audit_2026-07-20 test_coverage.md #15): the
no-HW-info (no cupy / no CUDA device / attribute-query failure) fallback contract, and that KTC
callers (``_gpu_resident_radix_ktc._radix_threads_variants``) correctly degrade to their raw seed
list when ``occupancy_block_candidates`` reports no HW info.
"""

from __future__ import annotations

import mlframe.feature_selection.filters._gpu_hw_launch as hw_launch


class TestDevicePropsNoCudaFallback:
    """device_props() returns {} (not raising) when cupy/CUDA is unavailable or the attribute
    query fails -- callers treat an empty dict as 'no HW info'."""

    def test_empty_dict_when_cupy_import_fails(self, monkeypatch):
        """Force the cupy import inside device_props to fail; the cached result must be {}."""
        monkeypatch.setattr(hw_launch, "_DEV_PROPS", None)  # bust the process-wide cache

        import builtins

        real_import = builtins.__import__

        def _poisoned_import(name, *args, **kwargs):
            """Raise ImportError specifically for cupy; delegate everything else."""
            if name == "cupy":
                raise ImportError("simulated no-cupy host")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _poisoned_import)
        try:
            props = hw_launch.device_props()
            assert props == {}
        finally:
            monkeypatch.setattr(hw_launch, "_DEV_PROPS", None)  # don't leak the poisoned-cache result


class TestOccupancyBlockCandidatesNoHwInfo:
    """occupancy_block_candidates() returns [] when device_props() is empty -- callers keep their
    historical sweep/default seed list."""

    def test_empty_list_when_no_hw_info(self, monkeypatch):
        """With device_props() forced empty, occupancy_block_candidates must return []."""
        monkeypatch.setattr(hw_launch, "device_props", lambda: {})
        out = hw_launch.occupancy_block_candidates(regs_per_thread=32, static_smem=0, dyn_smem=0)
        assert out == []

    def test_nonempty_when_hw_info_present(self, monkeypatch):
        """Sanity check (contrast case): with plausible fake HW info, at least one candidate is returned."""
        fake_props = {
            "sm_count": 20,
            "max_threads_per_block": 1024,
            "max_threads_per_sm": 2048,
            "max_blocks_per_sm": 16,
            "warp": 32,
            "shared_per_block": 48 * 1024,
            "regs_per_block": 65536,
            "regs_per_sm": 65536,
            "shared_per_sm": 48 * 1024,
        }
        monkeypatch.setattr(hw_launch, "device_props", lambda: fake_props)
        out = hw_launch.occupancy_block_candidates(regs_per_thread=32, static_smem=0, dyn_smem=0)
        assert out, "with plausible HW info, at least one warp-multiple block size must be occupancy-valid"
        assert all(b % fake_props["warp"] == 0 for b in out), "every candidate must be a warp multiple"


class TestFillGrid1dNoHwInfoFallback:
    """fill_grid_1d() falls back to plain ceil(n_work/block) coverage when there's no HW info,
    instead of raising or under-covering."""

    def test_falls_back_to_ceil_division_when_no_hw_info(self, monkeypatch):
        """No HW info -> grid = ceil(n_work / block), never less."""
        monkeypatch.setattr(hw_launch, "device_props", lambda: {})
        n_work, block = 1000, 256
        grid = hw_launch.fill_grid_1d(n_work, block)
        assert grid == (n_work + block - 1) // block

    def test_non_positive_block_returns_one(self):
        """A degenerate block<=0 must return 1 (never zero/negative), avoiding a divide-by-zero grid."""
        assert hw_launch.fill_grid_1d(1000, 0) == 1
        assert hw_launch.fill_grid_1d(1000, -5) == 1


class TestRadixThreadsVariantsDegradesToSeedListWithoutHwInfo:
    """_gpu_resident_radix_ktc._radix_threads_variants must fall back to its raw seed list when
    occupancy_block_candidates reports no HW info (the caller-side contract this module exists to serve)."""

    def test_falls_back_to_seed_list(self, monkeypatch):
        """With device_props() forced empty, _radix_threads_variants must return exactly the seed list."""
        import mlframe.feature_selection.filters._gpu_resident_radix_ktc as radix_ktc

        monkeypatch.setattr(hw_launch, "device_props", lambda: {})
        out = radix_ktc._radix_threads_variants()
        assert out == list(radix_ktc._RADIX_THREADS_SEED)
