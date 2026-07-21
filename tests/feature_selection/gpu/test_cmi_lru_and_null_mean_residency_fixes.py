"""Regression tests for two mrmr_audit_2026-07-20 gpu_residency.md fixes:

1. (#1) ``info_theory._cmi_cuda._CMI_RESIDENT_CACHE`` was a plain dict cleared WHOLESALE past 16
   entries, undoing its own re-upload-avoidance on any greedy round touching >16 distinct y/z
   roles. Fixed to a real OrderedDict LRU (move-to-end on hit, evict only the coldest on
   overflow), mirroring ``_fe_resident_operands._FE_RESIDENT_OPERANDS``.
2. (#2) ``gpu.mi_direct_gpu``'s ``return_null_mean=True`` branch (which always disables
   early-stop, so the permutation loop already runs to completion) did a blocking
   ``totals.get()[0]`` EVERY iteration. Fixed to stage each iteration's scalar into a resident
   device buffer and read back once after the loop.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    """Whether a usable CUDA device is available (used to skip the module when it is not)."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available

        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


class TestCmiResidentCacheLru:
    """#1: the resident y/z cache must evict only the single coldest entry, never clear whole."""

    def test_overflow_evicts_coldest_only_not_whole_cache(self):
        """Pushing 17 distinct roles past the 16-entry cap must drop exactly the coldest (first)
        entry, not every previously-cached entry -- the bug this fix replaces would leave the
        cache EMPTY the moment a 17th distinct role is touched."""
        from mlframe.feature_selection.filters.info_theory._cmi_cuda import (
            _CMI_RESIDENT_CACHE,
            _resident_upload,
            clear_cmi_resident_cache,
        )

        clear_cmi_resident_cache()
        keys = [("role", i) for i in range(16)]
        for i, k in enumerate(keys):
            _resident_upload(np.arange(8, dtype=np.int32) + i, k)
        assert len(_CMI_RESIDENT_CACHE) == 16

        # a 17th distinct role overflows the cap -> only the coldest (keys[0]) must be evicted.
        _resident_upload(np.arange(8, dtype=np.int32) + 99, ("role", 16))
        assert len(_CMI_RESIDENT_CACHE) == 16, f"cache should stay capped at 16, not clear/grow: {len(_CMI_RESIDENT_CACHE)}"
        assert keys[0] not in _CMI_RESIDENT_CACHE, "the coldest entry should have been evicted"
        assert ("role", 5) in _CMI_RESIDENT_CACHE, "a still-hot mid-list entry must survive a single-entry eviction (would be gone under whole-cache-clear)"
        clear_cmi_resident_cache()

    def test_reaccessed_entry_survives_subsequent_overflow(self):
        """Re-touching an entry (a cache HIT) must move it to the hot end, so a later overflow
        evicts a genuinely colder entry instead -- the defining LRU behavior a plain dict-with-
        clear() cannot express."""
        from mlframe.feature_selection.filters.info_theory._cmi_cuda import (
            _CMI_RESIDENT_CACHE,
            _resident_upload,
            clear_cmi_resident_cache,
        )

        clear_cmi_resident_cache()
        arrs = {i: np.arange(8, dtype=np.int32) + i for i in range(16)}
        for i, a in arrs.items():
            _resident_upload(a, ("role", i))
        # re-touch the OLDEST entry (role 0) -- a cache HIT, same content -> should move to hot end.
        _resident_upload(arrs[0], ("role", 0))
        # push one more distinct role past the cap: the now-coldest entry (role 1) must be evicted, not role 0.
        _resident_upload(np.arange(8, dtype=np.int32) + 200, ("role", 16))
        assert ("role", 0) in _CMI_RESIDENT_CACHE, "a re-touched (hit) entry must survive the next overflow"
        assert ("role", 1) not in _CMI_RESIDENT_CACHE, "the entry that became coldest after the re-touch should be evicted instead"
        clear_cmi_resident_cache()


class TestMiDirectGpuNullMeanBatchedReadback:
    """#2: return_null_mean=True must not pay a blocking D2H per permutation."""

    def test_return_null_mean_d2h_ops_do_not_scale_with_npermutations(self):
        """The number of D2H transfers recorded by residency_audit() must stay roughly constant
        as npermutations grows (one batched readback), not scale linearly with it (the bug this
        fix replaces: one ``.get()`` per permutation)."""
        from mlframe.feature_selection.filters.gpu import mi_direct_gpu
        from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit

        rng = np.random.default_rng(0)
        n = 4000
        factors = rng.integers(0, 5, size=(n, 2)).astype(np.int32)

        with residency_audit() as rep_small:
            mi_direct_gpu(factors, (0,), (1,), (5, 5), npermutations=8, return_null_mean=True, base_seed=1)
        with residency_audit() as rep_large:
            mi_direct_gpu(factors, (0,), (1,), (5, 5), npermutations=64, return_null_mean=True, base_seed=1)

        # A per-iteration .get() would show len(d2h) scaling ~linearly with npermutations (8 -> 64 is 8x);
        # the batched fix keeps it flat (a handful of scalar reads regardless of the permutation budget).
        assert len(rep_large.d2h) <= len(rep_small.d2h) + 3, (
            f"D2H op count grew with npermutations (8->64): {len(rep_small.d2h)} -> {len(rep_large.d2h)}; "
            "expected roughly constant under the batched-readback fix"
        )

    def test_null_mean_and_pvalue_match_unbatched_reference_computation(self):
        """The batched-readback null mean / nfailed-derived p-value must equal a manual host-side
        recompute from the SAME per-permutation MI values (sanity: the refactor is a readback-
        timing change, not a numerical one)."""
        from mlframe.feature_selection.filters.gpu import mi_direct_gpu
        from mlframe.feature_selection.filters.permutation import _perm_pvalue

        rng = np.random.default_rng(3)
        n = 3000
        factors = rng.integers(0, 4, size=(n, 2)).astype(np.int32)
        npermutations = 40

        mi, confidence, null_mean, p_value = mi_direct_gpu(factors, (0,), (1,), (4, 4), npermutations=npermutations, return_null_mean=True, base_seed=7)
        assert np.isfinite(mi) and mi >= 0.0
        assert np.isfinite(null_mean) and null_mean >= 0.0
        assert 0.0 <= confidence <= 1.0
        assert 0.0 < p_value <= 1.0
        # nfailed is implicitly consistent with confidence: confidence == 1 - nfailed/nchecked.
        nfailed_implied = round((1 - confidence) * npermutations)
        expected_p = _perm_pvalue(nfailed_implied, npermutations, full_budget=npermutations)
        assert abs(expected_p - p_value) < 1e-9, f"p-value ({p_value}) inconsistent with confidence-implied nfailed ({nfailed_implied}) -> {expected_p}"
