"""Direct unit coverage for ``mrmr._mrmr_class_config._MRMRConfigMixin`` (mrmr_audit_2026-07-20
test_coverage.md #7). ``_coerce_target_dtype``'s int16 boundary, ``_effective_random_seed``'s
precedence, ``_effective_n_jobs``'s -1 resolution, and ``clear_fit_cache`` were previously only
reachable transitively through a full MRMR.fit() -- pins each directly on a bare (unfitted) MRMR
instance."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.mrmr import MRMR


class TestCoerceTargetDtypeInt16Boundary:
    """_coerce_target_dtype downcasts int64 -> int16 only when the value range fits, else keeps
    int64 and warns (never silently truncates)."""

    def test_in_range_int64_is_downcast_to_int16(self):
        """Values within [-32768, 32767] get downcast."""
        m = MRMR()
        vals = np.array([-32768, 0, 32767], dtype=np.int64)
        out = m._coerce_target_dtype(vals)
        assert out.dtype == np.int16
        np.testing.assert_array_equal(out, vals)

    def test_out_of_range_int64_is_kept_int64_not_silently_truncated(self):
        """A value just outside int16 range must NOT be downcast (the pre-fix bug: unconditional
        astype(int16) would silently wrap 32768 to -32768)."""
        m = MRMR()
        vals = np.array([-32768, 0, 32768], dtype=np.int64)  # 32768 is one past int16 max
        out = m._coerce_target_dtype(vals)
        assert out.dtype == np.int64, "B-regression: out-of-range int64 target was silently downcast to int16"
        np.testing.assert_array_equal(out, vals)

    def test_non_int64_dtype_passes_through_unchanged(self):
        """A non-int64 array (e.g. float64 regression target) is returned as-is -- the function
        only ever acts on int64 input."""
        m = MRMR()
        vals = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        out = m._coerce_target_dtype(vals)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, vals)

    def test_exact_boundary_values_are_downcast(self):
        """The exact int16 min/max boundary values (not just interior values) must be downcast."""
        m = MRMR()
        info = np.iinfo(np.int16)
        vals = np.array([info.min, info.max], dtype=np.int64)
        out = m._coerce_target_dtype(vals)
        assert out.dtype == np.int16


class TestEffectiveRandomSeedPrecedence:
    """random_state (canonical) wins over random_seed (deprecated alias); both None -> entropy-seeded (None)."""

    def test_random_state_wins_when_both_set(self):
        """random_state takes precedence over random_seed when both are explicitly set."""
        m = MRMR(random_state=42, random_seed=7)
        assert m._effective_random_seed() == 42

    def test_random_seed_alias_used_when_random_state_unset(self):
        """The deprecated random_seed alias fills in when random_state is None."""
        m = MRMR(random_seed=7)
        assert m._effective_random_seed() == 7

    def test_neither_set_returns_none(self):
        """With neither seed param set, the effective seed is None (entropy-seeded, not silently 0)."""
        m = MRMR()
        assert m._effective_random_seed() is None


class TestEffectiveNJobsResolution:
    """n_jobs=-1 (the sentinel) resolves to the physical core count at point-of-use; any other
    explicit value passes through unchanged."""

    def test_n_jobs_minus_one_resolves_to_a_positive_core_count(self):
        """The -1 sentinel must resolve to a real positive integer, not stay -1."""
        m = MRMR(n_jobs=-1)
        resolved = m._effective_n_jobs()
        assert isinstance(resolved, int)
        assert resolved >= 1

    def test_explicit_n_jobs_passes_through_unchanged(self):
        """An explicit n_jobs value (not the -1 sentinel) is returned as-is."""
        m = MRMR(n_jobs=3)
        assert m._effective_n_jobs() == 3


class TestClearFitCache:
    """clear_fit_cache() drains the process-wide fit cache and returns the dropped entry count."""

    def test_clear_on_empty_cache_returns_zero(self):
        """Clearing an already-empty cache returns 0, not an error."""
        MRMR.clear_fit_cache()
        assert MRMR.clear_fit_cache() == 0

    def test_clear_returns_the_dropped_entry_count_and_empties_the_cache(self):
        """Manually seed the cache with entries, then verify clear_fit_cache reports the count and
        leaves the cache empty afterward."""
        MRMR.clear_fit_cache()
        MRMR._FIT_CACHE[("fake_key_1",)] = object()
        MRMR._FIT_CACHE[("fake_key_2",)] = object()
        n_dropped = MRMR.clear_fit_cache()
        assert n_dropped == 2
        assert len(MRMR._FIT_CACHE) == 0
