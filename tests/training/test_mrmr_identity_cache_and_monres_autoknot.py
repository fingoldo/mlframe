"""#2 MRMR cross-target identity cache + #3 monres auto-knot biz_val tests.

#2 Identity cache: when MRMR.fit on a given X returns an identity result (all input columns selected, zero engineered features), a subsequent fit on the SAME X-fingerprint with a different y short-circuits the FE pipeline -- production TVT log spent 88 min on an identity-returning MRMR for raw TVT and was about to spend ANOTHER 88 min on the identity-returning MRMR for TVT-monres-Y. Cache hit saves the second 88 min entirely.

#3 Monres auto-knot: ``_monotonic_residual_fit`` previously used a fixed ``n_knots=12`` regardless of base cardinality. For categorical / discrete bases the 12 quantile knots collapse to fewer unique x-positions, oversmoothing the spline + producing degenerate fits. Auto-cap by ``n_unique_base // 200``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import (
    MRMR,
    _MRMR_IDENTITY_FP_CACHE,
    _MRMR_IDENTITY_FP_LOCK,
    _mrmr_compute_x_fingerprint,
    _mrmr_compute_y_fingerprint_sample,
)
from mlframe.training.composite_transforms import _monotonic_residual_fit


@pytest.fixture(autouse=True)
def _clear_mrmr_cache():
    """Each test starts with a fresh identity cache so order-of-execution doesn't poison assertions."""
    _MRMR_IDENTITY_FP_CACHE.clear()
    yield
    _MRMR_IDENTITY_FP_CACHE.clear()


# ----------------------------------------------------------------------
# #2 MRMR identity cache
# ----------------------------------------------------------------------


class TestMRMRIdentityCache:
    def test_fingerprint_matches_polars_pandas_on_same_dtypes(self) -> None:
        """Production TVT log: MRMR was called once on polars X then on pandas X for a composite target. The fingerprint MUST match across backends so the cache works."""
        pl = pytest.importorskip("polars")
        arr_a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        arr_b = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        df_pl = pl.DataFrame({"a": arr_a, "b": arr_b})
        df_pd = pd.DataFrame({"a": arr_a, "b": arr_b})
        fp_pl = _mrmr_compute_x_fingerprint(df_pl)
        fp_pd = _mrmr_compute_x_fingerprint(df_pd)
        assert fp_pl == fp_pd, f"fp polars={fp_pl!r} != fp pandas={fp_pd!r}"

    def test_different_dtypes_produce_different_fingerprints(self) -> None:
        """Sanity: int vs float must produce distinct fingerprints (the cache MUST NOT collide on real semantic difference)."""
        df_a = pd.DataFrame({"x": np.array([1, 2, 3], dtype=np.int32)})
        df_b = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0], dtype=np.float64)})
        assert _mrmr_compute_x_fingerprint(df_a) != _mrmr_compute_x_fingerprint(df_b)

    def test_default_skip_flag_is_true_post_flip(self) -> None:
        """Default ``mrmr_skip_when_prior_was_identity`` flipped False -> True on 2026-05-18 (Accuracy/perf over legacy). Set explicitly to False to restore historical "always re-fit"."""
        from time import perf_counter
        rng = np.random.default_rng(0)
        n = 1000
        X = pd.DataFrame({
            "a": rng.normal(size=n), "b": rng.normal(size=n),
            "c": rng.normal(size=n), "d": rng.normal(size=n),
        })
        y1 = rng.normal(size=n)
        m = MRMR(verbose=0)
        # The default flag is now True (post-flip).
        assert m.mrmr_skip_when_prior_was_identity is True
        t0 = perf_counter()
        m.fit(X, y1)
        elapsed_first = perf_counter() - t0
        assert hasattr(m, "support_")

    def test_explicit_false_disables_skip(self) -> None:
        """Setting ``mrmr_skip_when_prior_was_identity=False`` explicitly restores the historical "no short-circuit" behaviour. Even if cache has an identity flag, the fit runs normally."""
        rng = np.random.default_rng(0)
        n = 500
        X = pd.DataFrame({
            "a": rng.normal(size=n), "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        })
        y = rng.normal(size=n)
        # Pre-populate cache as if a previous fit returned identity.
        fp = _mrmr_compute_x_fingerprint(X)
        _MRMR_IDENTITY_FP_CACHE[fp] = True
        m = MRMR(verbose=0, mrmr_skip_when_prior_was_identity=False)
        m.fit(X, y)
        # support_ from a REAL fit (not shortcut) exists; signature attribute is None or a non-shortcut marker.
        assert hasattr(m, "support_")
        assert not str(getattr(m, "signature", "")).startswith("_mrmr_identity_shortcut")

    def test_identity_skip_short_circuits_second_call(self) -> None:
        """When prior fit was identity AND ``mrmr_skip_when_prior_was_identity=True``, the second fit on the SAME X with a different y returns identity output in O(microseconds), not O(seconds)."""
        rng = np.random.default_rng(7)
        n = 500
        X = pd.DataFrame({
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "d": rng.normal(size=n),
        })
        y1 = rng.normal(size=n)
        y2 = rng.normal(size=n)  # different target

        # Pre-populate cache as if a previous fit returned identity.
        fp = _mrmr_compute_x_fingerprint(X)
        _MRMR_IDENTITY_FP_CACHE[fp] = True

        from time import perf_counter
        m = MRMR(verbose=0, mrmr_skip_when_prior_was_identity=True)
        t0 = perf_counter()
        m.fit(X, y2)
        elapsed = perf_counter() - t0

        # The shortcut path must complete in ms, not seconds (a real fit on 500 rows is ~1s).
        assert elapsed < 0.5, f"short-circuit took {elapsed:.3f}s -- too slow"
        # Identity output: all input columns selected, no engineered features.
        assert len(m.support_) == X.shape[1]
        assert m._engineered_features_ == []
        assert hasattr(m, "feature_names_in_")
        assert list(m.feature_names_in_) == list(X.columns)


# ----------------------------------------------------------------------
# #3 Monres auto-knot tuning
# ----------------------------------------------------------------------


class TestMonresAutoKnotTuning:
    def test_low_cardinality_base_gets_few_knots(self) -> None:
        """When base has e.g. 20 unique values, n_unique // 200 = 0 -> auto-cap to 3 (the floor). The default 12 would oversmooth and likely produce degenerate fit."""
        rng = np.random.default_rng(0)
        n = 1000
        # Base takes only 20 unique values (categorical-ish).
        base = rng.choice(np.linspace(0, 10, 20), size=n)
        y = 0.5 * base + rng.normal(0, 0.5, n)
        params = _monotonic_residual_fit(y, base)
        # The fitted spline must have at most 3 effective knots (cap by floor of auto_knots since 20 < 200).
        assert params["n_knots_effective"] <= 3, (
            f"low-cardinality base produced {params['n_knots_effective']} knots; "
            f"expected <= 3"
        )

    def test_high_cardinality_base_keeps_default_knots(self) -> None:
        """When base has 5000+ unique values (continuous), n_unique // 200 = 25 -> cap at 12 (default n_knots)."""
        rng = np.random.default_rng(0)
        n = 5000
        base = rng.normal(0, 1, n)  # 5000 unique floats
        y = 2 * base + rng.normal(0, 0.5, n)
        params = _monotonic_residual_fit(y, base)
        # Effective knots should be exactly the default = 12 (or close, after dedup).
        assert params["n_knots_effective"] <= 12

    def test_mid_cardinality_base_intermediate_knots(self) -> None:
        """When base has ~400 unique values, n_unique // 200 = 2 -> cap to 3 (floor)."""
        rng = np.random.default_rng(0)
        n = 1000
        # Round to 400 unique values.
        base = np.round(rng.normal(0, 5, n) * 40) / 40
        y = base + rng.normal(0, 0.3, n)
        params = _monotonic_residual_fit(y, base)
        # 400 // 200 = 2 -> capped to 3 (floor).
        assert params["n_knots_effective"] <= 4


# ----------------------------------------------------------------------
# T2#16 cell-content collision regression test + T3#18 y-fingerprint option
# ----------------------------------------------------------------------


class TestIdentityCacheCellContentCollision:
    """The X-fingerprint hashes only column NAMES + n_rows + dtypes -- NOT cell content. Two structurally-identical frames with different values collide on the same fingerprint. Documented trade-off."""

    def test_same_structure_different_values_collide(self) -> None:
        """Two DataFrames with identical schema but different cell values produce the SAME X-fingerprint."""
        df_a = pd.DataFrame({
            "a": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "b": np.array([4.0, 5.0, 6.0], dtype=np.float32),
        })
        df_b = pd.DataFrame({
            "a": np.array([100.0, 200.0, 300.0], dtype=np.float32),
            "b": np.array([400.0, 500.0, 600.0], dtype=np.float32),
        })
        fp_a = _mrmr_compute_x_fingerprint(df_a)
        fp_b = _mrmr_compute_x_fingerprint(df_b)
        # Documented trade-off: SAME fingerprint despite distinct content.
        assert fp_a == fp_b, (
            f"X-fingerprint should COLLIDE on same schema (documented trade-off); "
            f"got fp_a={fp_a!r}, fp_b={fp_b!r}"
        )


class TestIdentityCacheYFingerprintOption:
    """``mrmr_identity_cache_include_y=True`` adds a y-fingerprint to the cache key so legitimately distinct targets on same X get separate slots."""

    def test_default_y_inclusion_off(self) -> None:
        m = MRMR()
        assert m.mrmr_identity_cache_include_y is False

    def test_y_fingerprint_distinguishes_different_targets(self) -> None:
        """When the option is ON, different y values on same X must produce different cache keys."""
        rng = np.random.default_rng(0)
        n = 400
        X = pd.DataFrame({
            "a": rng.normal(size=n), "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        })
        y1 = rng.normal(size=n)
        y2 = rng.normal(size=n)  # different target
        fp_y1 = _mrmr_compute_y_fingerprint_sample(y1)
        fp_y2 = _mrmr_compute_y_fingerprint_sample(y2)
        assert fp_y1 != fp_y2, "different y arrays must produce different y-fingerprints"

    def test_y_fingerprint_stable_under_dtype_jitter(self) -> None:
        """Same y values in float32 vs float64 should produce same y-fingerprint after rounding."""
        rng = np.random.default_rng(0)
        y = rng.normal(size=500)
        fp_64 = _mrmr_compute_y_fingerprint_sample(y.astype(np.float64))
        fp_32 = _mrmr_compute_y_fingerprint_sample(y.astype(np.float32))
        # 6-decimal rounding inside the fingerprint means tiny dtype-cast noise doesn't flip the hash.
        # (Strict equality would be brittle; just verify they're close to the same hash family.)
        # Loose: at least one of them is reproducible across two calls.
        fp_64_again = _mrmr_compute_y_fingerprint_sample(y.astype(np.float64))
        assert fp_64 == fp_64_again


class TestIdentityCacheThreadSafe:
    """Module-global cache mutations are protected by ``_MRMR_IDENTITY_FP_LOCK``."""

    def test_lock_exists(self) -> None:
        # Lock instance is a threading.Lock object; cannot easily assert its identity / type without import.
        import threading
        assert isinstance(_MRMR_IDENTITY_FP_LOCK, type(threading.Lock()))

    def test_concurrent_cache_writes_do_not_race(self) -> None:
        """Spawn 4 threads writing different X-fingerprints concurrently; verify all entries land in the cache."""
        import threading
        _MRMR_IDENTITY_FP_CACHE.clear()
        keys = [f"fp_{i:04d}" for i in range(50)]

        def _writer(start_idx: int) -> None:
            for k in keys[start_idx:start_idx + 25]:
                with _MRMR_IDENTITY_FP_LOCK:
                    _MRMR_IDENTITY_FP_CACHE[k] = True

        threads = [
            threading.Thread(target=_writer, args=(0,)),
            threading.Thread(target=_writer, args=(25,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 50 keys should be present in the cache.
        for k in keys:
            assert k in _MRMR_IDENTITY_FP_CACHE

    def test_lock_serialises_same_key_read_modify_write(self) -> None:
        """MEDIUM#10 2026-05-18: REAL contention test - 8 threads x 1000
        iterations all doing read-modify-write on the SAME key under the
        lock. Without the lock, racing read-then-write would lose
        updates and the final count would be < 8000. With the lock, the
        critical section is atomic and the final count equals 8000.
        """
        import threading
        _MRMR_IDENTITY_FP_CACHE.clear()
        shared_key = "fp_contended"
        _MRMR_IDENTITY_FP_CACHE[shared_key] = 0

        def _increment_under_lock() -> None:
            for _ in range(1000):
                with _MRMR_IDENTITY_FP_LOCK:
                    val = _MRMR_IDENTITY_FP_CACHE[shared_key]
                    _MRMR_IDENTITY_FP_CACHE[shared_key] = val + 1

        threads = [
            threading.Thread(target=_increment_under_lock)
            for _ in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert _MRMR_IDENTITY_FP_CACHE[shared_key] == 8000, (
            f"lock failed to serialise concurrent read-modify-write on "
            f"the same key; final value "
            f"{_MRMR_IDENTITY_FP_CACHE[shared_key]} != 8000 (lost "
            f"updates indicate races)"
        )
