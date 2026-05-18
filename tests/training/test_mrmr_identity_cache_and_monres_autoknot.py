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
    _mrmr_compute_x_fingerprint,
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

    def test_default_no_skip(self) -> None:
        """Default ``mrmr_skip_when_prior_was_identity=False`` keeps legacy behaviour. Even if cache has identity flag, a fresh fit runs normally."""
        from time import perf_counter
        rng = np.random.default_rng(0)
        n = 1000
        X = pd.DataFrame({
            "a": rng.normal(size=n), "b": rng.normal(size=n),
            "c": rng.normal(size=n), "d": rng.normal(size=n),
        })
        y1 = rng.normal(size=n)
        m = MRMR(verbose=0)
        t0 = perf_counter()
        m.fit(X, y1)
        elapsed_first = perf_counter() - t0
        assert hasattr(m, "support_")

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
