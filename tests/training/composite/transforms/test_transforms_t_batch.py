"""Unit + regression tests for the T1-T10 transform-registry extension batch.

Covers the behaviour the shared registry-contract suite cannot pin:

- T1: the default ``rolling_quantile_ratio`` is TRAILING (past-only); the centred variant leaks a future regime jump into pre-jump rows, trailing does not.
- T2: grouped recurrent transforms reset state at group boundaries (first rows of each group uncontaminated by the other group's level).
- T3: ``box_cox_y`` MLE lambda + strictly-positive domain.
- T4: ``seasonal_residual`` period selection + phase-mean round trip.
- T5: ``volatility_normalized_residual`` regime-invariant scale.
- T6/T8: multi-base arcsinh / trimmed-LS joint OLS correctness.
- T7: Nadaraya-Watson non-monotone g(base) recovery.
- T9: grouped quantile / monotonic small-group fallback to the global fit.
- T10: gaussian_copula_residual normal-scores algebra + bounded inverse.

Round-trip / serialisation / domain / purity for every entry is exercised by the parametrised
``test_composite_transforms_registry_contract.py`` suite (the new names are registered there via TRANSFORMS_REGISTRY).
"""
from __future__ import annotations

import pickle

import numpy as np
import pytest

from mlframe.training.composite.transforms import (
    TRANSFORMS_REGISTRY,
    compose_target_name,
    get_transform,
    is_composite_target_name,
)
from mlframe.training.composite.transforms.simple import _rolling_median_trailing


# ---------------------------------------------------------------------------
# T1: trailing rolling_quantile_ratio
# ---------------------------------------------------------------------------


class TestRollingQuantileRatioTrailing:
    def test_default_fit_mode_is_trailing(self) -> None:
        t = get_transform("rolling_quantile_ratio")
        base = np.linspace(1.0, 10.0, 50)
        y = base * 1.1
        params = t.fit(y, base)
        assert params["mode"] == "trailing"

    def test_centered_variant_registered(self) -> None:
        t = get_transform("rolling_quantile_ratio_centered")
        base = np.linspace(1.0, 10.0, 50)
        y = base * 1.1
        params = t.fit(y, base)
        assert params["mode"] == "centered"
        assert t.recurrent

    def test_params_without_mode_keep_centered_behaviour(self) -> None:
        """Params fitted before the mode field existed must reproduce the historical centred window on load."""
        t = get_transform("rolling_quantile_ratio")
        rng = np.random.default_rng(0)
        base = np.abs(rng.normal(10.0, 2.0, 200)) + 1.0
        y = base * 1.2
        legacy_params = {"k": 7, "eps": 1e-6}  # no "mode" key
        centered_params = {"k": 7, "eps": 1e-6, "mode": "centered"}
        np.testing.assert_array_equal(
            t.forward(y, base, legacy_params), t.forward(y, base, centered_params),
        )

    def test_trailing_median_matches_pandas_reference(self) -> None:
        import pandas as pd
        rng = np.random.default_rng(1)
        arr = rng.normal(size=101)
        for k in (1, 2, 3, 7, 8, 150):
            ref = pd.Series(arr).rolling(window=k, min_periods=1).median().to_numpy()
            np.testing.assert_allclose(_rolling_median_trailing(arr, k), ref, atol=1e-12)

    def test_trailing_median_skips_nan_like_pandas(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, 5.0])
        out = _rolling_median_trailing(arr, 3)
        # Window at i=2 is [1, nan, 3] -> pandas median skips NaN -> 2.0.
        assert out[2] == 2.0

    def test_round_trip_trailing(self) -> None:
        t = get_transform("rolling_quantile_ratio")
        rng = np.random.default_rng(2)
        base = np.abs(rng.normal(10.0, 2.0, 300)) + 1.0
        y = base * rng.uniform(0.8, 1.2, 300)
        params = t.fit(y, base)
        np.testing.assert_allclose(t.inverse(t.forward(y, base, params), base, params), y, rtol=1e-9)

    def test_regression_centered_leaks_future_regime_jump_trailing_does_not(self) -> None:
        """A base regime jump at row 100 must NOT move trailing-T of rows BEFORE the jump; the centred window reads future rows so its pre-jump tail T shifts. This test fails if the trailing path were secretly centred."""
        n = 200
        jump_at = 100
        base = np.ones(n)
        base[jump_at:] = 100.0
        y = np.ones(n)  # constant y isolates the denominator's window behaviour
        k = 7
        t_trail = get_transform("rolling_quantile_ratio")
        t_cent = get_transform("rolling_quantile_ratio_centered")
        p_trail = t_trail.fit(y, base, k=k)
        p_cent = t_cent.fit(y, base, k=k)
        T_trail = t_trail.forward(y, base, p_trail)
        T_cent = t_cent.forward(y, base, p_cent)
        # Rows just before the jump: centred window includes post-jump base rows once the
        # window majority flips, so its rolling median jumps BEFORE row 100 -> T drops early.
        pre = slice(jump_at - k // 2, jump_at)
        assert np.allclose(T_trail[pre], 1.0), (
            f"trailing T reflected the FUTURE jump in pre-jump rows: {T_trail[pre]}"
        )
        assert not np.allclose(T_cent[pre], 1.0), (
            "centred variant no longer reflects the future jump pre-jump; window semantics changed"
        )

    def test_naming_and_registry(self) -> None:
        assert compose_target_name("y", "rolling_quantile_ratio_centered", "b") == "y-rqrC-b"
        assert is_composite_target_name("y-rqrC-b")


# ---------------------------------------------------------------------------
# T2: grouped recurrent transforms reset state at group boundaries
# ---------------------------------------------------------------------------


def _panel(seed: int, n_per_group: int = 300, levels: tuple[float, ...] = (0.0, 1000.0, -500.0)):
    """Panel with very different per-group levels: y = group_level + random walk + resid; base = y lagged by 1 within the group."""
    rng = np.random.default_rng(seed)
    ys, bases, groups, resids = [], [], [], []
    for gi, level in enumerate(levels):
        walk = np.cumsum(rng.normal(scale=1.0, size=n_per_group))
        resid = rng.normal(scale=0.5, size=n_per_group)
        y_g = level + walk + resid
        base_g = np.concatenate([[y_g[0]], y_g[:-1]])
        ys.append(y_g)
        bases.append(base_g)
        resids.append(resid)
        groups.append(np.full(n_per_group, gi, dtype=np.int64))
    return (np.concatenate(ys), np.concatenate(bases), np.concatenate(groups), np.concatenate(resids), n_per_group)


class TestGroupedRecurrent:
    def test_ewma_grouped_flags(self) -> None:
        for name in ("ewma_residual_grouped", "rolling_quantile_ratio_grouped", "frac_diff_grouped"):
            t = TRANSFORMS_REGISTRY[name]
            assert t.requires_groups and t.recurrent, name
        assert not TRANSFORMS_REGISTRY["frac_diff_grouped"].requires_base

    def test_regression_ewma_grouped_uncontaminated_at_group_start(self) -> None:
        """First rows of each group: grouped T is close to the true residual (group-level cancelled); the ungrouped EWMA carries the PREVIOUS group's level in, contaminating them."""
        y, base, groups, resid, n_per = _panel(0)
        k = 7
        tg = get_transform("ewma_residual_grouped")
        tu = get_transform("ewma_residual")
        pg = tg.fit(y, base, k=k, groups=groups)
        pu = tu.fit(y, base, k=k)
        Tg = tg.forward(y, base, pg, groups=groups)
        Tu = tu.forward(y, base, pu)
        # Rows 1..10 of the 2nd and 3rd group (row 0 is the seeded anchor row).
        head_idx = np.concatenate([np.arange(n_per + 1, n_per + 11), np.arange(2 * n_per + 1, 2 * n_per + 11)])
        err_g = float(np.mean(np.abs(Tg[head_idx] - resid[head_idx])))
        err_u = float(np.mean(np.abs(Tu[head_idx] - resid[head_idx])))
        # Grouped error is bounded by EWMA smoothing bias (a few units); ungrouped carries a ~1000-unit level in.
        assert err_g < err_u * 0.1, f"grouped head err {err_g:.2f} should crush ungrouped {err_u:.2f}"

    def test_regression_frac_diff_grouped_uncontaminated_at_group_start(self) -> None:
        """Per-group anchor padding: the first rows of each group must not see the OTHER group's level through the weight tail."""
        y, base, groups, _resid, n_per = _panel(1)
        tg = get_transform("frac_diff_grouped")
        tu = get_transform("frac_diff")
        pg = tg.fit(y, None, d=0.5, lags=20, groups=groups)
        pu = tu.fit(y, np.zeros_like(y), d=0.5, lags=20)
        Tg = tg.forward(y, None, pg, groups=groups)
        Tu = tu.forward(y, np.zeros_like(y), pu)
        # Group 2 (level +1000) heads: ungrouped pads with the GLOBAL mean (~166), grouped with the group's own mean.
        head_idx = np.arange(n_per, n_per + 5)
        grp_mean = float(np.mean(y[n_per : 2 * n_per]))
        # d=0.5 frac-diff of a level-L constant-ish head ~ L * sum(w_0..w_i); compare deviation scales.
        dev_g = float(np.mean(np.abs(Tg[head_idx] - (1 - 0.5) * 0)))  # grouped: anchored at own mean, T near y - mean-ish, small
        assert np.mean(np.abs(Tg[head_idx])) < np.mean(np.abs(Tu[head_idx])), (
            f"grouped frac-diff head |T| {np.mean(np.abs(Tg[head_idx])):.1f} should be below "
            f"ungrouped {np.mean(np.abs(Tu[head_idx])):.1f} (own-mean padding), grp_mean={grp_mean:.1f}, dev_g={dev_g:.2f}"
        )

    def test_frac_diff_grouped_round_trip_with_interleaved_groups(self) -> None:
        """Rows of one group need not be contiguous: interleaved panel still round-trips exactly per group."""
        rng = np.random.default_rng(3)
        n = 400
        groups = rng.integers(0, 3, size=n)
        y = rng.normal(size=n).cumsum() + groups * 50.0
        t = get_transform("frac_diff_grouped")
        p = t.fit(y, None, groups=groups)
        T = t.forward(y, None, p, groups=groups)
        y_back = t.inverse(T, None, p, groups=groups)
        np.testing.assert_allclose(y_back, y, rtol=1e-7, atol=1e-7)

    def test_rqr_grouped_window_confined_to_group(self) -> None:
        """A huge base level in group 0 must not enter group 1's rolling median even when rows interleave."""
        groups = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        base = np.where(groups == 0, 1000.0, 2.0)
        y = np.where(groups == 0, 2000.0, 4.0)
        t = get_transform("rolling_quantile_ratio_grouped")
        p = t.fit(y, base, k=3, groups=groups)
        T = t.forward(y, base, p, groups=groups)
        np.testing.assert_allclose(T, 2.0, rtol=1e-12)

    def test_ewma_grouped_unseen_group_falls_back_to_global_anchor(self) -> None:
        y, base, groups, _r, _n = _panel(4)
        t = get_transform("ewma_residual_grouped")
        p = t.fit(y, base, k=5, groups=groups)
        new_groups = np.full(10, 99, dtype=np.int64)
        T = t.forward(y[:10], base[:10], p, groups=new_groups)
        assert np.all(np.isfinite(T))

    def test_grouped_params_pickle_round_trip(self) -> None:
        y, base, groups, _r, _n = _panel(5, n_per_group=100)
        for name in ("ewma_residual_grouped", "rolling_quantile_ratio_grouped", "frac_diff_grouped"):
            t = TRANSFORMS_REGISTRY[name]
            b = base if t.requires_base else None
            p = t.fit(y, b, groups=groups)
            p2 = pickle.loads(pickle.dumps(p))
            np.testing.assert_array_equal(
                t.forward(y, b, p, groups=groups), t.forward(y, b, p2, groups=groups),
            )


# ---------------------------------------------------------------------------
# T3: box_cox_y
# ---------------------------------------------------------------------------


class TestBoxCoxY:
    def test_lambda_recovers_log_scale(self) -> None:
        """Lognormal y: the Box-Cox MLE lambda must land near 0 (log transform)."""
        rng = np.random.default_rng(0)
        y = np.exp(rng.normal(size=4000))
        t = get_transform("box_cox_y")
        p = t.fit(y, None)
        assert abs(p["lambda"]) < 0.12, p

    def test_domain_rejects_non_positive(self) -> None:
        t = get_transform("box_cox_y")
        mask = t.domain_check(np.array([1.0, 0.0, -3.0, np.nan, 2.0]), None)
        assert mask.tolist() == [True, False, False, False, True]

    def test_round_trip_and_inverse_matches_scipy(self) -> None:
        from scipy.special import inv_boxcox
        rng = np.random.default_rng(1)
        y = np.exp(rng.normal(size=500)) + 0.1
        t = get_transform("box_cox_y")
        p = t.fit(y, None)
        T = t.forward(y, None, p)
        y_back = t.inverse(T, None, p)
        np.testing.assert_allclose(y_back, y, rtol=1e-8)
        np.testing.assert_allclose(y_back, inv_boxcox(T, p["lambda"]), rtol=1e-10)

    def test_degenerate_constant_y_identity_lambda(self) -> None:
        t = get_transform("box_cox_y")
        assert t.fit(np.full(50, 3.0), None)["lambda"] == 1.0


# ---------------------------------------------------------------------------
# T4: seasonal_residual
# ---------------------------------------------------------------------------


class TestSeasonalResidual:
    def test_period_selected_from_grid(self) -> None:
        rng = np.random.default_rng(0)
        n = 600
        pattern = np.array([0.0, 5.0, -3.0, 8.0, 1.0, -6.0, 2.0])  # period 7
        y = pattern[np.arange(n) % 7] + rng.normal(scale=0.2, size=n)
        t = get_transform("seasonal_residual")
        p = t.fit(y, None)
        assert p["period"] == 7
        T = t.forward(y, None, p)
        assert float(np.var(T)) < 0.1 * float(np.var(y))

    def test_explicit_period_kwarg(self) -> None:
        y = np.arange(100, dtype=np.float64)
        t = get_transform("seasonal_residual")
        p = t.fit(y, None, period=12)
        assert p["period"] == 12 and len(p["phase_means"]) == 12

    def test_period_grid_capped_at_n_over_3(self) -> None:
        rng = np.random.default_rng(1)
        y = rng.normal(size=30)  # n//3 = 10 -> candidates {4, 5, 7}
        t = get_transform("seasonal_residual")
        assert t.fit(y, None)["period"] in (4, 5, 7)

    def test_round_trip_exact(self) -> None:
        rng = np.random.default_rng(2)
        y = rng.normal(size=200)
        t = get_transform("seasonal_residual")
        p = t.fit(y, None)
        np.testing.assert_allclose(t.inverse(t.forward(y, None, p), None, p), y, rtol=1e-12)


# ---------------------------------------------------------------------------
# T5: volatility_normalized_residual
# ---------------------------------------------------------------------------


class TestVolatilityNormalizedResidual:
    def test_regime_invariant_scale(self) -> None:
        """Calm-then-turbulent base: the vol-normalised T has a similar scale in both regimes while the plain EWMA residual scale explodes with the regime."""
        rng = np.random.default_rng(0)
        n = 4000
        sigma = np.where(np.arange(n) < n // 2, 0.5, 10.0)
        base = np.cumsum(rng.normal(scale=sigma))
        y = base + rng.normal(scale=sigma)
        tv = get_transform("volatility_normalized_residual")
        te = get_transform("ewma_residual")
        pv = tv.fit(y, base, k=10)
        pe = te.fit(y, base, k=10)
        Tv = tv.forward(y, base, pv)
        Te = te.forward(y, base, pe)
        calm, wild = slice(200, n // 2), slice(n // 2 + 200, n)
        ratio_v = float(np.std(Tv[wild]) / np.std(Tv[calm]))
        ratio_e = float(np.std(Te[wild]) / np.std(Te[calm]))
        assert ratio_v < 0.25 * ratio_e, f"vol-normalised regime ratio {ratio_v:.2f} vs ewma {ratio_e:.2f}"

    def test_round_trip_exact_including_floor_rows(self) -> None:
        t = get_transform("volatility_normalized_residual")
        base = np.full(100, 5.0)  # zero volatility -> floor active on every row
        y = base + np.linspace(-1, 1, 100)
        p = t.fit(y, base, k=5)
        np.testing.assert_allclose(t.inverse(t.forward(y, base, p), base, p), y, rtol=1e-10)

    def test_recurrent_flag(self) -> None:
        assert TRANSFORMS_REGISTRY["volatility_normalized_residual"].recurrent


# ---------------------------------------------------------------------------
# T6 / T8: multi-base extras
# ---------------------------------------------------------------------------


class TestMultiBaseExtras:
    def test_asinh_multi_recovers_arcsinh_plane(self) -> None:
        rng = np.random.default_rng(0)
        n = 3000
        b = rng.normal(size=(n, 2)) * 3.0
        z = 0.7 * np.arcsinh(b[:, 0]) - 0.4 * np.arcsinh(b[:, 1]) + 0.3 + rng.normal(scale=0.01, size=n)
        y = np.sinh(z)
        t = get_transform("asinh_residual_multi")
        p = t.fit(y, b)
        assert not p["collinear_fallback"]
        np.testing.assert_allclose(p["alphas"], [0.7, -0.4], atol=0.02)
        assert abs(p["beta"] - 0.3) < 0.02

    def test_asinh_multi_collinear_guard(self) -> None:
        rng = np.random.default_rng(1)
        b0 = rng.normal(size=500)
        b = np.column_stack([b0, b0 * (1 + 1e-12)])
        y = rng.normal(size=500)
        p = get_transform("asinh_residual_multi").fit(y, b)
        assert p["collinear_fallback"] and p["alphas"] == [0.0, 0.0]

    def test_multi_robust_ignores_outliers(self) -> None:
        rng = np.random.default_rng(2)
        n = 5000
        b = rng.normal(size=(n, 2))
        y = 2.0 * b[:, 0] - 1.0 * b[:, 1] + 0.5 + rng.normal(scale=0.1, size=n)
        idx = rng.choice(n, size=int(0.05 * n), replace=False)
        y_out = y.copy()
        y_out[idx] += rng.standard_normal(idx.size) * 200.0 + 100.0
        robust = get_transform("linear_residual_multi_robust").fit(y_out, b)
        plain = get_transform("linear_residual_multi").fit(y_out, b)
        err_r = abs(robust["beta"] - 0.5) + abs(robust["alphas"][0] - 2.0) + abs(robust["alphas"][1] + 1.0)
        err_p = abs(plain["beta"] - 0.5) + abs(plain["alphas"][0] - 2.0) + abs(plain["alphas"][1] + 1.0)
        assert err_r < 0.05 and err_r < err_p * 0.2, (err_r, err_p)
        assert robust["is_redundant_with_linres_multi"] is False

    def test_multi_robust_redundant_flag_when_nothing_trimmed(self) -> None:
        rng = np.random.default_rng(3)
        b = rng.normal(size=(50, 2))
        y = b @ np.array([1.0, 2.0])  # exact plane, zero residual -> sigma_MAD == 0 -> first pass
        p = get_transform("linear_residual_multi_robust").fit(y, b)
        assert p["is_redundant_with_linres_multi"] is True


# ---------------------------------------------------------------------------
# T7: nadaraya_watson_residual
# ---------------------------------------------------------------------------


class TestNadarayaWatson:
    def test_recovers_non_monotone_g(self) -> None:
        """y = sin(base) + noise: NW residual variance far below raw y variance (monotone PCHIP cannot capture the sine)."""
        rng = np.random.default_rng(0)
        n = 4000
        base = rng.uniform(-6, 6, n)
        y = np.sin(base) + rng.normal(scale=0.1, size=n)
        t = get_transform("nadaraya_watson_residual")
        p = t.fit(y, base)
        T = t.forward(y, base, p)
        assert float(np.var(T)) < 0.15 * float(np.var(y))

    def test_knot_cap(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=10_000)
        y = base + rng.normal(size=10_000)
        p = get_transform("nadaraya_watson_residual").fit(y, base)
        assert len(p["knots_x"]) == 2000

    def test_far_from_support_converges_to_edge_knot(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.uniform(0, 1, 500)
        y = 2.0 * base
        t = get_transform("nadaraya_watson_residual")
        p = t.fit(y, base)
        from mlframe.training.composite.transforms._nadaraya_watson import _nw_g
        g_far = _nw_g(np.array([1e6]), p)
        assert np.isfinite(g_far[0]) and abs(g_far[0] - 2.0) < 0.2


# ---------------------------------------------------------------------------
# T9: grouped quantile / monotonic
# ---------------------------------------------------------------------------


class TestGroupedNonParametric:
    def _data(self, seed: int, small_group_n: int = 5):
        rng = np.random.default_rng(seed)
        n_big = 400
        base = np.concatenate([rng.uniform(0, 10, n_big), rng.uniform(0, 10, n_big), rng.uniform(0, 10, small_group_n)])
        levels = np.concatenate([np.zeros(n_big), np.full(n_big, 100.0), np.full(small_group_n, 50.0)])
        y = levels + 2.0 * base + rng.normal(scale=1.0, size=base.size)
        groups = np.concatenate([np.zeros(n_big), np.ones(n_big), np.full(small_group_n, 2.0)]).astype(np.int64)
        return y, base, groups

    def test_small_group_falls_back_to_global(self) -> None:
        for name in ("quantile_residual_grouped", "monotonic_residual_grouped"):
            y, base, groups = self._data(0)
            t = get_transform(name)
            p = t.fit(y, base, groups=groups)
            assert "2" not in p["per_group"], name
            assert set(p["per_group"]) == {"0", "1"}, name
            assert p["group_sizes"]["2"] == 5

    def test_grouped_beats_global_on_shifted_levels(self) -> None:
        """Neutral-T reconstruction: inverting T=0 recovers each row's bin median. The grouped fit's per-group medians land near the row's own group level (error ~ slope-within-bin + noise); the ungrouped fit's medians sit between the two group levels (~50 units off every row)."""
        y, base, groups = self._data(1)
        tg = get_transform("quantile_residual_grouped")
        tu = get_transform("quantile_residual")
        pg = tg.fit(y, base, groups=groups)
        pu = tu.fit(y, base)
        zeros = np.zeros_like(y)
        err_g = float(np.sqrt(np.mean((tg.inverse(zeros, base, pg, groups=groups) - y) ** 2)))
        err_u = float(np.sqrt(np.mean((tu.inverse(zeros, base, pu) - y) ** 2)))
        assert err_g < 0.25 * err_u, (err_g, err_u)

    def test_unseen_group_routes_to_global(self) -> None:
        y, base, groups = self._data(2)
        t = get_transform("monotonic_residual_grouped")
        p = t.fit(y, base, groups=groups)
        T = t.forward(y[:20], base[:20], p, groups=np.full(20, 77, dtype=np.int64))
        assert np.all(np.isfinite(T))

    def test_shrinkage_factor_in_unit_interval(self) -> None:
        y, base, groups = self._data(3)
        p = get_transform("quantile_residual_grouped").fit(y, base, groups=groups)
        assert 0.0 <= p["shrinkage_factor"] <= 1.0


# ---------------------------------------------------------------------------
# T10: gaussian_copula_residual
# ---------------------------------------------------------------------------


class TestGaussianCopula:
    def test_alpha_matches_copula_correlation(self) -> None:
        """Monotone-warped joint-Gaussian pair: fitted alpha must recover the latent normal-scores slope regardless of the marginal warps."""
        rng = np.random.default_rng(0)
        n = 6000
        z_b = rng.normal(size=n)
        z_y = 0.8 * z_b + np.sqrt(1 - 0.8**2) * rng.normal(size=n)
        y = np.exp(z_y)          # lognormal warp of the y marginal
        base = np.sinh(z_b)      # sinh warp of the base marginal
        p = get_transform("gaussian_copula_residual").fit(y, base)
        assert abs(p["alpha"] - 0.8) < 0.05, p["alpha"]
        assert abs(p["beta"]) < 0.05

    def test_inverse_bounded_by_train_support(self) -> None:
        rng = np.random.default_rng(1)
        y = np.exp(rng.normal(size=1000))
        base = rng.normal(size=1000)
        t = get_transform("gaussian_copula_residual")
        p = t.fit(y, base)
        wild_T = np.array([-50.0, 50.0, 0.0])
        y_hat = t.inverse(wild_T, base[:3], p)
        assert np.all(y_hat >= y.min()) and np.all(y_hat <= y.max())

    def test_round_trip_median_error_small(self) -> None:
        rng = np.random.default_rng(2)
        base = rng.normal(size=2000)
        y = np.exp(0.5 * base + rng.normal(scale=0.3, size=2000))
        t = get_transform("gaussian_copula_residual")
        p = t.fit(y, base)
        y_back = t.inverse(t.forward(y, base, p), base, p)
        assert float(np.median(np.abs(y_back - y))) < 1e-6


# ---------------------------------------------------------------------------
# Naming coverage for the whole batch.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,short", [
    ("rolling_quantile_ratio_centered", "rqrC"),
    ("ewma_residual_grouped", "ewmaG"),
    ("rolling_quantile_ratio_grouped", "rqrG"),
    ("frac_diff_grouped", "fdiffG"),
    ("quantile_residual_grouped", "qresG"),
    ("monotonic_residual_grouped", "monresG"),
    ("box_cox_y", "bcY"),
    ("seasonal_residual", "seas"),
    ("volatility_normalized_residual", "volnr"),
    ("asinh_residual_multi", "asinhrM"),
    ("linear_residual_multi_robust", "linresMR"),
    ("nadaraya_watson_residual", "nwres"),
    ("gaussian_copula_residual", "gcopula"),
])
def test_t_batch_short_names_registered(name: str, short: str) -> None:
    from mlframe.training.composite.transforms import TRANSFORM_NAME_SHORT
    assert name in TRANSFORMS_REGISTRY
    assert TRANSFORM_NAME_SHORT[name] == short
