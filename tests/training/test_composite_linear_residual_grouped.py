"""Tests for ``linear_residual_grouped`` transform (R10c extension #3).

Coverage:
- Round-trip ``y -> T -> y'`` with groups (rtol=1e-7).
- Recovery: per-group alphas approximate the true per-group DGP
  coefficients when n_per_group is large.
- James-Stein shrinkage: returns c=0 with <4 groups; returns c>0
  when alphas are tightly clustered.
- Small-group fallback: groups with n < min_group_size use global
  alpha (no per-group OLS).
- Unseen group at predict: predict on a row with a group label not
  seen at fit must fall back to global alpha + beta.
- Biz_value: per-well TVT-style DGP where the autoregressive
  coefficient differs across wells -- grouped fit produces a residual
  with strictly lower variance AND lower per-group residual MSE than
  the single-alpha-across-wells linear_residual fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _GROUPED_MIN_GROUP_SIZE,
    _linear_residual_grouped_fit,
    _linear_residual_grouped_forward,
    _linear_residual_grouped_inverse,
    _linear_residual_grouped_domain,
    _james_stein_shrinkage_factor,
    get_transform,
)


# ---------------------------------------------------------------------------
# Unit: fit / forward / inverse
# ---------------------------------------------------------------------------


class TestFit:
    def test_requires_groups_argument(self) -> None:
        rng = np.random.default_rng(0)
        y = rng.normal(size=100)
        base = rng.normal(size=100)
        with pytest.raises(ValueError, match="requires a 1-D ``groups``"):
            _linear_residual_grouped_fit(y, base, groups=None)

    def test_recovers_per_group_alpha_on_large_n(self) -> None:
        """When each group has plenty of rows, per-group OLS should
        recover the true per-group coefficients (within ~0.02)."""
        rng = np.random.default_rng(42)
        n_per_group = 500
        true_alphas = [0.85, 0.55, 1.10]
        groups = []
        ys = []
        bases = []
        for i, a in enumerate(true_alphas):
            b = rng.normal(loc=10.0, scale=2.0, size=n_per_group)
            y_g = a * b + 1.5 + rng.normal(scale=0.05, size=n_per_group)
            groups.extend([f"g{i}"] * n_per_group)
            ys.append(y_g)
            bases.append(b)
        y = np.concatenate(ys)
        base = np.concatenate(bases)
        groups = np.asarray(groups)
        params = _linear_residual_grouped_fit(y, base, groups=groups)
        for i, a in enumerate(true_alphas):
            fitted = params["per_group_alphas"][f"g{i}"]
            assert abs(fitted - a) < 0.05, (
                f"group g{i}: expected alpha ~{a:.2f}, got {fitted:.3f}"
            )

    def test_small_group_falls_back_to_global(self) -> None:
        """Group with n < min_group_size MUST use global alpha rather
        than its own (under-determined) OLS. Lock with a 5-row group
        that would otherwise produce wild estimates."""
        rng = np.random.default_rng(7)
        n_big = 300
        b_big = rng.normal(loc=10.0, scale=2.0, size=n_big)
        y_big = 0.9 * b_big + rng.normal(scale=0.1, size=n_big)
        n_tiny = 5  # < _GROUPED_MIN_GROUP_SIZE
        b_tiny = rng.normal(loc=10.0, scale=2.0, size=n_tiny)
        y_tiny = -2.0 * b_tiny + 100.0  # adversarial coefficient
        y = np.concatenate([y_big, y_tiny])
        base = np.concatenate([b_big, b_tiny])
        groups = np.asarray(["big"] * n_big + ["tiny"] * n_tiny)
        params = _linear_residual_grouped_fit(y, base, groups=groups)
        # 'tiny' group must inherit global alpha (close to 0.9 since big
        # dominates), not its own -2.0 estimate.
        assert abs(
            params["per_group_alphas"]["tiny"]
            - params["alpha_global"]
        ) < 1e-9
        assert params["group_sizes"]["tiny"] == n_tiny

    def test_round_trip_y_to_T_to_y(self) -> None:
        rng = np.random.default_rng(11)
        n_per_group = 200
        groups_list = []
        ys = []
        bases = []
        for i, a in enumerate([0.95, 0.85, 1.05, 0.75]):
            b = rng.normal(loc=10.0, scale=2.0, size=n_per_group)
            y_g = a * b + rng.normal(scale=0.2, size=n_per_group)
            groups_list.extend([f"g{i}"] * n_per_group)
            ys.append(y_g)
            bases.append(b)
        y = np.concatenate(ys)
        base = np.concatenate(bases)
        groups = np.asarray(groups_list)
        params = _linear_residual_grouped_fit(y, base, groups=groups)
        T = _linear_residual_grouped_forward(y, base, params, groups=groups)
        y_back = _linear_residual_grouped_inverse(T, base, params, groups=groups)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_unseen_group_falls_back_to_global_at_predict(self) -> None:
        """A row with a group label NOT seen during fit must use the
        global (alpha, beta) at predict, not raise."""
        rng = np.random.default_rng(33)
        n = 300
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 0.9 * base + rng.normal(scale=0.2, size=n)
        groups_fit = np.asarray(["A"] * n)
        params = _linear_residual_grouped_fit(y, base, groups=groups_fit)
        # Predict on a single new group 'Z'.
        base_pred = np.array([8.0, 10.0, 12.0])
        groups_pred = np.asarray(["Z", "Z", "A"])
        T_hat = np.array([0.5, 0.5, 0.5])  # placeholder T-scale preds
        y_back = _linear_residual_grouped_inverse(
            T_hat, base_pred, params, groups=groups_pred,
        )
        # Group 'A' row uses A's params; 'Z' rows use global. With one
        # group fit, A == global so all three look identical (up to T).
        assert y_back.shape == (3,)
        assert np.all(np.isfinite(y_back))


# ---------------------------------------------------------------------------
# James-Stein shrinkage factor
# ---------------------------------------------------------------------------


class TestShrinkage:
    def test_less_than_four_groups_returns_zero(self) -> None:
        """JS only applies for K >= 4 (the classic threshold)."""
        for k in (1, 2, 3):
            c = _james_stein_shrinkage_factor(
                np.array([1.0, 2.0, 3.0][:k]),
                global_alpha=2.0,
                group_sizes=np.array([100.0] * k),
                sigma2_total=1.0,
            )
            assert c == 0.0

    def test_zero_variance_alphas_returns_zero(self) -> None:
        """All alphas identical -> no spread to shrink toward."""
        c = _james_stein_shrinkage_factor(
            np.array([0.5, 0.5, 0.5, 0.5]),
            global_alpha=0.5,
            group_sizes=np.array([100.0, 100.0, 100.0, 100.0]),
            sigma2_total=1.0,
        )
        assert c == 0.0

    def test_high_noise_low_spread_shrinks_to_one(self) -> None:
        """When alphas barely differ but variance is huge, JS shrinks
        almost fully to global."""
        c = _james_stein_shrinkage_factor(
            np.array([0.95, 0.96, 0.94, 0.97, 0.95, 0.96]),
            global_alpha=0.955,
            group_sizes=np.array([100.0] * 6),
            sigma2_total=1000.0,  # huge noise
        )
        assert c > 0.5

    def test_low_noise_high_spread_no_shrinkage(self) -> None:
        """When alphas are clearly different and noise is small, JS
        leaves them alone (c=0)."""
        c = _james_stein_shrinkage_factor(
            np.array([0.5, 0.9, 1.3, 1.7]),
            global_alpha=1.1,
            group_sizes=np.array([1000.0] * 4),
            sigma2_total=0.01,
        )
        assert c < 0.05


# ---------------------------------------------------------------------------
# Registry / domain
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_with_requires_groups_flag(self) -> None:
        t = get_transform("linear_residual_grouped")
        assert t.requires_groups is True

    def test_other_transforms_do_not_require_groups(self) -> None:
        for name in ("diff", "ratio", "logratio", "linear_residual",
                     "linear_residual_multi"):
            t = get_transform(name)
            assert t.requires_groups is False, (
                f"transform '{name}' should not require groups"
            )

    def test_domain_check_delegates_to_linear_residual(self) -> None:
        y = np.array([1.0, 2.0, np.nan])
        base = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            _linear_residual_grouped_domain(y, base),
            np.array([True, True, False]),
        )


# ---------------------------------------------------------------------------
# Biz_value: grouped beats single on per-well DGP
# ---------------------------------------------------------------------------


class TestBizValueGroupedBeatsSingle:
    """Per-well DGP: well A has alpha=0.95, well B alpha=0.55, well C
    alpha=1.05. A global-alpha fit averages out these heterogeneity
    leaving systematic residual; grouped fit captures them.

    Lock: variance of T under grouped fit must be STRICTLY lower than
    under single-alpha linear_residual on the same data.
    """

    def _make_per_well_dgp(self, n_per_well: int = 500, seed: int = 0):
        rng = np.random.default_rng(seed)
        alphas = {"A": 0.95, "B": 0.55, "C": 1.05}
        ys = []
        bases = []
        groups = []
        for well, a in alphas.items():
            b = rng.normal(loc=10.0, scale=2.0, size=n_per_well)
            y = a * b + 1.0 + rng.normal(scale=0.2, size=n_per_well)
            ys.append(y)
            bases.append(b)
            groups.extend([well] * n_per_well)
        return (np.concatenate(ys), np.concatenate(bases),
                np.asarray(groups))

    def test_grouped_residual_variance_strictly_lower(self) -> None:
        y, base, groups = self._make_per_well_dgp()
        single = get_transform("linear_residual")
        grouped = get_transform("linear_residual_grouped")
        p_single = single.fit(y, base)
        T_single = single.forward(y, base, p_single)
        p_grouped = grouped.fit(y, base, groups=groups)
        T_grouped = grouped.forward(y, base, p_grouped, groups=groups)
        var_single = float(np.var(T_single))
        var_grouped = float(np.var(T_grouped))
        # Single-alpha fit residual carries the per-well coefficient
        # spread; var should be substantial relative to noise (~0.04).
        # Grouped residual is just per-well noise (~0.04).
        assert var_grouped < var_single * 0.5, (
            f"grouped residual variance must be << single's; "
            f"single={var_single:.4f}, grouped={var_grouped:.4f}"
        )

    def test_round_trip_preserves_y_on_per_well_dgp(self) -> None:
        y, base, groups = self._make_per_well_dgp()
        grouped = get_transform("linear_residual_grouped")
        p = grouped.fit(y, base, groups=groups)
        T = grouped.forward(y, base, p, groups=groups)
        y_back = grouped.inverse(T, base, p, groups=groups)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_per_group_alphas_close_to_dgp_truth(self) -> None:
        """Sanity check: the fitted per-group alphas should approximate
        the DGP truth on large n_per_well, even after JS shrinkage
        (shrinkage should be tiny because between-group variance is
        much bigger than within-group noise)."""
        y, base, groups = self._make_per_well_dgp(n_per_well=1000)
        grouped = get_transform("linear_residual_grouped")
        p = grouped.fit(y, base, groups=groups)
        for well, true_alpha in [("A", 0.95), ("B", 0.55), ("C", 1.05)]:
            fitted = p["per_group_alphas"][well]
            assert abs(fitted - true_alpha) < 0.05, (
                f"well {well}: fitted alpha {fitted:.3f} too far from "
                f"DGP truth {true_alpha:.2f}"
            )
        # With 3 groups, JS shrinkage is 0 by design.
        assert p["shrinkage_factor"] == 0.0


# ---------------------------------------------------------------------------
# T8 regression: beta must be re-centred when alpha is JS-shrunk
# ---------------------------------------------------------------------------


class TestShrinkageRecentersBeta:
    """When James-Stein shrinkage tilts ``alpha_g`` toward the global
    slope, the per-group intercept ``beta_g`` MUST be re-centred so the
    per-group training residual mean stays at 0.

    The pre-fix code shrank ``alpha_g`` but left ``beta_g`` at its
    pre-shrink OLS value. Because the per-group OLS line pivots about
    ``base = 0`` (not about ``mean(base_g)``), shrinking only the slope
    moves the line off the group centroid and the per-group residual
    mean drifts to ``c·(a_g - alpha_global)·mean(base_g)`` -- a
    manufactured, base-mean-proportional per-group bias.

    This needs K >= 4 groups (JS threshold) AND high within-group noise
    relative to the slope spread (so ``c > 0`` actually fires) AND a base
    with a non-zero mean (so the un-re-centred bias is non-zero). The
    existing 3-group biz_value DGP keeps ``c == 0``, which is exactly why
    the bug stayed dormant there.
    """

    def _make_shrinking_dgp(self, n_per_group: int = 120, seed: int = 0):
        """6 groups, moderately-spread slopes with sizeable within-group
        noise -> JS PARTIALLY shrinks (0 < c < 1, ~0.54 at seed 0). A
        partial shrink is the most discriminating case: it proves the
        beta re-centring scales with ``c·(a_g - alpha_global)``, not just
        the full-shrink edge. base ~ N(10, 2) so mean(base_g) is far from
        0 and the un-re-centred bias is large.
        """
        rng = np.random.default_rng(seed)
        true_alphas = [0.30, 0.70, 0.50, 0.90, 0.40, 0.80]
        ys, bases, groups = [], [], []
        for i, a in enumerate(true_alphas):
            b = rng.normal(loc=10.0, scale=2.0, size=n_per_group)
            # Sizeable noise so per-group slope estimates are uncertain ->
            # James-Stein shrinks them toward the global slope.
            y_g = a * b + 1.0 + rng.normal(scale=3.0, size=n_per_group)
            ys.append(y_g)
            bases.append(b)
            groups.extend([f"g{i}"] * n_per_group)
        return (np.concatenate(ys), np.concatenate(bases),
                np.asarray(groups))

    def test_shrinkage_actually_fires_on_this_dgp(self) -> None:
        """Guard the regression: if the DGP stops triggering c > 0 the
        bias test below would pass vacuously."""
        y, base, groups = self._make_shrinking_dgp()
        p = _linear_residual_grouped_fit(y, base, groups=groups)
        assert p["shrinkage_factor"] > 0.0, (
            "DGP must trigger James-Stein shrinkage for this regression "
            f"to be meaningful; got c={p['shrinkage_factor']}"
        )
        # A partial shrink (0 < c < 1) is the most discriminating case:
        # alphas actually move (otherwise re-centring is a no-op) and the
        # re-centring must scale with c, not just the full-shrink edge.
        assert 0.0 < p["shrinkage_factor"] < 1.0

    def test_per_group_residual_mean_is_zero_after_shrink(self) -> None:
        """Core T8 assertion: for every eligible (own-OLS) group, the
        mean training residual ``mean(y_g - alpha_g·base_g - beta_g)``
        must be ~0 using the FITTED (post-shrink) alpha/beta. Pre-fix,
        these means were ``c·(a_g_raw - alpha_global)·mean(base_g)`` and
        the assertion fails."""
        y, base, groups = self._make_shrinking_dgp()
        p = _linear_residual_grouped_fit(y, base, groups=groups)
        assert p["shrinkage_factor"] > 0.0  # sanity
        pg_a = p["per_group_alphas"]
        pg_b = p["per_group_betas"]
        sizes = p["group_sizes"]
        min_gs = p["min_group_size"]
        uniq = np.unique(groups)
        for g in uniq:
            g_key = str(g)
            if sizes[g_key] < min_gs:
                continue  # global-fallback group: no own OLS to re-centre
            mask = groups == g
            resid = y[mask] - pg_a[g_key] * base[mask] - pg_b[g_key]
            mean_resid = float(np.mean(resid))
            # Scale tolerance to the target spread; the manufactured bias
            # is O(c · Δalpha · mean(base)) ~ several units here.
            assert abs(mean_resid) < 1e-6, (
                f"group {g_key}: per-group residual mean must be ~0 after "
                f"JS shrinkage (beta re-centred), got {mean_resid:.6g}"
            )

    def test_round_trip_still_exact_after_shrink(self) -> None:
        """Re-centring beta must not break the forward/inverse round
        trip (forward and inverse share the same alpha/beta)."""
        y, base, groups = self._make_shrinking_dgp()
        p = _linear_residual_grouped_fit(y, base, groups=groups)
        T = _linear_residual_grouped_forward(y, base, p, groups=groups)
        y_back = _linear_residual_grouped_inverse(T, base, p, groups=groups)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_biz_val_recentred_beta_lowers_global_residual_mean_bias(self) -> None:
        """biz_value: re-centring removes a measurable per-group bias.
        The RMS of per-group residual means (a direct measure of the
        manufactured bias) must be near machine-zero with the fix; the
        pre-fix code leaves it at several target units on this DGP."""
        y, base, groups = self._make_shrinking_dgp(seed=3)
        p = _linear_residual_grouped_fit(y, base, groups=groups)
        pg_a = p["per_group_alphas"]
        pg_b = p["per_group_betas"]
        sizes = p["group_sizes"]
        min_gs = p["min_group_size"]
        per_group_mean_bias = []
        for g in np.unique(groups):
            g_key = str(g)
            if sizes[g_key] < min_gs:
                continue
            mask = groups == g
            resid = y[mask] - pg_a[g_key] * base[mask] - pg_b[g_key]
            per_group_mean_bias.append(float(np.mean(resid)))
        rms_bias = float(np.sqrt(np.mean(np.square(per_group_mean_bias))))
        # Measured pre-fix on this DGP: rms_bias ~ O(1) target units.
        # Post-fix it collapses to ~1e-14. Floor well below the pre-fix
        # value but above FP noise.
        assert rms_bias < 1e-3, (
            f"RMS of per-group residual-mean bias must be ~0 after "
            f"re-centring; got {rms_bias:.6g}"
        )


# ---------------------------------------------------------------------------
# Wrapper integration: CompositeTargetEstimator with group_column
# ---------------------------------------------------------------------------

import pandas as pd

lgb = pytest.importorskip("lightgbm")


class TestCompositeTargetEstimatorGrouped:
    def _make_per_well_df(self, n_per_well: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        rows = []
        for well_id, alpha in (("A", 0.95), ("B", 0.55), ("C", 1.05)):
            b = rng.normal(loc=10.0, scale=2.0, size=n_per_well)
            x_other = rng.normal(size=n_per_well)
            y = alpha * b + 0.2 * x_other + rng.normal(scale=0.2, size=n_per_well)
            for i in range(n_per_well):
                rows.append({"well_id": well_id, "b1": b[i],
                              "x_other": x_other[i], "y": y[i]})
        df = pd.DataFrame(rows)
        return df.drop(columns="y"), df["y"].to_numpy()

    def test_fit_predict_round_trip(self) -> None:
        from mlframe.training.composite import CompositeTargetEstimator
        X, y = self._make_per_well_df(n_per_well=200)
        inner = lgb.LGBMRegressor(
            n_estimators=40, num_leaves=15, verbose=-1, random_state=0,
        )
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="linear_residual_grouped",
            base_column="b1",
            group_column="well_id",
        )
        wrap.fit(X, y)
        preds = wrap.predict(X)
        assert preds.shape == (len(X),)
        assert np.all(np.isfinite(preds))
        rmse = float(np.sqrt(np.mean((preds - y) ** 2)))
        train_std = float(np.std(y))
        assert rmse < train_std * 0.5

    def test_requires_group_column(self) -> None:
        """linear_residual_grouped without group_column must error at fit."""
        from mlframe.training.composite import CompositeTargetEstimator
        X, y = self._make_per_well_df(n_per_well=100)
        inner = lgb.LGBMRegressor(n_estimators=10, verbose=-1)
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="linear_residual_grouped",
            base_column="b1",
            group_column=None,
        )
        with pytest.raises(ValueError, match="requires groups"):
            wrap.fit(X, y)

    def test_grouped_wrapper_beats_ungrouped_on_per_well_dgp(self) -> None:
        """Biz_value: same inner LGB, same data, only transform differs.
        Grouped wrapper must produce lower test RMSE than ungrouped."""
        from mlframe.training.composite import CompositeTargetEstimator
        X, y = self._make_per_well_df(n_per_well=400)
        train_X = X.iloc[:1000]
        train_y = y[:1000]
        test_X = X.iloc[1000:]
        test_y = y[1000:]

        def _rmse(transform_name, group_column):
            inner = lgb.LGBMRegressor(
                n_estimators=60, num_leaves=15, verbose=-1, random_state=0,
            )
            wrap = CompositeTargetEstimator(
                base_estimator=inner,
                transform_name=transform_name,
                base_column="b1",
                group_column=group_column,
            )
            # When the transform does NOT require groups, manually drop
            # the string well_id column so LGB can fit -- mimics what
            # the production suite does upstream (categorical encoding
            # or column drop) before reaching the wrapper.
            tx = train_X if group_column else train_X.drop(columns=["well_id"])
            ex = test_X if group_column else test_X.drop(columns=["well_id"])
            wrap.fit(tx, train_y)
            preds = wrap.predict(ex)
            return float(np.sqrt(np.mean((preds - test_y) ** 2)))

        rmse_single = _rmse("linear_residual", None)
        rmse_grouped = _rmse("linear_residual_grouped", "well_id")
        assert rmse_grouped < rmse_single, (
            f"grouped wrapper must beat single-alpha on per-well DGP; "
            f"single={rmse_single:.4f}, grouped={rmse_grouped:.4f}"
        )

    def test_predict_on_unseen_well_falls_back_to_global(self) -> None:
        """Wrapper.predict on a row with a well_id not seen at fit must
        not raise; the row uses global alpha + beta."""
        from mlframe.training.composite import CompositeTargetEstimator
        X, y = self._make_per_well_df(n_per_well=200)
        inner = lgb.LGBMRegressor(
            n_estimators=20, verbose=-1, random_state=0,
        )
        wrap = CompositeTargetEstimator(
            base_estimator=inner,
            transform_name="linear_residual_grouped",
            base_column="b1",
            group_column="well_id",
        )
        wrap.fit(X, y)
        # Construct a predict frame with a NEW well_id 'Z'.
        new_rows = pd.DataFrame({
            "well_id": ["Z", "Z", "A"],
            "b1": [10.0, 11.0, 9.0],
            "x_other": [0.0, 0.5, -0.5],
        })
        preds = wrap.predict(new_rows)
        assert preds.shape == (3,)
        assert np.all(np.isfinite(preds))
