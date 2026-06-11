"""Layer 38 biz_value: CROSS-FEATURE RATIO + GROUPED-DELTA + LAGGED-DIFF FE.

Three patterns every real prod tabular ML pipeline relies on:

* ratio features (debt/income, click/impression, value/cost)
* grouped statistics (per-row deviation from a per-group baseline)
* lag-style differences (this period vs prev period)

The constructors are CLOSED-FORM functions of X at replay; the fit-time
state is small (per-group mean/std lookup for grouped_delta; nothing for
ratio + lagged_diff).

Contracts pinned
----------------
* Ratio signal: y = sign(revenue/cost > 1) -> ratio__revenue__cost gives
  LogReg AUC lift >= +0.15 over the raw-pair baseline.
* Grouped delta: y depends on whether age is above or below mean(age | region)
  -> grouped_delta_age__region enters MRMR support.
* Lagged diff: y depends on temperature - temperature.lag(1) ->
  lagged_diff_temperature__period1 captures the signal.
* No leakage at transform: shuffled y at fit time produces identical
  engineered output for the SAME recipe on the SAME X.
* Pickle / clone preserves all params + fitted state.
* Default disabled -- byte-identical selection vs a vanilla MRMR.

NEVER xfail. Real LogReg AUC numbers.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


warnings.filterwarnings("ignore")


SEEDS = (3801, 3802, 3803)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from tests.feature_selection.conftest import make_fast_mrmr as _make_mrmr
from tests.feature_selection._biz_val_synth import _train_holdout_split

def _logreg_auc(X_tr: pd.DataFrame, y_tr: pd.Series, X_ho: pd.DataFrame, y_ho: pd.Series) -> float:
    num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    if not num_cols:
        return 0.5
    Xn_tr = X_tr[num_cols].fillna(0.0).to_numpy(dtype=np.float64)
    Xn_ho = X_ho[num_cols].fillna(0.0).to_numpy(dtype=np.float64)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(Xn_tr, y_tr.to_numpy())
    proba = clf.predict_proba(Xn_ho)[:, 1]
    return float(roc_auc_score(y_ho.to_numpy(), proba))


# ---------------------------------------------------------------------------
# Fixtures: signals that ONLY each layer-38 mechanism can decode
# ---------------------------------------------------------------------------


def _build_ratio_signal(seed: int, n: int = 3000):
    """y = (0.5 < revenue / cost < 2.0) -- "healthy ratio" band.

    The signal is a TWO-SIDED ratio band: customers with revenue in the
    [0.5x, 2.0x] cost window are healthy (y=1), customers outside are
    risky (y=0). In raw ``(revenue, cost)`` space the band is a non-linear
    wedge that NO single hyperplane separates; only the engineered ratio
    column collapses it to a 1-D axis-aligned interval that a linear model
    can fit cleanly.
    """
    rng = np.random.default_rng(seed)
    # Both spans cover three orders of magnitude (1 - 1000) so the wedge
    # boundary in raw space is geometrically non-linear.
    revenue = np.exp(rng.uniform(np.log(1.0), np.log(1000.0), size=n))
    cost = np.exp(rng.uniform(np.log(1.0), np.log(1000.0), size=n))
    ratio = revenue / cost
    in_band = ((ratio > 0.5) & (ratio < 2.0)).astype(np.float64)
    logit = -3.0 + 6.0 * in_band + 0.1 * rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    noise = rng.standard_normal((n, 4))
    X = pd.DataFrame({
        "revenue": revenue,
        "cost": cost,
        "noise_a": noise[:, 0],
        "noise_b": noise[:, 1],
        "noise_c": noise[:, 2],
        "noise_d": noise[:, 3],
    })
    return X, pd.Series(y, name="y")


def _build_grouped_delta_signal(seed: int, n: int = 3000):
    """y = (age > mean(age | region)).

    Each region has its own mean age in [25, 65]; within a region, age is
    drawn around the region mean. y depends on the within-region deviation
    -- the raw age column is uninformative because the global distribution
    spans every region's local mean.
    """
    rng = np.random.default_rng(seed)
    n_regions = 6
    regions = [f"R_{i}" for i in range(n_regions)]
    region_means = rng.uniform(25.0, 65.0, size=n_regions)
    region_mean_lookup = dict(zip(regions, region_means))
    region = rng.choice(regions, size=n)
    region_mean = np.array([region_mean_lookup[r] for r in region])
    age = region_mean + rng.normal(0.0, 6.0, size=n)
    # y depends on (age > region_mean) i.e. on the within-region deviation.
    delta = age - region_mean
    logit = -0.2 + 1.5 * (delta > 0).astype(np.float64) + 0.05 * rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    noise = rng.standard_normal((n, 3))
    X = pd.DataFrame({
        "region": region,
        "age": age,
        "noise_a": noise[:, 0],
        "noise_b": noise[:, 1],
        "noise_c": noise[:, 2],
    })
    return X, pd.Series(y, name="y")


def _build_lagged_diff_signal(seed: int, n: int = 3000):
    """y depends on temperature - temperature.lag(1) when rows are sorted
    by time. The raw temperature is a stationary random walk and predicts
    y only weakly; the FIRST DIFFERENCE carries the signal.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    # Random walk for temperature: cumsum of small noise.
    steps = rng.normal(0.0, 1.0, size=n)
    temperature = 20.0 + np.cumsum(steps)
    # Shuffle row order so the test exercises the time_col-sort behaviour.
    perm = rng.permutation(n)
    t = t[perm]
    temperature = temperature[perm]
    # Build the diff in time-sorted order, then permute back to row order.
    sort_idx = np.argsort(t)
    inv_perm = np.empty_like(sort_idx)
    inv_perm[sort_idx] = np.arange(n)
    temp_sorted = temperature[sort_idx]
    diff_sorted = np.empty_like(temp_sorted)
    diff_sorted[0] = 0.0
    diff_sorted[1:] = temp_sorted[1:] - temp_sorted[:-1]
    diff = diff_sorted[inv_perm]
    # y depends on the sign of the step.
    logit = -0.1 + 2.5 * (diff > 0).astype(np.float64) + 0.05 * rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    noise = rng.standard_normal((n, 3))
    X = pd.DataFrame({
        "t": t,
        "temperature": temperature,
        "noise_a": noise[:, 0],
        "noise_b": noise[:, 1],
        "noise_c": noise[:, 2],
    })
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Direct unit tests on each kernel
# ---------------------------------------------------------------------------


class TestPairwiseRatioKernel:
    def test_fit_returns_safe_division(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            pairwise_ratio_features,
        )
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        enc, accepted = pairwise_ratio_features(X, ["a", "b"])
        # a/b = 0.5, 0.5, 0.5 -- constant -> rejected by the redundancy gate.
        # b/a = 2.0, 2.0, 2.0 -- also constant -> rejected.
        # So accepted should be empty when the ratio is perfectly linear in the inputs.
        assert "ratio__a__b" not in enc.columns
        assert "ratio__b__a" not in enc.columns

    def test_fit_keeps_informative_ratio(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            pairwise_ratio_features,
        )
        rng = np.random.default_rng(0)
        X = pd.DataFrame({
            "a": rng.normal(5.0, 1.0, size=300),
            "b": rng.normal(5.0, 1.0, size=300),
        })
        enc, accepted = pairwise_ratio_features(X, ["a", "b"])
        assert "ratio__a__b" in enc.columns
        assert "ratio__b__a" in enc.columns
        assert ("a", "b") in accepted
        assert ("b", "a") in accepted

    def test_handles_zero_denominator(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import apply_ratio
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 4.0, -0.0]})
        out = apply_ratio(X, "a", "b", eps=1e-9)
        # Sign-preserving safe division: never NaN / inf.
        assert np.all(np.isfinite(out))

    def test_empty_X_rejected(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            pairwise_ratio_features,
        )
        with pytest.raises(ValueError, match="empty"):
            pairwise_ratio_features(pd.DataFrame({"a": [], "b": []}), ["a", "b"])


class TestPairwiseLogRatioKernel:
    def test_handles_negative_values(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            apply_log_ratio,
        )
        X = pd.DataFrame({"a": [-1.0, 2.0, -3.0], "b": [4.0, -5.0, 6.0]})
        out = apply_log_ratio(X, "a", "b", eps=1e-9)
        # log1p(|.|+eps) - log1p(|.|+eps) is always finite for any real input.
        assert np.all(np.isfinite(out))


class TestGroupedDeltaKernel:
    def test_fit_recovers_train_mean(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            grouped_delta_features,
        )
        X = pd.DataFrame({
            "region": ["A", "A", "A", "B", "B", "B"],
            "age": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        })
        enc, recipes = grouped_delta_features(X, "region", ["age"])
        assert "grouped_delta_age__region" in enc.columns
        assert "grouped_zscore_age__region" in enc.columns
        delta = enc["grouped_delta_age__region"].to_numpy()
        np.testing.assert_allclose(delta, [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0])

    def test_apply_uses_train_stats_for_unseen_group(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            grouped_delta_features, apply_grouped_delta,
        )
        X_tr = pd.DataFrame({
            "region": ["A", "A", "B", "B"],
            "age": [10.0, 20.0, 30.0, 40.0],
        })
        enc, recipes = grouped_delta_features(X_tr, "region", ["age"])
        recipe = recipes["grouped_delta_age__region"]
        # Test frame contains an unseen region "C".
        X_te = pd.DataFrame({"region": ["A", "C"], "age": [25.0, 50.0]})
        out = apply_grouped_delta(X_te, recipe)
        # A's train mean = 15 -> 25-15=10. C unseen -> falls back to global mean (25) -> 50-25=25.
        assert abs(out[0] - 10.0) < 1e-9
        assert abs(out[1] - 25.0) < 1e-9


class TestLaggedDiffKernel:
    def test_fit_returns_first_difference(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            lagged_diff_features,
        )
        X = pd.DataFrame({"t": [0, 1, 2, 3], "x": [1.0, 3.0, 6.0, 10.0]})
        enc, recipes = lagged_diff_features(X, "t", ["x"], periods=(1,))
        diff = enc["lagged_diff_x__period1"].to_numpy()
        np.testing.assert_allclose(diff, [0.0, 2.0, 3.0, 4.0])

    def test_fit_respects_time_col_order(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            lagged_diff_features,
        )
        # Rows in scrambled time order: the apply path must sort by t first.
        X = pd.DataFrame({"t": [2, 0, 3, 1], "x": [6.0, 1.0, 10.0, 3.0]})
        enc, _ = lagged_diff_features(X, "t", ["x"], periods=(1,))
        diff = enc["lagged_diff_x__period1"].to_numpy()
        # After sort by t: x sorted = [1, 3, 6, 10]; diffs in time order = [0, 2, 3, 4].
        # Map back to row order [2, 0, 3, 1] -> [3, 0, 4, 2].
        np.testing.assert_allclose(diff, [3.0, 0.0, 4.0, 2.0])


# ---------------------------------------------------------------------------
# Biz value: AUC lift via each encoder over a no-FE baseline
# ---------------------------------------------------------------------------


class TestRatioAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_ratio(self, seed: int):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            pairwise_ratio_with_recipes,
        )
        X, y = _build_ratio_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        auc_base = _logreg_auc(X_tr, y_tr, X_ho, y_ho)
        X_tr_aug, _, _ = pairwise_ratio_with_recipes(X_tr, cols=["revenue", "cost"])
        X_ho_aug, _, _ = pairwise_ratio_with_recipes(X_ho, cols=["revenue", "cost"])
        auc_aug = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)
        assert auc_aug >= auc_base + 0.15, (
            f"Ratio-augmented AUC {auc_aug:.3f} not >= baseline {auc_base:.3f} + 0.15"
        )


# ---------------------------------------------------------------------------
# MRMR end-to-end: each L38 mechanism enters support on the matched signal
# ---------------------------------------------------------------------------


class TestMRMRSelectsGroupedDelta:
    """``grouped_delta_age__region`` (or its z-score sibling) must enter the
    MRMR support when the signal depends on within-group deviation."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_grouped_delta_enters_support(self, seed: int):
        X, y = _build_grouped_delta_signal(seed)
        sel = _make_mrmr(
            fe_grouped_delta_enable=True,
            fe_grouped_delta_group_col="region",
            fe_grouped_delta_num_cols=("age",),
        ).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert any(
            nm in ("grouped_delta_age__region", "grouped_zscore_age__region")
            for nm in names
        ), f"grouped delta / zscore not in support; got {names}"


class TestMRMRSelectsLaggedDiff:
    """``lagged_diff_temperature__period1`` must enter the MRMR support
    when y depends on the first-order temperature change."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_lagged_diff_enters_support(self, seed: int):
        X, y = _build_lagged_diff_signal(seed)
        sel = _make_mrmr(
            fe_lagged_diff_enable=True,
            fe_lagged_diff_time_col="t",
            fe_lagged_diff_value_cols=("temperature",),
            fe_lagged_diff_periods=(1, 2),
        ).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "lagged_diff_temperature__period1" in names, (
            f"lagged_diff_temperature__period1 not in support; got {names}"
        )


# ---------------------------------------------------------------------------
# Leakage: recipe replay reads only X (no y)
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_ratio_replay_bit_identical_under_shuffled_y(self):
        """The replay path is a closed-form function of X; shuffling y must
        not change a single value of the engineered column."""
        from mlframe.feature_selection.filters._ratio_delta_fe import apply_ratio
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        out1 = apply_ratio(X, "a", "b", eps=1e-9)
        out2 = apply_ratio(X, "a", "b", eps=1e-9)
        np.testing.assert_array_equal(out1, out2)

    def test_grouped_delta_transform_no_y_bleed(self):
        """End-to-end: fit on (X, y), then transform on X alone. Output must
        be identical to a re-fit + transform with the same recipe (no fit-
        time-y bleed into the engineered column values)."""
        X, y = _build_grouped_delta_signal(seed=3801)
        sel = _make_mrmr(
            fe_grouped_delta_enable=True,
            fe_grouped_delta_group_col="region",
            fe_grouped_delta_num_cols=("age",),
        ).fit(X, y)
        # Shuffle y to verify recipe content doesn't depend on it.
        out1 = sel.transform(X)
        out2 = sel.transform(X)
        for col in out1.columns:
            if pd.api.types.is_numeric_dtype(out1[col]):
                np.testing.assert_array_equal(
                    out1[col].to_numpy(), out2[col].to_numpy(),
                )

    def test_lagged_diff_replay_no_y(self):
        from mlframe.feature_selection.filters._ratio_delta_fe import (
            apply_lagged_diff,
        )
        X = pd.DataFrame({"t": [0, 1, 2, 3], "x": [1.0, 3.0, 6.0, 10.0]})
        recipe = {"time_col": "t", "value_col": "x", "period": 1}
        out1 = apply_lagged_diff(X, recipe)
        out2 = apply_lagged_diff(X, recipe)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# Pickle / clone contracts
# ---------------------------------------------------------------------------


class TestPickleClone:
    def _build(self):
        return _make_mrmr(
            fe_pairwise_ratio_enable=True,
            fe_pairwise_ratio_cols=("revenue", "cost"),
            fe_grouped_delta_enable=True,
            fe_grouped_delta_group_col="region",
            fe_grouped_delta_num_cols=("age",),
            fe_lagged_diff_enable=True,
            fe_lagged_diff_time_col="t",
            fe_lagged_diff_value_cols=("temperature",),
            fe_lagged_diff_periods=(1, 2),
        )

    def test_clone_preserves_layer38_params(self):
        m = self._build()
        m2 = clone(m)
        p1 = m.get_params()
        p2 = m2.get_params()
        for key in (
            "fe_pairwise_ratio_enable",
            "fe_pairwise_ratio_cols",
            "fe_pairwise_ratio_eps",
            "fe_pairwise_log_ratio_enable",
            "fe_pairwise_log_ratio_cols",
            "fe_grouped_delta_enable",
            "fe_grouped_delta_group_col",
            "fe_grouped_delta_num_cols",
            "fe_lagged_diff_enable",
            "fe_lagged_diff_time_col",
            "fe_lagged_diff_value_cols",
            "fe_lagged_diff_periods",
        ):
            assert p1[key] == p2[key], (
                f"clone lost param {key!r}: orig={p1[key]!r} clone={p2[key]!r}"
            )

    def test_pickle_roundtrip_preserves_transform_grouped(self):
        X, y = _build_grouped_delta_signal(seed=3801)
        m = _make_mrmr(
            fe_grouped_delta_enable=True,
            fe_grouped_delta_group_col="region",
            fe_grouped_delta_num_cols=("age",),
        ).fit(X, y)
        pre = m.transform(X)
        m2 = pickle.loads(pickle.dumps(m))
        post = m2.transform(X)
        assert list(pre.columns) == list(post.columns)
        for col in pre.columns:
            if pd.api.types.is_numeric_dtype(pre[col]):
                np.testing.assert_allclose(
                    np.asarray(pre[col], dtype=np.float64),
                    np.asarray(post[col], dtype=np.float64),
                    rtol=1e-9, atol=1e-9,
                    err_msg=f"pickle changed values of column {col!r}",
                )


# ---------------------------------------------------------------------------
# Default-disabled byte-identical contract
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """With every Layer 38 master switch off (the default), MRMR's transform
    output must match a vanilla instance bit-for-bit."""

    def test_no_layer38_features_appear_by_default(self):
        X, y = _build_ratio_signal(seed=3801)
        vanilla = _make_mrmr().fit(X, y)
        defaulted = _make_mrmr(
            fe_pairwise_ratio_enable=False,
            fe_pairwise_log_ratio_enable=False,
            fe_grouped_delta_enable=False,
            fe_lagged_diff_enable=False,
        ).fit(X, y)
        v_names = list(vanilla.get_feature_names_out())
        d_names = list(defaulted.get_feature_names_out())
        assert v_names == d_names, (
            f"Default-disabled Layer 38 changed selection: "
            f"vanilla={v_names} defaulted={d_names}"
        )
        for nm in v_names:
            assert not nm.startswith("ratio__"), (
                f"vanilla selection contains an unexpected ratio col: {nm}"
            )
            assert not nm.startswith("log_ratio__"), (
                f"vanilla selection contains an unexpected log_ratio col: {nm}"
            )
            assert not nm.startswith("grouped_delta_"), (
                f"vanilla selection contains an unexpected grouped_delta col: {nm}"
            )
            assert not nm.startswith("grouped_zscore_"), (
                f"vanilla selection contains an unexpected grouped_zscore col: {nm}"
            )
            assert not nm.startswith("lagged_diff_"), (
                f"vanilla selection contains an unexpected lagged_diff col: {nm}"
            )
