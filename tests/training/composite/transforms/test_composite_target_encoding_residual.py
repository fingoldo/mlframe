"""Tests for ``target_encoding_residual`` -- high-cardinality categorical
target-encoding residual transform.

Coverage:
- fit requires the ``groups`` (category) array; raises without it.
- Empirical-Bayes smoothing: a singleton category shrinks toward the global
  mean; a well-populated one keeps (near) its own raw mean.
- Round-trip ``y -> T -> y'`` (rtol=1e-7).
- Unseen category at predict falls back to the global mean (finite output).
- domain_check: finite-y gate at fit; all-True at predict (y=None).
- Registry: registered with ``requires_groups=True`` / ``requires_base=False``.
- biz_value: on a target driven by a 50-level high-cardinality categorical +
  noise, the target-encoding residual + a linear model beats raw-y linear by a
  clear RMSE margin, and an unseen-category predict stays finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import (
    CompositeTargetEstimator,
    compose_target_name,
    get_transform,
    is_composite_target_name,
)
from mlframe.training.composite.transforms.categorical import (
    _TARGET_ENCODING_DEFAULT_SMOOTHING,
    _target_encoding_residual_domain,
    _target_encoding_residual_fit,
    _target_encoding_residual_forward,
    _target_encoding_residual_inverse,
)


# ---------------------------------------------------------------------------
# Unit: fit / forward / inverse / smoothing
# ---------------------------------------------------------------------------


class TestFit:
    """Groups tests covering fit."""
    def test_requires_groups_argument(self) -> None:
        """Requires groups argument."""
        rng = np.random.default_rng(0)
        y = rng.normal(size=50)
        base = np.zeros_like(y)
        with pytest.raises(ValueError, match="requires a 1-D ``groups``"):
            _target_encoding_residual_fit(y, base, groups=None)

    def test_groups_length_mismatch_raises(self) -> None:
        """Groups length mismatch raises."""
        y = np.arange(10.0)
        with pytest.raises(ValueError, match="groups has"):
            _target_encoding_residual_fit(
                y,
                np.zeros(10),
                groups=np.array(["a"] * 9),
            )

    def test_negative_smoothing_raises(self) -> None:
        """Negative smoothing raises."""
        y = np.arange(10.0)
        groups = np.array(["a"] * 10)
        with pytest.raises(ValueError, match="non-negative"):
            _target_encoding_residual_fit(y, np.zeros(10), groups=groups, smoothing=-1.0)

    def test_zero_smoothing_is_raw_category_mean(self) -> None:
        """``a = 0`` -> the encoding is the raw per-category mean."""
        y = np.array([1.0, 3.0, 10.0, 20.0])
        groups = np.array(["a", "a", "b", "b"])
        p = _target_encoding_residual_fit(y, np.zeros(4), groups=groups, smoothing=0.0)
        assert p["encoding"]["a"] == pytest.approx(2.0)
        assert p["encoding"]["b"] == pytest.approx(15.0)
        assert p["global_mean"] == pytest.approx(8.5)

    def test_singleton_category_shrinks_toward_global(self) -> None:
        """A tiny (singleton) category must be pulled toward the global mean by
        the smoothing prior; a well-populated category stays near its own mean."""
        rng = np.random.default_rng(1)
        # Big category 'A': 1000 rows around 0.0; singleton 'Z': one extreme row.
        y_big = rng.normal(loc=0.0, scale=1.0, size=1000)
        y = np.concatenate([y_big, np.array([100.0])])
        groups = np.asarray(["A"] * 1000 + ["Z"])
        a = 20.0
        p = _target_encoding_residual_fit(y, np.zeros_like(y), groups=groups, smoothing=a)
        global_mean = p["global_mean"]
        enc_z = p["encoding"]["Z"]
        enc_a = p["encoding"]["A"]
        # Singleton 'Z' raw mean is 100 but smoothed = (100 + 20*gm)/(1+20),
        # i.e. heavily pulled toward the global mean (far below 100).
        assert enc_z < 0.5 * 100.0, "singleton must be shrunk toward global mean"
        assert abs(enc_z - global_mean) < abs(100.0 - global_mean), "singleton encoding must be closer to global mean than its raw mean"
        # Big category 'A' (1000 rows) barely shrinks -> stays near its own mean.
        raw_a = float(np.mean(y_big))
        assert abs(enc_a - raw_a) < 0.05, "large category keeps ~its own mean"

    def test_round_trip_y_to_T_to_y(self) -> None:
        """Round trip y to t to y."""
        rng = np.random.default_rng(11)
        n = 2000
        cats = np.asarray([f"c{i}" for i in rng.integers(0, 50, size=n)])
        y = rng.normal(size=n) + rng.normal(size=n)
        p = _target_encoding_residual_fit(y, np.zeros(n), groups=cats)
        T = _target_encoding_residual_forward(y, np.zeros(n), p, groups=cats)
        y_back = _target_encoding_residual_inverse(T, np.zeros(n), p, groups=cats)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_unseen_category_uses_global_mean(self) -> None:
        """Unseen category uses global mean."""
        rng = np.random.default_rng(33)
        n = 300
        y = rng.normal(loc=5.0, size=n)
        groups_fit = np.asarray(["A"] * n)
        p = _target_encoding_residual_fit(y, np.zeros(n), groups=groups_fit)
        global_mean = p["global_mean"]
        # Predict on unseen category 'Z' + seen 'A'.
        t_hat = np.array([0.0, 0.0, 0.0])
        groups_pred = np.asarray(["Z", "Z", "A"])
        y_back = _target_encoding_residual_inverse(
            t_hat,
            np.zeros(3),
            p,
            groups=groups_pred,
        )
        assert np.all(np.isfinite(y_back))
        # Unseen rows invert with the global mean (since T_hat=0).
        assert y_back[0] == pytest.approx(global_mean)
        assert y_back[1] == pytest.approx(global_mean)


# ---------------------------------------------------------------------------
# Registry / domain / naming
# ---------------------------------------------------------------------------


class TestRegistry:
    """Groups tests covering registry."""
    def test_registered_with_flags(self) -> None:
        """Registered with flags."""
        t = get_transform("target_encoding_residual")
        assert t.requires_groups is True
        assert t.requires_base is False

    def test_domain_check_gates_finite_y_at_fit(self) -> None:
        """Domain check gates finite y at fit."""
        y = np.array([1.0, 2.0, np.nan])
        base = np.zeros(3)
        np.testing.assert_array_equal(
            _target_encoding_residual_domain(y, base),
            np.array([True, True, False]),
        )

    def test_domain_check_all_true_at_predict(self) -> None:
        """Domain check all true at predict."""
        base = np.zeros(4)
        np.testing.assert_array_equal(
            _target_encoding_residual_domain(None, base),
            np.ones(4, dtype=bool),
        )

    def test_default_smoothing_constant(self) -> None:
        """Default smoothing constant."""
        assert _TARGET_ENCODING_DEFAULT_SMOOTHING == 20.0

    def test_composite_name_is_two_segment(self) -> None:
        # requires_base=False -> base-free 2-segment composite name.
        """Composite name is two segment."""
        assert compose_target_name("y", "target_encoding_residual") == "y-tgtenc"
        assert is_composite_target_name("y-tgtenc")


# ---------------------------------------------------------------------------
# biz_value: target-encoding residual + linear beats raw-y linear
# ---------------------------------------------------------------------------


class TestBizValueTargetEncodingBeatsRawLinear:
    """DGP: y = category_effect(cat) + 2.0*x + noise, where ``cat`` is a 50-level
    high-cardinality categorical whose per-level effect is a large random offset.
    A plain linear model on raw y cannot encode the 50 category offsets from a
    single numeric ``x`` feature (the category column is non-numeric and dropped),
    so it pays the full category-effect variance as error. Removing the per-
    category level via the target-encoding residual lets the SAME linear model
    fit the clean ``2.0*x + noise`` residual -> a clear RMSE win.

    Measured win (seed 0): raw-y linear RMSE ~9.9, composite RMSE ~1.0 (~9.5x).
    Floor pinned conservatively at 3x below the measured ratio.
    """

    def _make_high_card_dgp(self, n: int = 4000, n_levels: int = 50, seed: int = 0):
        # The per-category effect is a FIXED property of the DGP shared across
        # train/test (drawn from a dedicated fixed seed), so a category's level
        # learned at fit transfers to the same category at predict. Only x / cat
        # assignment / noise vary with ``seed`` (the train vs test split).
        """Make high card dgp."""
        eff_rng = np.random.default_rng(20260611)
        level_effect = eff_rng.normal(loc=0.0, scale=10.0, size=n_levels)
        rng = np.random.default_rng(seed)
        cat_idx = rng.integers(0, n_levels, size=n)
        cats = np.asarray([f"c{i}" for i in cat_idx])
        x = rng.normal(size=n)
        y = level_effect[cat_idx] + 2.0 * x + rng.normal(scale=1.0, size=n)
        df = pd.DataFrame({"x": x, "cat": cats})
        return df, y

    def _rmse(self, a: np.ndarray, b: np.ndarray) -> float:
        """Rmse."""
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

    def test_composite_beats_raw_linear_by_clear_margin(self) -> None:
        """Composite beats raw linear by clear margin."""
        df_tr, y_tr = self._make_high_card_dgp(seed=0)
        df_te, y_te = self._make_high_card_dgp(seed=1)

        # Baseline: raw-y linear on the numeric feature only (cat is non-numeric).
        base_lr = LinearRegression()
        base_lr.fit(df_tr[["x"]].to_numpy(), y_tr)
        rmse_raw = self._rmse(base_lr.predict(df_te[["x"]].to_numpy()), y_te)

        # Composite: target-encoding residual removes the per-category level,
        # the inner linear model fits the clean 2*x + noise residual.
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="target_encoding_residual",
            group_column="cat",
        )
        est.fit(df_tr[["x", "cat"]], y_tr)
        rmse_comp = self._rmse(est.predict(df_te[["x", "cat"]]), y_te)

        assert rmse_comp < rmse_raw / 3.0, f"target-encoding residual must beat raw-y linear by >=3x RMSE: raw={rmse_raw:.3f}, composite={rmse_comp:.3f}"

    def test_unseen_category_predict_stays_finite(self) -> None:
        """Unseen category predict stays finite."""
        df_tr, y_tr = self._make_high_card_dgp(n=2000, n_levels=50, seed=2)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="target_encoding_residual",
            group_column="cat",
        )
        est.fit(df_tr[["x", "cat"]], y_tr)
        # Predict batch with categories NEVER seen at fit (c900..c904).
        rng = np.random.default_rng(99)
        df_unseen = pd.DataFrame({"x": rng.normal(size=5), "cat": [f"c{900 + i}" for i in range(5)]})
        preds = est.predict(df_unseen[["x", "cat"]])
        assert preds.shape == (5,)
        assert np.all(np.isfinite(preds)), "unseen-category predict must be finite"
