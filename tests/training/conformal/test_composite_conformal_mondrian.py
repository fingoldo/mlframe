"""Unit + biz_value tests for Mondrian (group-conditional) split-conformal and
tiny-n robustness of the conformal radius.

Headline guarantees:
- Mondrian gives a SEPARATE per-group radius so conditional coverage stays
  >= 1-alpha within each group, where the marginal band under-covers the
  higher-variance minority group.
- conformal_quantile + the calibrate/predict paths never crash (and never
  silently mis-cover) at n_cal in {0, 1, 2}: they return a +inf radius (valid
  but uninformative) below the finite-sample rank threshold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.conformal import conformal_quantile


# --------------------------------------------------------------------------- #
# Tiny-n robustness of the radius itself.
# --------------------------------------------------------------------------- #
class TestTinyNRadius:
    """Groups tests covering tiny n radius."""
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_tiny_n_returns_inf(self, n: int) -> None:
        # n in {0,1,2} at alpha=0.1: ceil((n+1)*0.9) > n -> uninformative inf.
        """Tiny n returns inf."""
        r = np.arange(float(n))
        assert conformal_quantile(r, 0.1) == float("inf")

    def test_single_point_at_huge_alpha_is_finite(self) -> None:
        # n=1, alpha=0.6 -> rank=ceil(2*0.4)=1 <= 1 -> the lone residual.
        """Single point at huge alpha is finite."""
        assert conformal_quantile(np.array([7.0]), 0.6) == pytest.approx(7.0)

    def test_does_not_crash_on_all_nonfinite(self) -> None:
        """Does not crash on all nonfinite."""
        r = np.array([np.nan, np.inf, -np.inf])
        assert conformal_quantile(r, 0.1) == float("inf")


def _fit(seed: int, n: int = 600):
    """Fit."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = b + 0.5 * f + rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({"b": b, "feat": f})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="b",
    ).fit(X, y)
    return est


class TestTinyNCalibratePredict:
    """Groups tests covering tiny n calibrate predict."""
    def test_calibrate_predict_single_cal_row_no_crash(self) -> None:
        """Calibrate predict single cal row no crash."""
        est = _fit(0)
        Xc = pd.DataFrame({"b": [0.3], "feat": [0.1]})
        est.calibrate_conformal(Xc, np.array([0.5]), 0.1)
        # Single cal row -> inf radius (uninformative-but-valid), no crash.
        assert est._conformal_q_[round(0.1, 6)] == float("inf")
        lo, hi = est.predict_interval(Xc, 0.1)
        # Band is well-formed (may be clipped to the train envelope), 1 element.
        assert lo.shape == (1,) and hi.shape == (1,)
        assert hi[0] >= lo[0]

    def test_predict_interval_single_test_row(self) -> None:
        """Predict interval single test row."""
        est = _fit(1)
        rng = np.random.default_rng(1)
        Xc = pd.DataFrame({"b": rng.normal(size=300), "feat": rng.normal(size=300)})
        yc = Xc["b"].to_numpy() + rng.normal(size=300)
        est.calibrate_conformal(Xc, yc, 0.1)
        lo, hi = est.predict_interval(Xc.iloc[:1], 0.1)
        assert lo.shape == (1,) and hi.shape == (1,)
        assert hi[0] >= lo[0]


# --------------------------------------------------------------------------- #
# Mondrian group-conditional conformal.
# --------------------------------------------------------------------------- #
def _make_grouped(seed: int, n_major: int = 4000, n_minor: int = 400):
    """2-group heteroscedastic-by-group synthetic: group 1 (minority) has a much
    larger residual scale than group 0 (majority)."""
    rng = np.random.default_rng(seed)
    n = n_major + n_minor
    b = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    groups = np.empty(n, dtype=object)
    groups[:n_major] = "maj"
    groups[n_major:] = "min"
    noise = np.empty(n)
    noise[:n_major] = rng.normal(0.0, 0.5, n_major)
    noise[n_major:] = rng.normal(0.0, 4.0, n_minor)  # minority is noisier.
    y = b + 0.5 * f + noise
    X = pd.DataFrame({"b": b, "feat": f})
    return X, y, groups


def _split3(X, y, groups, seed: int):
    """Split3."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n3 = len(y) // 3
    tr, ca, te = idx[:n3], idx[n3 : 2 * n3], idx[2 * n3 :]
    return tr, ca, te


class TestMondrianBizValue:
    """Groups tests covering mondrian biz value."""
    def test_biz_mondrian_minority_coverage_beats_marginal(self) -> None:
        """On the noisier minority group, the marginal band under-covers while
        Mondrian's per-group radius keeps coverage at >= 1-alpha. Mondrian must
        be CLOSER to the 1-alpha target on the minority group than the marginal
        band, across seeds."""
        alpha = 0.1
        target = 1.0 - alpha
        marg_errs, mond_errs = [], []
        for seed in range(3):
            X, y, groups = _make_grouped(seed)
            tr, ca, te = _split3(X, y, groups, seed)
            est = CompositeTargetEstimator(
                base_estimator=LinearRegression(),
                transform_name="linear_residual",
                base_column="b",
            ).fit(X.iloc[tr], y[tr])
            est.calibrate_conformal(X.iloc[ca], y[ca], alpha)
            est.calibrate_conformal_mondrian(X.iloc[ca], y[ca], groups[ca], alpha)

            te_min = te[groups[te] == "min"]
            yt = y[te_min]
            lo_m, hi_m = est.predict_interval(X.iloc[te_min], alpha)
            cov_marg = float(np.mean((yt >= lo_m) & (yt <= hi_m)))
            lo_o, hi_o = est.predict_interval_mondrian(
                X.iloc[te_min],
                groups[te_min],
                alpha,
            )
            cov_mond = float(np.mean((yt >= lo_o) & (yt <= hi_o)))
            marg_errs.append(abs(cov_marg - target))
            mond_errs.append(abs(cov_mond - target))
            # Mondrian must actually reach the target on the minority group.
            assert cov_mond >= 0.86, f"seed {seed}: mondrian under-covered {cov_mond:.3f}"

        mean_marg = float(np.mean(marg_errs))
        mean_mond = float(np.mean(mond_errs))
        assert mean_mond < mean_marg, f"mondrian not closer to target on minority: mond_err={mean_mond:.3f} marg_err={mean_marg:.3f}"

    def test_mondrian_unseen_group_falls_back_with_warning(self) -> None:
        """Mondrian unseen group falls back with warning."""
        X, y, groups = _make_grouped(0)
        tr, ca, te = _split3(X, y, groups, 0)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        ).fit(X.iloc[tr], y[tr])
        est.calibrate_conformal_mondrian(X.iloc[ca], y[ca], groups[ca], 0.1)
        # A label never seen at calibration must fall back + warn.
        unseen = np.array(["brand_new"] * 3, dtype=object)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            lo, hi = est.predict_interval_mondrian(X.iloc[te[:3]], unseen, 0.1)
        assert any("global radius" in str(w.message) for w in rec)
        assert lo.shape == (3,) and hi.shape == (3,)

    def test_mondrian_tiny_group_falls_back_to_global(self) -> None:
        """A group with too few calibration rows to certify the level uses the
        finite global radius, not an inf band."""
        rng = np.random.default_rng(0)
        n = 600
        b = rng.normal(size=n)
        f = rng.normal(size=n)
        y = b + 0.5 * f + rng.normal(size=n)
        X = pd.DataFrame({"b": b, "feat": f})
        groups = np.array(["big"] * n, dtype=object)
        groups[:2] = "tiny"  # 2 rows -> per-group rank exceeds n_g at alpha=0.1.
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        ).fit(X, y)
        est.calibrate_conformal_mondrian(X, y, groups, 0.1)
        table = est._mondrian_q_[round(0.1, 6)]
        assert np.isfinite(table["tiny"]), "tiny group should fall back to finite global"
        assert table["tiny"] == pytest.approx(table[None])

    def test_predict_mondrian_without_calibration_raises(self) -> None:
        """Predict mondrian without calibration raises."""
        est = _fit(0)
        with pytest.raises(RuntimeError, match="no Mondrian radius"):
            est.predict_interval_mondrian(
                pd.DataFrame({"b": [0.0], "feat": [0.0]}),
                np.array(["a"]),
                0.1,
            )

    def test_mondrian_single_test_row(self) -> None:
        """Mondrian single test row."""
        X, y, groups = _make_grouped(0)
        tr, ca, te = _split3(X, y, groups, 0)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        ).fit(X.iloc[tr], y[tr])
        est.calibrate_conformal_mondrian(X.iloc[ca], y[ca], groups[ca], 0.1)
        i = te[:1]
        lo, hi = est.predict_interval_mondrian(X.iloc[i], groups[i], 0.1)
        assert lo.shape == (1,) and hi.shape == (1,)
        assert hi[0] >= lo[0]

    def test_mondrian_calibrate_before_fit_raises(self) -> None:
        """Mondrian calibrate before fit raises."""
        from sklearn.exceptions import NotFittedError

        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        )
        X = pd.DataFrame({"b": [1.0, 2.0], "feat": [3.0, 4.0]})
        with pytest.raises(NotFittedError):
            est.calibrate_conformal_mondrian(
                X,
                np.array([1.0, 2.0]),
                np.array(["a", "b"]),
                0.1,
            )


class TestMondrianRadiusGatherVectorization:
    """Pin the vectorized (pd.factorize) radius gather in predict_interval_mondrian.

    The factorize fast-path replaces a per-row Python loop (2.3x @10M). It must
    stay bit-identical to a row-by-row dict lookup, INCLUDING the two corners
    the loop handled implicitly: an unseen group falls back to the global radius
    (and is reported as missing), and a NaN group label is NOT a known group so
    it ALSO falls back to global. The latter only holds with
    ``pd.factorize(..., use_na_sentinel=False)``; default factorize drops NaN to
    code -1 and would gather the LAST unique radius instead -- this test fails on
    that regression.
    """

    @staticmethod
    def _loop_reference(g, per_group, global_r):
        """Loop reference."""
        radii = np.empty(g.shape[0], dtype=np.float64)
        missing = set()
        for i in range(g.shape[0]):
            lab = g[i]
            if lab in per_group:
                radii[i] = per_group[lab]
            else:
                radii[i] = global_r
                missing.add(lab)
        return radii

    def _build_est(self):
        """Build est."""
        est = _fit(0)
        per_group = {"a": 1.5, "b": 2.5, "c": 0.7, None: 9.0}
        est._mondrian_q_ = {round(0.1, 6): per_group}
        return est, per_group

    def test_radius_gather_matches_row_loop_with_unseen_and_nan_labels(self) -> None:
        """Radius gather matches row loop with unseen and nan labels."""
        est, per_group = self._build_est()
        global_r = per_group[None]
        # mix known, unseen ("z"), and a NaN label -- all must match the loop.
        g = np.array(["a", "z", "b", float("nan"), "a", "c", "z"], dtype=object)
        X = pd.DataFrame({"b": np.zeros(g.shape[0]), "feat": np.zeros(g.shape[0])})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = est.predict_interval_mondrian(X, g, 0.1)
        np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
        expected_radii = self._loop_reference(g, per_group, global_r)
        # Recover the radius the function applied (band is symmetric pre-clip; the
        # train envelope here is unbounded for this synthetic, so no clipping).
        applied = (hi - lo) / 2.0
        assert np.array_equal(applied, expected_radii), (
            "vectorized radius gather diverged from the row-by-row loop; NaN/unseen labels must fall back to the global radius"
        )

    def test_nan_label_falls_back_to_global_not_last_unique(self) -> None:
        # Direct guard against use_na_sentinel=True: with default factorize a NaN
        # label would gather radius_per_uniq[-1] (the last seen group's radius),
        # NOT the global fallback. Construct a case where those differ.
        """Nan label falls back to global not last unique."""
        est, per_group = self._build_est()
        g = np.array(["a", float("nan")], dtype=object)
        X = pd.DataFrame({"b": [0.0, 0.0], "feat": [0.0, 0.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lo, hi = est.predict_interval_mondrian(X, g, 0.1)
        applied = (hi - lo) / 2.0
        assert applied[0] == pytest.approx(per_group["a"])
        assert applied[1] == pytest.approx(per_group[None]), "NaN label must use the global radius, not the last unique group's"
