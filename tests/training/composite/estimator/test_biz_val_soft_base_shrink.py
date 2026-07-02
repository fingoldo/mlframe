"""Unit + biz_value + profiling coverage for T2-L soft base-shrink inverse + smart fallback.

The feature lives in ``mlframe.training.composite.estimator._soft_shrink`` and is wired into
``CompositeTargetEstimator._predict_unclipped``. It guards the base-additive inverse
(``y = T_hat + alpha*base + beta`` and siblings) against out-of-fit-range base values on unseen-group
tails: instead of the additive inverse extrapolating and collapsing, the base is soft-clipped toward the
calibration boundary (smooth, monotone, C1-continuous, no hard clamp), and deeply-OOD rows route to the
causal-lag failsafe (or the wrapper's ``fallback_predict``). In-range predictions are byte-identical to
``soft_base_shrink=False``.

cProfile (predict, n=200_000, 30% out-of-range base, linear_residual; after the njit optimization; see
``_profile_predict`` below to reproduce):

  tottime  function
   0.0025  transforms/linear.py:_linear_residual_inverse       (intrinsic transform inverse; not ours)
   0.0028  numpy.ufunc.reduce (max/any/sum across t-clip / domain / shrink; pipeline-wide, not isolable)
   0.0017  _soft_shrink.shrink_base   (the fused njit soft-clip kernel)

Optimization applied: the original vectorised ``shrink_base`` (broadcast + fancy-index over (n,K)
temporaries) was 18.6 ms/call and ~50% of predict tottime. Replacing it with a fused single-pass
``numba.njit(parallel=True)`` kernel (one loop computes base_eff + the per-row OOR distance, no
temporaries) cut it to ~1.7 ms/call (~11x) and dropped the whole soft-shrink overhead from 23.5 ms to
4.5 ms/200k rows -- bit-identical to the retained numpy oracle (``_shrink_base_numpy``). The residual
cost is unavoidable copies + the inner predict; no further actionable speedup. Soft-shrink runs only when
enabled AND a base-additive transform AND a fit-range was captured, so every other transform pays nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.estimator import _soft_shrink as ss


@pytest.fixture(scope="module", autouse=True)
def _warm_numba_kernel():
    """Compile the njit soft-clip kernel once so per-test wall time excludes the one-off JIT compile."""
    ss.shrink_base(
        np.array([-5.0, 0.0, 5.0, 50.0]),
        np.array([0.0]), np.array([10.0]), np.array([4.0]),
    )
    yield


class _SubsetLinear(LinearRegression):
    """LinearRegression that trains on the columns seen at fit and ignores any extra predict-time columns.

    Mirrors a real feature-selecting inner: the causal-lag column can be present in the predict frame (for
    the smart fallback) without leaking into the residual model, and without a feature-count mismatch.
    """

    def fit(self, X, y, **kw):
        self._cols = list(X.columns)
        return super().fit(X[self._cols].to_numpy(dtype=float), np.asarray(y, dtype=float), **kw)

    def predict(self, X):
        cols = getattr(self, "_cols", None)
        arr = X[cols].to_numpy(dtype=float) if cols is not None else np.asarray(X, dtype=float)
        return super().predict(arr)


def _fit_linear_residual(seed=0, n=800, slope=3.0, base_lo=0.0, base_hi=10.0, noise=0.3, cols=("base", "f")):
    rng = np.random.default_rng(seed)
    base = rng.uniform(base_lo, base_hi, n)
    y = slope * base + rng.normal(0.0, noise, n)
    data = {"base": base, "f": rng.normal(0.0, 1.0, n)}
    X = pd.DataFrame({c: data[c] for c in cols})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="linear_residual", base_column="base",
    )
    est.fit(X, y)
    return est, X, y


# --------------------------------------------------------------------------------------------------
# Unit: in-range byte-identity (the core contract)
# --------------------------------------------------------------------------------------------------

def test_in_range_predict_is_byte_identical_to_disabled():
    est, _X, _y = _fit_linear_residual()
    rng = np.random.default_rng(11)
    Xin = pd.DataFrame({"base": rng.uniform(1.0, 9.0, 300), "f": rng.normal(0, 1, 300)})
    est.soft_base_shrink = True
    p_on = est.predict(Xin)
    est.soft_base_shrink = False
    p_off = est.predict(Xin)
    assert np.array_equal(p_on, p_off), "in-range predictions must be bit-identical with soft-shrink on/off"
    # The mechanism IS active: an out-of-range base changes the prediction (so the identity above is a real
    # invariant on in-range rows, not a dead no-op that would pass even if we wrongly shrank in-range).
    Xoor = pd.DataFrame({"base": [1.0, 100.0], "f": [0.0, 0.0]})
    est.soft_base_shrink = True
    on = est.predict(Xoor)
    est.soft_base_shrink = False
    off = est.predict(Xoor)
    assert on[0] == off[0], "the in-range row of a mixed batch stays identical"
    assert on[1] != off[1], "the out-of-range row must differ (soft-shrink active)"


def test_disabled_flag_restores_raw_inverse_exactly():
    est, _X, _y = _fit_linear_residual()
    Xoor = pd.DataFrame({"base": [50.0, 200.0], "f": [0.0, 0.0]})
    est.soft_base_shrink = False
    p = est.predict(Xoor)
    # Raw additive inverse y = T_hat + alpha*base + beta with the raw (un-shrunk) base.
    a = est.fitted_params_["alpha"]
    b = est.fitted_params_["beta"]
    t_hat = est.estimator_.predict(Xoor)
    assert np.allclose(p, np.clip(t_hat + a * Xoor["base"].to_numpy() + b,
                                  est.fitted_params_["y_clip_low"], est.fitted_params_["y_clip_high"]))
    assert est.soft_shrink_info_["n_shrunk"] == 0


# --------------------------------------------------------------------------------------------------
# Unit: the soft-clip is smooth, monotone, saturating, and NOT a hard clamp
# --------------------------------------------------------------------------------------------------

def test_shrink_base_monotone_continuous_saturating_not_clamped():
    lo, hi, iqr = np.array([0.0]), np.array([10.0]), np.array([4.0])
    xs = np.linspace(5.0, 400.0, 600)  # spans in-range through deep out-of-range
    base_eff, d_row = ss.shrink_base(xs, lo, hi, iqr)

    # Monotone increasing everywhere (no ordering inversion across the boundary).
    assert np.all(np.diff(base_eff) > 0.0), "base_eff must be strictly monotone increasing in base"

    # Continuity / no jump at the boundary: slope from just-outside -> ~1 (matches the in-range slope of 1).
    eps = 1e-6
    be_in = ss.shrink_base(np.array([hi[0] - eps]), lo, hi, iqr)[0][0]
    be_out = ss.shrink_base(np.array([hi[0] + eps]), lo, hi, iqr)[0][0]
    assert abs(be_out - be_in) < 3.0 * eps, "no jump at the calibration boundary"
    assert abs((be_out - hi[0]) / eps - 1.0) < 1e-3, "slope ~1 just outside the boundary (C1-continuous)"

    # NOT a hard clamp: far out-of-range base_eff keeps growing (strictly > hi) yet saturates below hi+iqr.
    far = ss.shrink_base(np.array([1e6]), lo, hi, iqr)[0][0]
    assert far > hi[0], "soft-clip is not a hard clamp (value keeps moving past the boundary)"
    assert far < hi[0] + iqr[0], "the out-of-range contribution saturates within one IQR"

    # Further out => more shrink (the withheld amount base - base_eff strictly increases beyond the boundary).
    out = xs[xs > hi[0]]
    be_out_sweep = ss.shrink_base(out, lo, hi, iqr)[0]
    shrink_amount = out - be_out_sweep
    assert np.all(np.diff(shrink_amount) > 0.0), "shrink magnitude must grow monotonically with distance"
    # d_row grows with distance too (IQR units of OOR distance).
    assert np.all(np.diff(d_row[xs > hi[0]]) > 0.0)


def test_shrink_base_below_range_symmetric():
    lo, hi, iqr = np.array([0.0]), np.array([10.0]), np.array([4.0])
    below = ss.shrink_base(np.array([-40.0]), lo, hi, iqr)[0][0]
    assert below < lo[0], "below-range value keeps moving past the lower boundary"
    assert below > lo[0] - iqr[0], "below-range contribution saturates within one IQR"


@pytest.mark.parametrize("K", [1, 3])
def test_njit_kernel_matches_numpy_oracle(K):
    pytest.importorskip("numba")
    rng = np.random.default_rng(5)
    b = rng.uniform(-6.0, 6.0, (4000, K))
    b[rng.integers(0, 4000, 40), rng.integers(0, K, 40)] = np.nan  # domain-violation rows
    lo = np.full(K, -2.0)
    hi = np.full(K, 2.0)
    iqr = np.full(K, 1.5)
    inp = b if K > 1 else b[:, 0]
    be_j, d_j = ss.shrink_base(inp, lo, hi, iqr)
    be_n, d_n = ss._shrink_base_numpy(b, lo, hi, iqr)
    be_n = be_n if K > 1 else be_n.reshape(-1)
    assert np.array_equal(np.nan_to_num(be_j, nan=-999.0), np.nan_to_num(be_n, nan=-999.0))
    assert np.array_equal(d_j, d_n)


# --------------------------------------------------------------------------------------------------
# Unit: smart fallback routing for deeply-OOD rows
# --------------------------------------------------------------------------------------------------

def test_deep_ood_routes_to_causal_lag_when_present():
    # _SubsetLinear inner ignores the extra 'y_prev' column present only in the predict frame.
    rng = np.random.default_rng(12)
    n = 800
    base = rng.uniform(0, 10, n)
    y = 3.0 * base + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"base": base, "f": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=_SubsetLinear(), transform_name="linear_residual", base_column="base",
    )
    est.fit(X, y)
    est.target_name_ = "y"  # lets detect_causal_lag_column find the 'y_prev' failsafe column
    est.soft_base_shrink_severity_iqr = 3.0
    lag = np.array([30.0, 31.0])
    Xo = pd.DataFrame({"base": [80.0, 120.0], "f": [0.0, 0.0], "y_prev": lag})
    p = est.predict(Xo)
    info = est.soft_shrink_info_
    assert info["fallback_mask"].tolist() == [True, True]
    assert np.allclose(p, lag), "deeply-OOD rows must take the causal-lag value"


def test_deep_ood_routes_to_median_when_no_lag():
    est, _X, _y = _fit_linear_residual()
    est.soft_base_shrink_severity_iqr = 3.0
    med = est.fitted_params_["y_train_median"]
    Xo = pd.DataFrame({"base": [90.0, 150.0], "f": [0.0, 0.0]})
    p = est.predict(Xo)
    assert est.soft_shrink_info_["fallback_mask"].tolist() == [True, True]
    assert np.allclose(p, med), "no lag column -> deep-OOD rows fall back to y_train_median"


def test_deep_ood_routes_to_nan_when_no_lag_and_nan_fallback():
    rng = np.random.default_rng(2)
    n = 600
    base = rng.uniform(0, 10, n)
    y = 3.0 * base + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"base": base, "f": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="linear_residual",
        base_column="base", fallback_predict="nan",
    )
    est.fit(X, y)
    est.soft_base_shrink_severity_iqr = 3.0
    Xo = pd.DataFrame({"base": [90.0, 3.0], "f": [0.0, 0.0]})  # row0 deep-OOD, row1 in-range
    p = est.predict(Xo)
    assert np.isnan(p[0]), "deep-OOD row under 'nan' fallback returns NaN"
    assert np.isfinite(p[1]), "in-range row is unaffected"


# --------------------------------------------------------------------------------------------------
# Unit: the per-row flag is exact across in-range / moderate-OOR / deep-OOR
# --------------------------------------------------------------------------------------------------

def test_per_row_flag_is_exact():
    est, _X, _y = _fit_linear_residual()
    rng_ = est.fitted_params_[ss.BASE_FIT_RANGE_KEY]
    hi = float(rng_["hi"][0])
    iqr = float(rng_["iqr"][0])
    est.soft_base_shrink_severity_iqr = 3.0
    # Rows: in-range, moderately OOR (0.5 IQR beyond hi -> shrunk, not deep), deep OOR (5 IQR beyond hi).
    bases = np.array([hi * 0.5, hi + 0.5 * iqr, hi + 5.0 * iqr])
    Xo = pd.DataFrame({"base": bases, "f": np.zeros(3)})
    est.predict(Xo)
    info = est.soft_shrink_info_
    assert info["shrunk_mask"].tolist() == [False, True, True]
    assert info["fallback_mask"].tolist() == [False, False, True]
    assert info["n_shrunk"] == 2
    assert info["n_fallback"] == 1
    assert info["n_rows"] == 3


def test_no_ood_rows_flag_zero():
    est, _X, _y = _fit_linear_residual()
    Xin = pd.DataFrame({"base": np.linspace(1.0, 9.0, 50), "f": np.zeros(50)})
    est.predict(Xin)
    assert est.soft_shrink_info_["n_shrunk"] == 0
    assert est.soft_shrink_info_["n_fallback"] == 0


def test_all_rows_deep_ood():
    est, _X, _y = _fit_linear_residual()
    est.soft_base_shrink_severity_iqr = 3.0
    Xo = pd.DataFrame({"base": np.full(40, 500.0), "f": np.zeros(40)})
    est.predict(Xo)
    assert est.soft_shrink_info_["n_shrunk"] == 40
    assert est.soft_shrink_info_["n_fallback"] == 40


# --------------------------------------------------------------------------------------------------
# Unit: transform coverage (diff, multi-base) + non-additive / from_fitted_inner no-ops
# --------------------------------------------------------------------------------------------------

def test_diff_transform_captures_range_and_shrinks():
    rng = np.random.default_rng(4)
    n = 700
    base = rng.uniform(0, 20, n)
    y = base + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"base": base, "f": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="diff", base_column="base",
    )
    est.fit(X, y)
    assert ss.BASE_FIT_RANGE_KEY in est.fitted_params_
    Xo = pd.DataFrame({"base": [200.0], "f": [0.0]})
    on = est.predict(Xo)
    est.soft_base_shrink = False
    off = est.predict(Xo)
    assert on[0] != off[0]


def test_multibase_shrink_per_column():
    rng = np.random.default_rng(6)
    n = 900
    b1 = rng.uniform(0, 10, n)
    b2 = rng.uniform(-5, 5, n)
    y = 1.0 + 2.0 * b1 - 1.0 * b2 + rng.normal(0, 0.3, n)
    X = pd.DataFrame({"b1": b1, "b2": b2, "f": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="linear_residual_multi",
        base_columns=["b1", "b2"],
    )
    est.fit(X, y)
    rng_ = est.fitted_params_[ss.BASE_FIT_RANGE_KEY]
    assert rng_["lo"].size == 2
    # row0 in-range both cols; row1 col b2 far OOR -> shrunk.
    Xo = pd.DataFrame({"b1": [5.0, 5.0], "b2": [0.0, 60.0], "f": [0.0, 0.0]})
    est.predict(Xo)
    assert est.soft_shrink_info_["shrunk_mask"].tolist() == [False, True]


def test_non_additive_transform_captures_no_range():
    rng = np.random.default_rng(8)
    n = 600
    base = rng.uniform(1, 10, n)
    y = base * rng.uniform(0.5, 2.0, n)
    X = pd.DataFrame({"base": base, "f": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="ratio", base_column="base",
    )
    est.fit(X, y)
    assert ss.BASE_FIT_RANGE_KEY not in est.fitted_params_, "non-additive transforms stay byte-identical"


def test_from_fitted_inner_has_no_range_and_is_noop():
    est, X, y = _fit_linear_residual()
    wrapped = CompositeTargetEstimator.from_fitted_inner(
        est.estimator_, transform_name="linear_residual", base_column="base",
        transform_fitted_params={"alpha": est.fitted_params_["alpha"], "beta": est.fitted_params_["beta"]},
        y_train=y,
    )
    assert ss.BASE_FIT_RANGE_KEY not in wrapped.fitted_params_
    Xo = pd.DataFrame({"base": [80.0], "f": [0.0]})
    p = wrapped.predict(Xo)
    a = wrapped.fitted_params_["alpha"]
    b = wrapped.fitted_params_["beta"]
    t_hat = wrapped.estimator_.predict(Xo)
    raw = np.clip(t_hat + a * Xo["base"].to_numpy() + b,
                  wrapped.fitted_params_["y_clip_low"], wrapped.fitted_params_["y_clip_high"])
    assert np.allclose(p, raw), "no captured range -> raw inverse preserved"
    assert wrapped.soft_shrink_info_["n_shrunk"] == 0


# --------------------------------------------------------------------------------------------------
# biz_value: soft-shrink + smart fallback beats raw inverse AND plain median on unseen-group tails
# --------------------------------------------------------------------------------------------------

def _grouped_unseen_scenario():
    """Seen groups: base in [0,10], y = 3*base (learned by OLS). Unseen groups: base OOD in [40,60] while
    the TRUE y saturates near 30 -- so the linear additive inverse over-extrapolates (collapse) while the
    causal lag 'y_prev' (which tracks the saturated level) stays accurate."""
    rng = np.random.default_rng(7)
    ns = 1200
    bs = rng.uniform(0, 10, ns)
    ys = 3.0 * bs + rng.normal(0, 0.5, ns)
    Xtr = pd.DataFrame({"base": bs, "f": rng.normal(0, 1, ns)})  # inner trains WITHOUT the lag (no leakage)
    est = CompositeTargetEstimator(
        base_estimator=_SubsetLinear(), transform_name="linear_residual", base_column="base",
    )
    est.fit(Xtr, ys)
    est.target_name_ = "y"

    nu = 300
    ub = rng.uniform(40, 60, nu)
    uy = 30.0 + rng.normal(0, 1.0, nu)       # saturated true level
    uyp = uy + rng.normal(0, 1.0, nu)        # good causal lag of the saturated level
    Xun = pd.DataFrame({"base": ub, "f": rng.normal(0, 1, nu), "y_prev": uyp})
    return est, Xun, uy, bs, ys


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_biz_val_soft_base_shrink_beats_raw_and_median_on_unseen_groups():
    est, Xun, uy, _bs, _ys = _grouped_unseen_scenario()
    med = est.fitted_params_["y_train_median"]

    est.soft_base_shrink = True
    p_smart = est.predict(Xun)
    assert est.soft_shrink_info_["n_fallback"] == len(uy), "all unseen rows are deeply OOD -> lag fallback"

    est.soft_base_shrink = False
    p_raw = est.predict(Xun)
    p_med = np.full(len(uy), med)

    rmse_smart = _rmse(p_smart, uy)
    rmse_raw = _rmse(p_raw, uy)
    rmse_med = _rmse(p_med, uy)

    # Measured: smart ~1.0, median ~15.2, raw ~122 (floors set with generous margin).
    assert rmse_raw > 50.0, f"raw additive inverse should collapse on OOD base (got {rmse_raw:.2f})"
    assert rmse_med > 10.0, f"plain median fallback should be materially off (got {rmse_med:.2f})"
    assert rmse_smart < 4.0, f"smart soft-shrink+lag should be accurate (got {rmse_smart:.2f})"
    assert rmse_smart < 0.2 * rmse_raw, "smart must be materially lower than the raw inverse"
    assert rmse_smart < 0.4 * rmse_med, "smart must be materially lower than the plain median fallback"


def test_biz_val_soft_base_shrink_seen_group_rmse_unchanged():
    est, _Xun, _uy, bs, ys = _grouped_unseen_scenario()
    rng = np.random.default_rng(99)
    idx = rng.choice(len(bs), 300, replace=False)
    Xseen = pd.DataFrame({"base": bs[idx], "f": rng.normal(0, 1, 300),
                          "y_prev": ys[idx] + rng.normal(0, 0.5, 300)})
    est.soft_base_shrink = True
    p_on = est.predict(Xseen)
    est.soft_base_shrink = False
    p_off = est.predict(Xseen)
    assert np.array_equal(p_on, p_off), "seen (in-range) group predictions must be unchanged by the feature"
    assert _rmse(p_on, ys[idx]) == _rmse(p_off, ys[idx])


# --------------------------------------------------------------------------------------------------
# cProfile harness (run directly: python -m ... test_biz_val_soft_base_shrink); documented in the docstring.
# --------------------------------------------------------------------------------------------------

def _profile_predict(n: int = 200_000, oor_frac: float = 0.3):  # pragma: no cover - manual profiling entry
    import cProfile
    import pstats

    rng = np.random.default_rng(3)
    base = rng.uniform(0, 10, n)
    y = 3.0 * base + rng.normal(0, 0.5, n)
    X = pd.DataFrame({"base": base, "f": rng.normal(0, 1, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(), transform_name="linear_residual", base_column="base",
    )
    est.fit(X, y)
    n_out = int(n * oor_frac)
    bp = np.concatenate([rng.uniform(0, 10, n - n_out), rng.uniform(20, 80, n_out)])
    rng.shuffle(bp)
    Xp = pd.DataFrame({"base": bp, "f": rng.normal(0, 1, n)})
    est.predict(Xp)  # warm the njit kernel
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(10):
        est.predict(Xp)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(12)


if __name__ == "__main__":  # pragma: no cover
    _profile_predict()
