"""Regression + biz_value tests for the fitted-params-aware domain hook
(audit 2026-06-10, item T15).

Pre-fix, the ``Transform`` contract exposed only ``domain_check(y, base)``,
evaluated BEFORE ``fit``. Transforms whose true per-row domain depends on a
learned parameter could therefore never enforce it:

- ``log_y`` (T = log(y + offset)): a row with ``y + offset <= 0`` produces
  NaN under ``log`` -- but ``offset`` is unknown until ``fit`` runs, so the
  pre-fit ``domain_check`` (just ``isfinite(y)``) lets it through. When the
  fitted params come from a sample that does not cover the forwarded rows
  (screening params fit on a subsample, then the rerank forwards the FULL
  train), the NaN T silently poisoned the whole spec's CV score
  (``_tiny_cv_rmse_y_scale``'s non-finite guard returns NaN -> spec dropped).
- ``centered_ratio`` (T = y / (base + c)): a predict-time row whose
  ``base + c`` lands in the near-zero ``[-eps, eps]`` band has its denominator
  silently clamped to the eps-floor -- the inverse no longer recovers y -- yet
  the params-free ``domain_check(None, base)`` passed it as valid, so it was
  NOT routed to the ``y_train_median`` fallback.

The fix adds an OPTIONAL ``domain_check_fitted(y, base, params)`` hook to the
``Transform`` contract, wires it for ``log_y`` / ``centered_ratio``, and
applies it after ``fit`` in screening (``_eval.py``), the rerank
(``_screening_tiny.py``), the wrapper ``fit`` (``_estimator.py``), and the
predict-side base-domain gate (``_predict.py``).

Each test below FAILS on the pre-fix logic:
- ``test_*_hook_present`` / ``test_*_catches_*`` fail because the hook /
  attribute did not exist (and the registry entries lacked it).
- ``test_rerank_*`` fails because pre-fix the NaN T poisoned the whole spec
  (RMSE returned NaN).
- ``test_biz_*`` fails because the eps-band predict row got a distorted value
  rather than the fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY
from mlframe.training.composite.discovery._screening_tiny import _tiny_cv_rmse_y_scale
from mlframe.training.composite.estimator import CompositeTargetEstimator


# --------------------------------------------------------------------------
# 1. Contract: the hook exists and is wired exactly where params-dependent.
# --------------------------------------------------------------------------

PARAMS_DEPENDENT = ["log_y", "centered_ratio"]
PARAMS_FREE_SAMPLE = ["diff", "ratio", "linear_residual", "cbrt_y", "additive_residual"]


@pytest.mark.parametrize("name", PARAMS_DEPENDENT)
def test_params_dependent_transform_has_fitted_domain_hook(name):
    """log_y / centered_ratio expose a non-None ``domain_check_fitted`` so the
    fitted-params domain can be enforced (pre-fix: attribute absent / None)."""
    t = TRANSFORMS_REGISTRY[name]
    assert getattr(t, "domain_check_fitted", None) is not None, (
        f"{name!r} validity depends on a fitted param and MUST declare domain_check_fitted so callers can enforce its true domain"
    )


@pytest.mark.parametrize("name", PARAMS_FREE_SAMPLE)
def test_params_free_transforms_leave_hook_unset(name):
    """Transforms whose params-free domain is exact must NOT set the hook --
    otherwise callers pay an extra fit / mask pass for nothing and the path
    is no longer bit-identical for them."""
    t = TRANSFORMS_REGISTRY[name]
    assert getattr(t, "domain_check_fitted", "MISSING") is None, (
        f"{name!r} has no params-dependent domain; domain_check_fitted must stay None to keep its path bit-identical"
    )


# --------------------------------------------------------------------------
# 2. Hook correctness: it catches rows the params-free check misses.
# --------------------------------------------------------------------------


def test_log_y_fitted_domain_catches_below_offset_rows():
    """log_y forward NaNs at ``y + offset <= 0``; the params-free domain
    (isfinite only) misses those rows -- the fitted hook must flag them."""
    t = TRANSFORMS_REGISTRY["log_y"]
    y_train = np.array([2.0, 5.0, 10.0, 100.0])
    params = t.fit(y_train, None)
    offset = float(params["offset"])
    # A NEW sample with a row below -offset -> forward produces NaN there.
    y_new = np.array([2.0, -offset - 4.0, 10.0])
    dom_free = np.asarray(t.domain_check(y_new, None), dtype=bool)
    dom_fit = np.asarray(t.domain_check_fitted(y_new, None, params), dtype=bool)
    fwd = t.forward(y_new, None, params)
    nan_rows = ~np.isfinite(fwd)
    # Pre-fix evidence: the params-free check accepted the NaN-producing row.
    assert dom_free[1], "fixture stale: params-free check already rejected row"
    assert nan_rows[1], "fixture stale: forward did not NaN on the below-offset row"
    # Post-fix: the fitted hook rejects exactly the NaN-producing row(s).
    assert not dom_fit[1], "fitted domain failed to flag the below-offset row"
    assert dom_fit[0] and dom_fit[2], "fitted domain over-rejected in-domain rows"
    # The fitted hook must reject AT LEAST every row forward would NaN on.
    assert not np.any(dom_fit & nan_rows), "a row flagged valid by the fitted domain still produced NaN T"


def test_centered_ratio_fitted_domain_catches_eps_band_rows_predict():
    """centered_ratio's denominator (base + c) is clamped to the eps-floor in
    the near-zero band, so T no longer reflects the ratio. The params-free
    predict-side check (y=None, isfinite(base)) misses the band; the fitted
    hook must flag it (it knows c + eps)."""
    t = TRANSFORMS_REGISTRY["centered_ratio"]
    base_train = np.array([1.0, 2.0, 5.0, 10.0])
    y_train = np.array([1.0, 2.0, 3.0, 4.0])
    params = t.fit(y_train, base_train)
    c = float(params["c"])
    # Predict batch (y unknown): a row with base == -c => base + c == 0 (in band).
    base_new = np.array([1.0, -c, 5.0])
    dom_free = np.asarray(t.domain_check(None, base_new), dtype=bool)
    dom_fit = np.asarray(t.domain_check_fitted(None, base_new, params), dtype=bool)
    assert dom_free[1], "fixture stale: params-free check already rejected band row"
    assert not dom_fit[1], "fitted domain failed to flag the eps-band predict row"
    assert dom_fit[0] and dom_fit[2], "fitted domain over-rejected in-band rows"


# --------------------------------------------------------------------------
# 3. Integration: the rerank no longer NaN-poisons a spec whose fitted params
#    were learned on a subsample that does not cover the full forward sample.
# --------------------------------------------------------------------------


def test_rerank_finite_when_params_from_subsample_not_covering_train():
    """``_tiny_cv_rmse_y_scale`` receives the FULL train y but params fit on a
    screening subsample. For log_y the subsample-fit offset can be too small
    to keep the full-train tail in domain -> forward NaNs those rows. Pre-fix
    the non-finite guard nuked the WHOLE spec's score (NaN); post-fix the
    fitted-domain refinement drops only the out-of-domain rows -> finite RMSE."""
    rng = np.random.default_rng(0)
    t = TRANSFORMS_REGISTRY["log_y"]
    n = 400
    y_full = rng.normal(50.0, 10.0, n)
    y_full[:5] = np.array([-30.0, -25.0, -28.0, -22.0, -26.0])
    base = rng.normal(0.0, 1.0, n)
    X = (rng.normal(0.0, 1.0, (n, 3)) + 0.3 * y_full[:, None]).astype(np.float64)
    # Params fit on positives only (a screen sample that excludes the tail).
    params = t.fit(y_full[y_full > 0.0], None)
    offset = float(params["offset"])

    # Pre-fix reproduction: forward over the params-free-valid rows (all finite)
    # NaNs the tail rows -> the whole-spec non-finite guard would return NaN.
    valid_free = np.asarray(t.domain_check(y_full, None), dtype=bool)
    t_free = t.forward(y_full[valid_free], None, params)
    assert np.any(~np.isfinite(t_free)), "fixture stale: subsample offset already covers full train; no NaN T to poison the spec"
    assert int((y_full <= -offset).sum()) >= 1

    # Post-fix: the real function drops the out-of-domain rows -> finite RMSE.
    rmse = _tiny_cv_rmse_y_scale(
        y_full,
        base,
        t,
        params,
        X,
        family="linear",
        n_estimators=20,
        num_leaves=7,
        learning_rate=0.1,
        cv_folds=3,
        random_state=0,
    )
    assert np.isfinite(rmse), "rerank returned NaN: out-of-fitted-domain rows poisoned the whole spec instead of being dropped"
    assert rmse >= 0.0


# --------------------------------------------------------------------------
# 4. biz_value: predict-side centered_ratio routes the eps-band row to the
#    fallback instead of emitting a distorted clamped-divisor value.
# --------------------------------------------------------------------------


def test_biz_centered_ratio_predict_eps_band_routes_to_fallback():
    """A predict row whose ``base + c`` lands in the eps-floor band must fall
    back to ``y_train_median`` (the inverse cannot recover y once the divisor
    is clamped). Pre-fix the params-free domain passed the row and it got a
    distorted value; post-fix the fitted-domain gate routes it to the
    fallback. biz_value: the band row's prediction must equal the train
    median (the honest fallback), NOT a wild clamped-divisor extrapolation."""
    rng = np.random.default_rng(2)
    n = 300
    base = rng.normal(5.0, 1.0, n)
    y = 2.0 * base + rng.normal(0.0, 0.5, n)
    X = pd.DataFrame({"b": base, "f0": rng.normal(0.0, 1.0, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="centered_ratio",
        base_column="b",
        fallback_predict="y_train_median",
    ).fit(X, y)
    c = float(est.fitted_params_["c"])
    y_train_median = float(est.fitted_params_["y_train_median"])

    # Predict batch: idx1 is exactly in the eps-band (base + c == 0).
    Xp = pd.DataFrame({"b": np.array([5.0, -c, 6.0]), "f0": np.zeros(3)})
    yp = np.asarray(est.predict(Xp), dtype=np.float64)
    assert np.all(np.isfinite(yp)), "fallback must keep all predictions finite"
    # The in-domain rows track the linear fit (~2*base); the band row is the
    # fallback (train median), well separated from a true ratio prediction.
    assert yp[1] == pytest.approx(y_train_median, rel=1e-6), (
        f"eps-band predict row should equal y_train_median fallback ({y_train_median:.4f}); got {yp[1]:.4f} (distorted clamped divisor)"
    )
    # Sanity: the in-domain rows are model-driven (vary with base), NOT all
    # collapsed to the fallback. With different base values their predictions
    # must differ from each other and at least one must be away from the median.
    assert yp[0] != pytest.approx(yp[2]), "in-domain rows with different base collapsed to the same value"
    assert abs(yp[0] - y_train_median) > 1.0, "in-domain row unexpectedly equals the fallback median"

    # Direct pre-fix contrast: without the fitted-domain gate the band row
    # would have been forwarded through the clamped divisor + inverse, giving a
    # value that is NOT the honest fallback. Compute that un-gated value and
    # confirm the gate changed the outcome.
    transform = est.fitted_params_
    t = TRANSFORMS_REGISTRY["centered_ratio"]
    t_hat_band = float(est.estimator_.predict(Xp.iloc[[1]])[0])
    ungated = float(
        t.inverse(
            np.array([t_hat_band]),
            np.array([-c]),
            est.fitted_params_,
        )[0]
    )
    assert yp[1] != pytest.approx(ungated, rel=1e-3) or ungated == pytest.approx(y_train_median, rel=1e-6), (
        "fitted-domain gate did not change the band-row prediction vs the pre-fix clamped-divisor inverse"
    )
