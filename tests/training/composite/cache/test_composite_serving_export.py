"""Unit + biz_value tests for the lightweight serving export
(``mlframe.training.composite.serving``).

Covers: bit-identity of the rebuilt numpy-only predict vs estimator.predict for
each covered transform, json.dumps/loads round-trip, NotImplementedError on an
unsupported transform, and explicit clip + domain-fallback behaviour. The
biz_value test asserts NO sklearn on the serve path AND atol-0 bit-identity.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.serving import (
    LIGHTWEIGHT_TRANSFORMS,
    SERVING_SPEC_VERSION,
    export_serving_spec,
    load_serving_spec,
)


def _make_xy(n=400, seed=0, base_positive=False):
    rng = np.random.default_rng(seed)
    base = rng.normal(5.0, 2.0, size=n)
    if base_positive:
        base = np.abs(base) + 0.5
    feat = rng.normal(0.0, 1.0, size=n)
    y = 2.0 * base + 0.5 * feat + rng.normal(0.0, 0.3, size=n)
    if base_positive:
        y = np.abs(y) + 1.0
    import pandas as pd

    X = pd.DataFrame({"base": base, "feat": feat})
    return X, y


def _fit(transform_name, **kw):
    base_positive = transform_name in ("ratio", "logratio")
    X, y = _make_xy(base_positive=base_positive, **kw)
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name=transform_name,
        base_column="base",
    )
    est.fit(X, y)
    return est, X, y


def _serve_predict(est, X):
    """Rebuild via spec and produce the y-scale prediction from the inner raw T."""
    spec = export_serving_spec(est)
    fn = load_serving_spec(spec)
    raw = est.estimator_.predict(X)
    base = X["base"].to_numpy()
    return fn(base, raw)


# ---------------------------------------------------------------------------
# Bit-identity for every covered single-base transform.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "transform_name",
    ["diff", "additive_residual", "linear_residual", "ratio", "logratio"],
)
def test_serving_bit_identical(transform_name):
    est, X, _ = _fit(transform_name)
    y_full = est.predict(X)
    y_serve = _serve_predict(est, X)
    np.testing.assert_array_equal(y_serve, y_full)


def test_serving_multi_base_bit_identical():
    import pandas as pd

    rng = np.random.default_rng(3)
    n = 400
    b1 = rng.normal(5.0, 2.0, n)
    b2 = rng.normal(-3.0, 1.5, n)
    feat = rng.normal(0, 1, n)
    y = 1.5 * b1 - 0.7 * b2 + 0.3 * feat + rng.normal(0, 0.2, n)
    X = pd.DataFrame({"b1": b1, "b2": b2, "feat": feat})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual_multi",
        base_columns=["b1", "b2"],
    )
    est.fit(X, y)
    spec = export_serving_spec(est)
    fn = load_serving_spec(spec)
    raw = est.estimator_.predict(X)
    base = X[["b1", "b2"]].to_numpy()
    np.testing.assert_array_equal(fn(base, raw), est.predict(X))


# ---------------------------------------------------------------------------
# json round-trip.
# ---------------------------------------------------------------------------
def test_serving_spec_json_round_trip():
    est, X, _ = _fit("linear_residual")
    spec = export_serving_spec(est)
    reloaded = json.loads(json.dumps(spec))
    assert reloaded["spec_version"] == SERVING_SPEC_VERSION
    assert reloaded["transform_name"] == "linear_residual"
    assert reloaded["base_columns"] == ["base"]
    fn = load_serving_spec(reloaded)
    raw = est.estimator_.predict(X)
    np.testing.assert_array_equal(fn(X["base"].to_numpy(), raw), est.predict(X))


def test_serving_spec_json_handles_non_finite_clip():
    """A spec whose clip bounds are +/-inf must survive strict json round-trip."""
    est, X, _ = _fit("diff")
    spec = export_serving_spec(est)
    # Force a non-finite envelope to exercise the inf sentinel mapping.
    spec["fitted_params"]["t_clip_low"] = "__-inf__"
    spec["fitted_params"]["t_clip_high"] = "__inf__"
    s = json.dumps(spec)  # must not raise (no bare Infinity token)
    fn = load_serving_spec(json.loads(s))
    out = fn(X["base"].to_numpy(), est.estimator_.predict(X))
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Unsupported transform raises clearly.
# ---------------------------------------------------------------------------
def test_export_unsupported_transform_raises():
    est, X, _ = _fit("diff")
    est.transform_name = "monotonic_residual"  # not in the lightweight table
    with pytest.raises(NotImplementedError, match="lightweight numpy inverse"):
        export_serving_spec(est)


def test_export_unfitted_raises():
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="base",
    )
    with pytest.raises(ValueError, match="not fitted"):
        export_serving_spec(est)


def test_load_unsupported_transform_raises():
    spec = {
        "transform_name": "rolling_quantile_ratio",
        "fitted_params": {},
        "fallback_predict": "y_train_median",
    }
    with pytest.raises(NotImplementedError, match="lightweight numpy inverse"):
        load_serving_spec(spec)


# ---------------------------------------------------------------------------
# Domain fallback + clip applied.
# ---------------------------------------------------------------------------
def test_serving_domain_fallback_applied():
    """A non-finite / out-of-domain base row routes to the fallback median."""
    est, X, _ = _fit("logratio")  # base must be > 0
    spec = export_serving_spec(est)
    fn = load_serving_spec(spec)
    base = X["base"].to_numpy().copy()
    raw = est.estimator_.predict(X)
    base[0] = -1.0  # violates logratio domain (base > 0)
    base[1] = np.nan
    out = fn(base, raw)
    fallback = float(spec["fitted_params"]["y_train_median"])
    assert out[0] == pytest.approx(fallback)
    assert out[1] == pytest.approx(fallback)
    # In-domain rows stay bit-identical to the full estimator (only rows 0/1
    # were poisoned on the base; the inner raw is unchanged for the rest).
    y_full = est.predict(X)
    np.testing.assert_array_equal(out[2:], y_full[2:])


def test_serving_nan_fallback_leaves_nan():
    est, X, _ = _fit("ratio")
    est.fallback_predict = "nan"
    spec = export_serving_spec(est)
    fn = load_serving_spec(spec)
    base = X["base"].to_numpy().copy()
    base[0] = 0.0  # |base| == 0 violates ratio domain
    out = fn(base, est.estimator_.predict(X))
    assert np.isnan(out[0])


def test_serving_y_clip_applied():
    """An inner raw prediction blowing past the y-envelope is clipped."""
    est, X, _ = _fit("linear_residual")
    spec = export_serving_spec(est)
    fn = load_serving_spec(spec)
    base = X["base"].to_numpy()
    raw = est.estimator_.predict(X).copy()
    raw[0] = 1e9  # gross blow-up -> t-clip then y-clip must bound it
    out = fn(base, raw)
    y_high = float(spec["fitted_params"]["y_clip_high"])
    assert out[0] <= y_high + 1e-9


# ---------------------------------------------------------------------------
# biz_value: serve path is dependency-light (no sklearn import triggered) AND
# bit-identical at atol 0 through a json round-trip.
# ---------------------------------------------------------------------------
def test_biz_val_serving_bit_identical_no_sklearn_on_serve_path():
    """For a fitted linear_residual composite, the lightweight load_serving_spec
    predict reproduces estimator.predict BIT-IDENTICALLY (atol 0) given the same
    inner raw predictions, round-tripping through json -- and the serve callable
    pulls in NO sklearn (the inverse maths is numpy-only)."""
    est, X, _ = _fit("linear_residual", n=1000, seed=7)
    y_full = est.predict(X)
    raw = est.estimator_.predict(X)
    base = X["base"].to_numpy()

    spec = json.loads(json.dumps(export_serving_spec(est)))

    # Drop sklearn from sys.modules; building + calling the serve predict must
    # not re-import it (proves the serve path is sklearn-free).
    removed = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "sklearn" or k.startswith("sklearn.")}
    try:
        fn = load_serving_spec(spec)
        y_serve = fn(base, raw)
        assert not any(m == "sklearn" or m.startswith("sklearn.") for m in sys.modules), "serve path must not import sklearn"
    finally:
        sys.modules.update(removed)

    # atol 0: exactly equal.
    np.testing.assert_array_equal(y_serve, y_full)
    assert LIGHTWEIGHT_TRANSFORMS  # sanity: table is non-empty
