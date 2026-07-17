"""Unit + biz_value tests for ``composite.highlevel.discover_and_wrap``.

The one-call helper is pure orchestration over the composite public API. These
tests pin:

* the happy path returns a fitted estimator + a non-empty Markdown report
  (unit);
* on a synthetic with a discoverable composite structure the wrapped estimator's
  honest-test RMSE BEATS a plain inner trained on raw y (biz_value -- the actual
  measurable win the helper exists to deliver);
* the no-spec path returns ``estimator=None`` + a still-rendered report instead
  of crashing (edge);
* ``calibrate_conformal=True`` produces a usable prediction interval on a
  held-out slice (feature).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def _linear_composite_df(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """``y = 3.0 + 2.5*base + g(other) + small_noise``.

    ``base`` carries a strong LINEAR contribution to ``y``; ``other`` carries a
    weaker nonlinear one. A composite that subtracts the fitted linear part of
    ``base`` (``linear_residual``) leaves a small residual the inner learns from
    ``other`` cleanly, while a plain inner on raw ``y`` must fit the big linear
    ramp AND the residual together -- the composite should win on honest test.
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=2.0, size=n)
    other = rng.normal(size=n)
    y = 3.0 + 2.5 * base + 0.6 * np.sin(2.0 * other) + 0.15 * rng.normal(size=n)
    return pd.DataFrame({"base": base, "other": other, "y": y})


def _split(n: int, seed: int = 0):
    """Randomly splits n indices into 60/20/20 train/val/test partitions."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_tr = int(0.6 * n)
    n_va = int(0.2 * n)
    return perm[:n_tr], perm[n_tr : n_tr + n_va], perm[n_tr + n_va :]


def _small_inner():
    """Small deterministic inner, identical for composite + raw baseline so the
    RMSE delta isolates the composite transform, not the model family."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=200,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=20,
            verbose=-1,
            n_jobs=1,
            random_state=0,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            random_state=0,
        )


def _config():
    """Builds the shared CompositeTargetDiscoveryConfig used by the highlevel discover_and_wrap tests."""
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    return CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["base"],
        transforms=("diff", "ratio", "linear_residual"),
        top_k_after_mi=3,
        mi_sample_n=2000,
        eps_mi_gain=0.001,
        random_state=42,
        require_beats_raw_baseline=False,
        fail_on_no_gain="fallback_raw",
    )


def test_discover_and_wrap_returns_fitted_estimator_and_report():
    """Happy path: estimator fits + predicts, report_markdown is non-empty."""
    from mlframe.training.composite.highlevel import (
        DiscoverAndWrapResult,
        discover_and_wrap,
    )

    df = _linear_composite_df()
    train_idx, _, test_idx = _split(len(df))
    res = discover_and_wrap(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        config=_config(),
        base_estimator=_small_inner(),
    )
    assert isinstance(res, DiscoverAndWrapResult)
    assert res.spec is not None
    assert res.estimator is not None
    assert res.config is not None
    assert isinstance(res.report_markdown, str) and len(res.report_markdown) > 50

    X_test = df.iloc[test_idx][["base", "other"]]
    preds = res.estimator.predict(X_test)
    assert preds.shape[0] == test_idx.shape[0]
    assert np.isfinite(preds).all()


def test_biz_val_discover_and_wrap_beats_plain_inner_on_raw_y():
    """biz_value: wrapped composite RMSE < plain-inner-on-raw-y RMSE on honest
    test rows. Measured delta is large (composite removes the dominant linear
    ramp); floor at a conservative 5% relative improvement so a regression that
    silently drops the transform trips this."""
    from mlframe.training.composite.highlevel import discover_and_wrap

    df = _linear_composite_df()
    train_idx, _, test_idx = _split(len(df))

    res = discover_and_wrap(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        config=_config(),
        base_estimator=_small_inner(),
    )
    assert res.estimator is not None, "expected a discoverable composite spec"

    X_test = df.iloc[test_idx][["base", "other"]]
    y_test = df.iloc[test_idx]["y"].to_numpy()
    composite_rmse = float(np.sqrt(mean_squared_error(y_test, res.estimator.predict(X_test))))

    # Plain inner on raw y, SAME features (base + other), same family.
    plain = _small_inner()
    X_train = df.iloc[train_idx][["base", "other"]]
    y_train = df.iloc[train_idx]["y"].to_numpy()
    plain.fit(X_train, y_train)
    plain_rmse = float(np.sqrt(mean_squared_error(y_test, plain.predict(X_test))))

    assert composite_rmse < plain_rmse * 0.95, f"composite RMSE {composite_rmse:.4f} should beat plain {plain_rmse:.4f} by >=5% on this linear-ramp synthetic"


def test_discover_and_wrap_auto_config_when_none():
    """When ``config`` is omitted the helper auto-suggests one and still fits;
    config_rationale is populated."""
    from mlframe.training.composite.highlevel import discover_and_wrap

    df = _linear_composite_df()
    train_idx, _, _ = _split(len(df))
    res = discover_and_wrap(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        base_estimator=_small_inner(),
        config_overrides={"base_candidates": ["base"], "eps_mi_gain": 0.001, "require_beats_raw_baseline": False, "fail_on_no_gain": "fallback_raw"},
    )
    assert res.config is not None
    assert isinstance(res.config_rationale, dict)
    # auto-config always sets mi_sample_n (a steered field) -> rationale non-empty.
    assert res.config_rationale


def test_discover_and_wrap_no_spec_returns_none_estimator_and_report():
    """Edge: pure-noise target -> no spec clears the gate. estimator/spec None,
    report still renders (no crash)."""
    from mlframe.training.composite.highlevel import discover_and_wrap
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame(
        {
            "base": rng.normal(size=n),
            "other": rng.normal(size=n),
            "y": rng.normal(size=n),  # independent of every feature
        }
    )
    train_idx, _, _ = _split(n)
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["base"],
        transforms=("diff", "linear_residual"),
        mi_sample_n=1000,
        eps_mi_gain=0.5,  # absurdly high -> nothing passes
        random_state=42,
        require_beats_raw_baseline=False,
        fail_on_no_gain="fallback_raw",
    )
    res = discover_and_wrap(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        config=cfg,
        base_estimator=_small_inner(),
    )
    assert res.estimator is None
    assert res.spec is None
    assert isinstance(res.report_markdown, str) and res.report_markdown


def test_discover_and_wrap_conformal_interval_on_holdout():
    """Feature: calibrate_conformal=True yields a usable prediction interval with
    lower <= upper and roughly the requested marginal coverage on held-out
    rows."""
    from mlframe.training.composite.highlevel import discover_and_wrap

    df = _linear_composite_df()
    train_idx, val_idx, test_idx = _split(len(df))
    res = discover_and_wrap(
        df,
        target_col="y",
        feature_cols=["base", "other"],
        train_idx=train_idx,
        holdout_idx=val_idx,
        config=_config(),
        base_estimator=_small_inner(),
        calibrate_conformal=True,
        conformal_alpha=0.1,
    )
    assert res.estimator is not None
    assert res.conformal_alpha == pytest.approx(0.1)

    X_test = df.iloc[test_idx][["base", "other"]]
    y_test = df.iloc[test_idx]["y"].to_numpy()
    lower, upper = res.estimator.predict_interval(X_test, alpha=0.1)
    assert lower.shape == upper.shape == (test_idx.shape[0],)
    assert np.all(lower <= upper)
    coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
    # Split-conformal guarantees marginal coverage >= 1-alpha in expectation;
    # allow finite-sample slack on the lower side.
    assert coverage >= 0.82, f"conformal coverage {coverage:.3f} too low for alpha=0.1"
