"""Unit tests for CompositeTargetEstimator UX/robustness improvements:

1. ``_repr_html_`` -- rich Jupyter HTML repr (transform / base column(s) /
   headline fitted params / n_train_valid / conformal+CQR calibration state).
2. ``get_params`` completeness -- every __init__ arg is round-trippable through
   set_params + sklearn.clone for a non-default config (incl.
   recurrence_continuation) so GridSearchCV / clone work.
3. clone() robustness with an UNKNOWN (non-registry) transform_name -- fit must
   raise a clear error EARLY (not deep inside the inverse), and clone must
   preserve every init param.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator


@pytest.fixture
def linres_kit():
    """Small frame for a linear_residual transform (needs a base column)."""
    rng = np.random.default_rng(0)
    n = 80
    base = rng.normal(10.0, 2.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.5 * f1 + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f1": f1})
    return X, pd.Series(y), X  # X reused for cal set below


# ---------------------------------------------------------------------------
# 1. _repr_html_
# ---------------------------------------------------------------------------

def test_repr_html_unfitted_shows_config_and_not_fitted():
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
    )
    html = cte._repr_html_()
    assert isinstance(html, str)
    assert "CompositeTargetEstimator" in html
    assert "linear_residual" in html
    assert "base" in html
    assert "LinearRegression" in html
    # Unfitted: marks not-fitted and conformal/CQR uncalibrated.
    assert ">no<" in html
    assert "not calibrated" in html


def test_repr_html_fitted_shows_alpha_beta_clip_and_n_train(linres_kit):
    X, y, _ = linres_kit
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
    )
    cte.fit(X, y)
    html = cte._repr_html_()
    # Headline fitted params.
    assert "alpha" in html
    assert "beta" in html
    assert "y_clip_low" in html
    assert "y_clip_high" in html
    # n_train_valid row populated.
    assert "n_train_valid" in html
    assert str(int(cte.fitted_params_["n_train_valid"])) in html
    # Fitted marker.
    assert ">yes<" in html


def test_repr_html_reflects_conformal_and_cqr_calibration(linres_kit):
    X, y, X_cal = linres_kit
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
    )
    cte.fit(X, y)
    # Before calibration.
    assert cte._repr_html_().count("not calibrated") == 2
    # Calibrate split-conformal at one level.
    cte.calibrate_conformal(X_cal, y.to_numpy(), alpha=0.1)
    html = cte._repr_html_()
    assert "calibrated @ alpha=0.1" in html
    # CQR still not calibrated -> exactly one "not calibrated" remains.
    assert html.count("not calibrated") == 1


def test_repr_html_never_raises_on_broken_state():
    """A notebook repr must not blow up; a malformed instance returns the
    fail-soft div rather than propagating."""
    cte = CompositeTargetEstimator.__new__(CompositeTargetEstimator)
    # No attributes set at all -> _build_repr_html hits getattr defaults.
    html = cte._repr_html_()
    assert "CompositeTargetEstimator" in html


# ---------------------------------------------------------------------------
# 2. get_params completeness + round-trip
# ---------------------------------------------------------------------------

def test_get_params_returns_every_init_arg():
    """Every __init__ parameter (except self) must appear in get_params so
    sklearn GridSearchCV / clone can reconstruct the estimator faithfully."""
    init_params = [
        name for name in inspect.signature(CompositeTargetEstimator.__init__).parameters
        if name != "self"
    ]
    cte = CompositeTargetEstimator()
    got = set(cte.get_params(deep=False).keys())
    missing = [p for p in init_params if p not in got]
    assert not missing, f"get_params is missing init args: {missing}"


def test_get_params_set_params_clone_round_trip_non_default():
    """A fully non-default config (incl. recurrence_continuation) survives
    get_params -> set_params and sklearn.clone with every value preserved."""
    cfg = dict(
        base_estimator=LinearRegression(),
        transform_name="ewma_residual",
        base_column="base_legacy",
        fallback_predict="nan",
        drop_invalid_rows=False,
        auto_variance_stabilise=True,
        base_columns=["b1", "b2"],
        group_column="grp",
        online_refit_enabled=True,
        online_refit_buffer_n=5_000,
        online_refit_z_threshold=2.5,
        online_refit_min_buffer_n=123,
        recurrence_continuation=True,
    )
    cte = CompositeTargetEstimator(**cfg)

    # set_params round-trip onto a fresh instance.
    params = cte.get_params(deep=False)
    fresh = CompositeTargetEstimator()
    fresh.set_params(**params)
    for k, v in cfg.items():
        if k == "base_estimator":
            assert type(getattr(fresh, k)) is LinearRegression
        else:
            assert getattr(fresh, k) == v, f"set_params lost {k!r}"

    # sklearn.clone round-trip (the GridSearchCV path).
    cloned = clone(cte)
    assert cloned is not cte
    for k, v in cfg.items():
        if k == "base_estimator":
            assert type(getattr(cloned, k)) is LinearRegression
            assert getattr(cloned, k) is not cte.base_estimator  # cloned, not shared
        else:
            assert getattr(cloned, k) == v, f"clone lost {k!r}"
    # The recurrence flag specifically (the param the task calls out).
    assert cloned.recurrence_continuation is True


# ---------------------------------------------------------------------------
# 3. clone robustness + early error on unknown transform
# ---------------------------------------------------------------------------

def test_unknown_transform_fit_raises_early_with_clear_message(linres_kit):
    """An unknown transform_name must fail at fit() up-front (transform lookup),
    NOT silently proceed and explode deep inside the inverse at predict."""
    X, y, _ = linres_kit
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="totally_made_up_transform",
        base_column="base",
    )
    with pytest.raises(KeyError) as exc_info:
        cte.fit(X, y)
    msg = str(exc_info.value)
    assert "totally_made_up_transform" in msg
    assert "Unknown transform" in msg
    # Must have raised BEFORE fitting an inner estimator.
    assert not hasattr(cte, "estimator_")


def test_clone_preserves_unknown_transform_name():
    """clone() of an unfitted wrapper with a custom/unknown transform_name must
    preserve all params (the validation happens at fit, not at construction or
    clone) so a Pipeline/GridSearchCV that clones-then-fits surfaces the same
    clear error path on every clone."""
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="my_custom_transform",
        base_column="base",
        recurrence_continuation=True,
    )
    cloned = clone(cte)
    assert cloned.transform_name == "my_custom_transform"
    assert cloned.base_column == "base"
    assert cloned.recurrence_continuation is True
    assert type(cloned.base_estimator) is LinearRegression


def test_cloned_unknown_transform_still_raises_clear_error_on_fit(linres_kit):
    """End-to-end: clone an unknown-transform wrapper, then fit the clone -->
    same early, clear KeyError. Guards the GridSearchCV clone-then-fit flow."""
    X, y, _ = linres_kit
    cte = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="nope_not_registered",
        base_column="base",
    )
    cloned = clone(cte)
    with pytest.raises(KeyError) as exc_info:
        cloned.fit(X, y)
    assert "nope_not_registered" in str(exc_info.value)
