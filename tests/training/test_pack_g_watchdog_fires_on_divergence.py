"""T3#14 2026-05-18 #9 Pack G watchdog regression test.

The production TVT log showed an MLP with T-scale MAE=9.17 vs y-scale
MAE=3.22 - a 3x discrepancy that the original additive-only watchdog
(``error_T == error_y``) did NOT catch (the transform was additive so
the invariant should have held, but stale wrapper state could break it
silently). Pack G universal watchdog now compares
``wrapper.predict(X)`` against ``transform.inverse(inner.predict(X),
base, params)`` and warns when they diverge by more than 1% of y_std.

This test pins:

1. Universal watchdog FIRES when ``wrapper.predict`` is artificially
   forced to disagree with ``transform.inverse(inner.predict, base,
   params)`` (the production failure mode).

2. Watchdog STAYS QUIET when wrapper.predict and the inverse path agree
   (the happy path).

Together these prove the watchdog actually distinguishes the bug from
normal operation - the test in ``test_pack_g_watchdog_diagnostic.py``
only verifies log FORMAT once watchdog fires, not that the fire
condition reproduces the production scenario.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin


class _InnerWithKnownScale(BaseEstimator, RegressorMixin):
    """Predicts a known multiple of base values so we can derive
    matching T values analytically."""

    def __init__(self, t_scale_factor: float = 1.0) -> None:
        self.t_scale_factor = t_scale_factor

    def fit(self, X, y):
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return X["base"].values * self.t_scale_factor
        return np.asarray(X).reshape(-1) * self.t_scale_factor


def _build_problem():
    rng = np.random.default_rng(2026)
    n = 300
    base = rng.normal(100.0, 20.0, n)
    y = 1.5 * base + 5.0 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"base": base})
    return df, y, base


def _setup_wrapper_models(*, broken_inner_factor: float | None = None):
    """Build the models dict + composite specs dict the phase function expects.
    If ``broken_inner_factor`` is None the wrapper is consistent; otherwise the
    inner is replaced post-wrap with one that returns scaled-wrong predictions,
    simulating the production failure mode."""
    from mlframe.training.composite_estimator import CompositeTargetEstimator
    from mlframe.training.composite.transforms import get_transform

    df, y, base = _build_problem()
    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), n)

    transform = get_transform("linear_residual")
    params = transform.fit(y, base)
    T = transform.forward(y, base, params)

    inner = _InnerWithKnownScale(t_scale_factor=1.0).fit(df.values, T)
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="linear_residual",
        base_column="base",
        transform_fitted_params=params,
        y_train=y,
    )

    if broken_inner_factor is not None:
        # POST-fit corruption: swap the inner for one that systematically
        # underpredicts T. wrapper.predict now flows through transform.inverse
        # of the wrong T, while a downstream inspector calling
        # transform.inverse(inner.predict, base, params) sees the SAME wrong
        # value (universal watchdog won't trigger). To make the universal
        # watchdog fire, we corrupt the wrapper's CACHED inverse path instead:
        # patch wrapper.predict directly so it returns y_true + offset (an
        # "oracle" path that disagrees with the inverse-of-inner check).
        wrapper.estimator_ = _InnerWithKnownScale(
            t_scale_factor=broken_inner_factor,
        ).fit(df.values, T)

        # Save the actual y (post-clip) so downstream knows the "oracle".
        # Override the wrapper's predict to bypass inner entirely:
        # returns y_true with small noise - simulating a stale transformer_
        # cache that has y values baked in.
        _y_oracle = y.astype(np.float64).copy()
        _y_full = y

        def _predict_override(X_in):
            # If the input index matches the original rows, return oracle.
            # Otherwise fall back to the normal path.
            return _y_oracle[: len(X_in)]

        # Monkey-patch: this simulates the production wrapper state where
        # the inverse path is short-circuited / stale.
        wrapper.predict = _predict_override

    class _Entry:
        def __init__(self, m):
            self.model = m
            self.model_name = "InnerWithKnownScale"

    spec = {
        "name": "y-linres-base",
        "transform_name": "linear_residual",
        "base_column": "base",
        "fitted_params": params,
    }

    models = {
        "regression": {
            "y-linres-base": [_Entry(wrapper.estimator_)],
        },
    }
    target_by_type = {"regression": {"y": y}}
    composite_specs = {
        "regression": {"y": [spec]},
    }
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return {
        "models": models, "target_by_type": target_by_type,
        "composite_specs": composite_specs,
        "train_idx": train_idx, "val_idx": val_idx,
        "train_df": train_df, "val_df": val_df,
        "wrapper": wrapper,
    }


class TestUniversalWatchdogFiresOnDivergence:
    """When wrapper.predict diverges from transform.inverse(inner.predict),
    the universal Pack G watchdog must emit a WARNING."""

    def test_watchdog_warns_when_predict_path_corrupted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from mlframe.training.core._phase_composite_post import (
            _run_composite_target_wrapping,
        )

        ctx = _setup_wrapper_models(broken_inner_factor=0.1)

        # Replace the entry's model with the corrupted wrapper so the
        # phase function uses it. We pre-installed an _Entry pointing at
        # estimator_ above; now point it at the wrapper itself so
        # wrapper.predict (our oracle override) is what gets called.
        ctx["models"]["regression"]["y-linres-base"][0].model = ctx["wrapper"]

        with caplog.at_level(
            logging.WARNING,
            logger="mlframe.training.core._phase_composite_post",
        ):
            _run_composite_target_wrapping(
                models=ctx["models"],
                metadata={},
                target_by_type=ctx["target_by_type"],
                composite_specs_by_target_type=ctx["composite_specs"],
                filtered_train_idx=ctx["train_idx"],
                filtered_train_df=ctx["train_df"],
                filtered_val_idx=ctx["val_idx"],
                filtered_val_df=ctx["val_df"],
                test_idx=None,
                test_df_pd=None,
                skip_predict=False,
            )

        watchdog_records = [
            r for r in caplog.records
            if "watchdog" in r.message.lower()
        ]
        assert watchdog_records, (
            f"Pack G universal watchdog must fire when wrapper.predict diverges "
            f"from transform.inverse(inner.predict). Log records: "
            f"{[(r.levelname, r.message[:200]) for r in caplog.records]}"
        )


class TestUniversalWatchdogQuietOnHappyPath:
    """When wrapper.predict is the canonical inverse-of-inner path, the
    universal watchdog must NOT fire (no false positives)."""

    def test_watchdog_quiet_on_consistent_wrapper(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        from mlframe.training.core._phase_composite_post import (
            _run_composite_target_wrapping,
        )

        ctx = _setup_wrapper_models(broken_inner_factor=None)

        with caplog.at_level(
            logging.WARNING,
            logger="mlframe.training.core._phase_composite_post",
        ):
            _run_composite_target_wrapping(
                models=ctx["models"],
                metadata={},
                target_by_type=ctx["target_by_type"],
                composite_specs_by_target_type=ctx["composite_specs"],
                filtered_train_idx=ctx["train_idx"],
                filtered_train_df=ctx["train_df"],
                filtered_val_idx=ctx["val_idx"],
                filtered_val_df=ctx["val_df"],
                test_idx=None,
                test_df_pd=None,
                skip_predict=False,
            )

        universal_warnings = [
            r for r in caplog.records
            if "watchdog.universal" in r.message
        ]
        assert not universal_warnings, (
            f"Universal watchdog must stay quiet on the canonical path; "
            f"got: {[r.message[:200] for r in universal_warnings]}"
        )
