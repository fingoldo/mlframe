"""Regression: the y-scale wrap-pass watchdog must cover multi-base specs.

``linear_residual_multi`` is in ``_ADDITIVE_TRANSFORMS`` so both watchdog
blocks (universal predict-vs-inverse-of-inner, and the additive
T-MAE==y-MAE invariant) are supposed to guard it. Pre-fix, both blocks
built a 1-D base from ``spec['base_column']`` alone. For a K-base spec the
transform's ``inverse``/``forward`` then raise

    linear_residual_multi: base has 1 columns but fitted alphas has K entries

which the surrounding ``except ...: logger.debug(...)`` swallowed -- so a
corrupted multi-base wrapper produced NO watchdog WARNING at all. The
spec family most prone to wrapper-math bugs (multi-column base matrix vs
K saved alphas) had zero watchdog coverage.

The fix builds the full ``(n, K)`` base matrix (primary + ``extra_base_columns``)
via the same ``_extract_base_matrix`` helper the wrapper uses, so the
inverse/forward succeed and the divergence is detected.

These tests pin:

1. The watchdog FIRES when a multi-base wrapper.predict is corrupted to
   diverge from ``transform.inverse(inner.predict, base_matrix, params)``.
   On pre-fix code the watchdog crashed internally on the base-width
   mismatch and emitted nothing -> this test FAILS pre-fix.
2. The watchdog stays QUIET on a consistent multi-base wrapper (no false
   positive, and -- crucially -- proves the check actually ran rather
   than silently crashing).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin


class _MultiBaseInnerT(BaseEstimator, RegressorMixin):
    """Predicts the supplied T values (so the canonical wrapper agrees) or
    a scaled-wrong version when ``t_scale_factor != 1`` (to corrupt)."""

    def __init__(self, t_values: np.ndarray, t_scale_factor: float = 1.0) -> None:
        self.t_values = np.asarray(t_values, dtype=np.float64)
        self.t_scale_factor = t_scale_factor

    def fit(self, X, y):
        return self

    def predict(self, X):
        n = len(X)
        return self.t_values[:n] * self.t_scale_factor


def _build_multibase_problem():
    rng = np.random.default_rng(20260611)
    n = 300
    base0 = rng.normal(100.0, 20.0, n)
    base1 = rng.normal(50.0, 10.0, n)
    # y depends on BOTH base columns -> a genuine multi-base relationship.
    y = 1.5 * base0 - 0.8 * base1 + 7.0 + rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({"base0": base0, "base1": base1})
    return df, y, base0, base1


def _setup_multibase_wrapper(*, corrupt: bool):
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.training.composite.transforms import get_transform

    df, y, base0, base1 = _build_multibase_problem()
    n = len(df)
    train_idx = np.arange(int(0.7 * n))
    val_idx = np.arange(int(0.7 * n), n)

    transform = get_transform("linear_residual_multi")
    base_matrix = np.column_stack([base0, base1])
    params = transform.fit(y, base_matrix)
    assert len(params["alphas"]) == 2, "expected a 2-base fit (alphas has K=2 entries)"
    T = transform.forward(y, base_matrix, params)

    inner = _MultiBaseInnerT(t_values=T).fit(df.values, T)
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=inner,
        transform_name="linear_residual_multi",
        base_column="base0",
        base_columns=("base0", "base1"),
        transform_fitted_params=params,
        y_train=y,
    )

    if corrupt:
        # Short-circuit wrapper.predict to an oracle (y itself) that diverges
        # from transform.inverse(inner.predict, base_matrix, params). Mirrors
        # the single-base watchdog regression's stale-cache simulation.
        _y_oracle = y.astype(np.float64).copy()

        def _predict_override(X_in):
            return _y_oracle[: len(X_in)]

        wrapper.predict = _predict_override

    class _Entry:
        def __init__(self, m):
            self.model = m
            self.model_name = "MultiBaseInnerT"

    spec = {
        "name": "y-linresmulti-base0",
        "transform_name": "linear_residual_multi",
        "base_column": "base0",
        "extra_base_columns": ("base1",),
        "fitted_params": params,
    }

    models = {"regression": {"y-linresmulti-base0": [_Entry(wrapper)]}}
    target_by_type = {"regression": {"y": y}}
    composite_specs = {"regression": {"y": [spec]}}
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    return {
        "models": models,
        "target_by_type": target_by_type,
        "composite_specs": composite_specs,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "train_df": train_df,
        "val_df": val_df,
    }


def _run_wrap(ctx):
    from mlframe.training.core._phase_composite_post import (
        _run_composite_target_wrapping,
    )

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


class TestMultiBaseWatchdogFires:
    def test_watchdog_warns_when_multibase_predict_corrupted(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = _setup_multibase_wrapper(corrupt=True)
        with caplog.at_level(
            logging.WARNING,
            logger="mlframe.training.core._phase_composite_post",
        ):
            _run_wrap(ctx)

        watchdog_records = [r for r in caplog.records if "watchdog" in r.message.lower()]
        assert watchdog_records, (
            "Watchdog must fire for a corrupted linear_residual_multi wrapper. "
            "Pre-fix the watchdog built a 1-D base and the inverse/forward raised "
            "the alphas-width mismatch, swallowed at DEBUG -> no WARNING. "
            f"Records: {[(r.levelname, r.message[:160]) for r in caplog.records]}"
        )


class TestMultiBaseWatchdogQuietOnHappyPath:
    def test_watchdog_quiet_on_consistent_multibase(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = _setup_multibase_wrapper(corrupt=False)
        with caplog.at_level(
            logging.WARNING,
            logger="mlframe.training.core._phase_composite_post",
        ):
            _run_wrap(ctx)

        watchdog_warnings = [r for r in caplog.records if "watchdog" in r.message.lower()]
        assert not watchdog_warnings, f"Watchdog must stay quiet on a consistent multi-base wrapper; got: {[r.message[:160] for r in watchdog_warnings]}"
