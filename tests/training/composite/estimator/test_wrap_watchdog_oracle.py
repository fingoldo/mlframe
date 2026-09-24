"""The wrap-pass watchdog runs by default, checks against the raw split frame, and says so when it cannot run.

It ran only inside the metric block that the default ``skip_wrap_pass_predict=True`` skips; its universal check compared
the wrapper with the wrapper's own machinery; grouped transforms raised inside it for lack of ``groups``; and both its own
errors and a split whose predict raised were logged at DEBUG.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform
from mlframe.training.core._composite_wrap_watchdog import run_wrap_watchdog
from mlframe.training.core._phase_composite_wrapping import _run_composite_target_wrapping

from .test_watchdog_multibase import _setup_multibase_wrapper


class _TOracle(BaseEstimator, RegressorMixin):
    """An inner that returns given T values, so a consistent wrapper reproduces y exactly."""

    def __init__(self, t_values=None):
        self.t_values = t_values

    def fit(self, X, y):
        """No-op fit."""
        return self

    def predict(self, X):
        """The stored T values for the first ``len(X)`` rows."""
        return np.asarray(self.t_values, dtype=np.float64)[: len(X)]


def _watchdog_warnings(caplog) -> list[str]:
    """Messages of watchdog WARNING records."""
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and "watchdog" in r.getMessage().lower()]


def _wrap(ctx, *, skip_predict: bool) -> None:
    """Run the wrap pass on a prepared context."""
    _run_composite_target_wrapping(
        models=ctx["models"], metadata={}, target_by_type=ctx["target_by_type"], composite_specs_by_target_type=ctx["composite_specs"],
        filtered_train_idx=ctx["train_idx"], filtered_train_df=ctx["train_df"], filtered_val_idx=ctx["val_idx"], filtered_val_df=ctx["val_df"],
        test_idx=None, test_df_pd=None, skip_predict=skip_predict,
    )


def test_the_watchdog_runs_when_the_metric_block_is_skipped(caplog):
    """With the default skip of the metric block, a corrupted wrapper is still caught on a val sample."""
    ctx = _setup_multibase_wrapper(corrupt=True)
    with caplog.at_level(logging.WARNING):
        _wrap(ctx, skip_predict=True)
    assert _watchdog_warnings(caplog), "the watchdog must check a val sample when the metric block is skipped"


def test_the_watchdog_stays_quiet_on_a_consistent_wrapper_when_skipping(caplog):
    """Control: a consistent wrapper raises no watchdog warning on the skip path."""
    ctx = _setup_multibase_wrapper(corrupt=False)
    with caplog.at_level(logging.WARNING):
        _wrap(ctx, skip_predict=True)
    assert not _watchdog_warnings(caplog)


def _grouped_wrapper(n: int = 400, seed: int = 0):
    """A consistent ``linear_residual_grouped`` wrapper over a T oracle, with its spec, frame and y."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 4, n)
    base = rng.uniform(1.0, 10.0, n)
    y = (1.0 + 0.3 * grp) * base + rng.normal(0.0, 0.1, n)
    df = pd.DataFrame({"base": base, "feat": rng.normal(size=n), "grp": grp})
    t = get_transform("linear_residual_grouped")
    params = t.fit(y, base, groups=grp)
    T = t.forward(y, base, params, groups=grp)
    wrapper = CompositeTargetEstimator.from_fitted_inner(
        fitted_inner=_TOracle(T), transform_name="linear_residual_grouped", base_column="base", transform_fitted_params=params,
        y_train=y, group_column="grp",
    )
    spec = {"name": "y-lrg-base", "transform_name": "linear_residual_grouped", "base_column": "base", "fitted_params": params}
    return wrapper, spec, df, y


def test_a_grouped_transform_is_checked_with_its_groups(caplog):
    """The grouped forward gets the wrapper's group column: the check runs and a consistent wrapper passes it."""
    wrapper, spec, df, y = _grouped_wrapper()
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, df, y, composite_name="c", split_name="val")
    assert not _watchdog_warnings(caplog), _watchdog_warnings(caplog)


def test_a_grouped_wrapper_whose_prediction_drifts_is_caught(caplog, monkeypatch):
    """A wrapped prediction off by a constant breaks the additive y-error == T-error invariant."""
    wrapper, spec, df, y = _grouped_wrapper()
    orig = wrapper.predict
    monkeypatch.setattr(wrapper, "predict", lambda X, inner_X=None: np.asarray(orig(X)) + 0.5)
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, df, y, composite_name="c", split_name="val")
    assert any("y-MAE" in m for m in _watchdog_warnings(caplog))


def test_a_different_base_at_predict_is_caught(caplog, monkeypatch):
    """The base the wrapper reads at predict is compared with the spec's base columns in the split frame."""
    wrapper, spec, df, y = _grouped_wrapper()
    orig = wrapper._extract_base_for_transform
    monkeypatch.setattr(wrapper, "_extract_base_for_transform", lambda X, cols: np.asarray(orig(X, cols)) * 1.1)
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, df, y, composite_name="c", split_name="val")
    assert any("base the wrapper reads" in m for m in _watchdog_warnings(caplog))


def test_a_check_that_cannot_run_is_reported(caplog):
    """A spec the watchdog cannot evaluate is logged at WARNING, not swallowed at DEBUG."""
    wrapper, spec, df, y = _grouped_wrapper()
    broken = dict(spec, fitted_params={"not": "params"})
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, broken, df, y, composite_name="c", split_name="val")
    assert any("could not run" in m for m in _watchdog_warnings(caplog))


def test_a_split_whose_predict_raises_is_reported(caplog):
    """A composite whose predict raises no longer vanishes from the y-scale verdict with only a DEBUG line."""
    ctx = _setup_multibase_wrapper(corrupt=False)
    entry = ctx["models"]["regression"]["y-linresmulti-base0"][0]

    def _boom(*_a, **_k):
        raise RuntimeError("predict exploded")

    # The shared predict engine raises, so every scoring path (predict, predict_with_pre_clip) fails the way a broken inner does.
    entry.model._predict_unclipped = _boom
    with caplog.at_level(logging.WARNING):
        _wrap(ctx, skip_predict=False)
    assert any("skipped" in r.getMessage() and "predict exploded" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


@pytest.mark.parametrize("name", ["quantile_residual"])
def test_a_non_additive_transform_gets_no_additive_alarm(caplog, name):
    """``quantile_residual`` inverts as ``T * IQR + median``: the additive invariant must not be applied to it."""
    rng = np.random.default_rng(1)
    n = 400
    base = rng.uniform(1.0, 10.0, n)
    y = 2.0 * base + rng.normal(0.0, 1.0, n)
    t = get_transform(name)
    params = t.fit(y, base)
    T = t.forward(y, base, params)
    wrapper = CompositeTargetEstimator.from_fitted_inner(fitted_inner=_TOracle(T + 0.3), transform_name=name, base_column="base", transform_fitted_params=params, y_train=y)
    spec = {"name": "y-q-base", "transform_name": name, "base_column": "base", "fitted_params": params}
    with caplog.at_level(logging.WARNING):
        run_wrap_watchdog(wrapper, spec, pd.DataFrame({"base": base}), y, composite_name="c", split_name="val")
    assert not any("y-MAE" in m for m in _watchdog_warnings(caplog))
