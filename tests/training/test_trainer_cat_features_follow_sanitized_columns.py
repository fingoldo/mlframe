"""cat_features must be remapped together with the hostile column rename done before fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from mlframe.training.configs import DataConfig, MetricsConfig, NamingConfig, ReportingConfig, TrainingControlConfig
from mlframe.training.trainer import train_and_evaluate_model


class _Stop(Exception):
    """Raised by the spy to abort the fit once cat_features has been captured."""


def _run(monkeypatch, cat_features):
    """Run the trainer on a frame with one GBM-hostile column name and return the cat_features seen by the filter."""
    import mlframe.training._trainer_train_and_evaluate as mod

    seen: dict = {}

    def _spy(fit_params, train_df, val_df=None, test_df=None):
        """Record the cat list and the frame columns the filter would compare against."""
        seen["cat"] = list(fit_params["cat_features"])
        seen["cols"] = list(train_df.columns)
        raise _Stop

    monkeypatch.setattr(mod, "_filter_categorical_features", _spy)
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"mul(log(a),sin(b))": rng.integers(0, 3, n), "plain": rng.normal(size=n)})
    y = rng.normal(size=n)
    data = DataConfig(train_df=df, val_df=df.copy(), test_df=df.copy(), train_target=y, val_target=y, test_target=y)
    control = TrainingControlConfig(verbose=False, fit_params={"cat_features": cat_features}, compute_trainset_metrics=False)
    with pytest.raises(_Stop):
        train_and_evaluate_model(
            model=Ridge(),
            data=data,
            control=control,
            metrics=MetricsConfig(),
            reporting=ReportingConfig(print_report=False, show_perf_chart=False, show_fi=False),
            naming=NamingConfig(),
        )
    return seen


def test_hostile_named_cat_feature_follows_the_frame_rename(monkeypatch):
    """The cat list names the SAME safe column the frame was renamed to, so the filter keeps it."""
    seen = _run(monkeypatch, ["mul(log(a),sin(b))"])
    assert seen["cat"] == ["mul(log(a)_sin(b))"]
    assert seen["cat"][0] in seen["cols"]


def test_clean_cat_feature_is_untouched(monkeypatch):
    """A clean name passes through unchanged (negative control)."""
    seen = _run(monkeypatch, ["plain"])
    assert seen["cat"] == ["plain"]
