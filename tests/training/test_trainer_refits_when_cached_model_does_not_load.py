"""A cached model file that fails to load must be retrained, not evaluated unfitted."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import mlframe.training._trainer_train_and_evaluate as mod
from mlframe.training.configs import DataConfig, MetricsConfig, NamingConfig, ReportingConfig, TrainingControlConfig, OutputConfig
from mlframe.training.trainer import train_and_evaluate_model


def _run(model):
    """Train on a tiny frame with use_cache on and return the model the trainer hands back."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)})
    y = df["a"].to_numpy() * 2.0
    data = DataConfig(train_df=df, val_df=df.copy(), test_df=df.copy(), train_target=y, val_target=y, test_target=y)
    control = TrainingControlConfig(verbose=False, use_cache=True, compute_trainset_metrics=False)
    result = train_and_evaluate_model(
        model=model,
        data=data,
        control=control,
        metrics=MetricsConfig(),
        reporting=ReportingConfig(print_report=False, show_perf_chart=False, show_fi=False),
        naming=NamingConfig(),
        output=OutputConfig(),
    )
    return result


def test_unloadable_cache_file_triggers_a_refit(monkeypatch):
    """The loader hands the unfitted model back (a file that exists but cannot be read): the trainer fits it."""
    monkeypatch.setattr(mod, "_train_and_evaluate_use_cache_exists_model", lambda _uc, _f, _t, model, pre: (model, pre))
    model = Ridge()
    _run(model)
    assert hasattr(model, "coef_")
