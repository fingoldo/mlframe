"""Cluster 28 regression: a failed early-stopping strip is logged at WARNING, not DEBUG (audit 2026-11-20, finding 28-F56).

The failure leaves native early stopping wired alongside slice early stopping, which changes training behaviour, so it must be visible in a
normal run. The test drives the public ``train_and_evaluate_model`` with slice-stable ES enabled and a LightGBM model whose ``set_params``
refuses the strip, then checks the record the run emits.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import (
    DataConfig,
    MetricsConfig,
    NamingConfig,
    PredictionsContainer,
    ReportingConfig,
    TrainingControlConfig,
)
from mlframe.training._training_runtime_configs import SliceStableESConfig
from mlframe.training.trainer import train_and_evaluate_model

_STRIP_MESSAGE = "does not expose set_params for early_stopping_rounds"


def _make_model_class(set_params_refuses: bool):
    """Build a LightGBM regressor whose ``get_params`` advertises native early stopping and whose ``set_params`` may refuse it.

    The class name starts with ``LGBM`` so mlframe resolves the LightGBM model category from it.
    """
    lgb = pytest.importorskip("lightgbm")

    class LGBMRegressorStrip(lgb.LGBMRegressor):
        """LightGBM regressor that reports ``early_stopping_rounds`` in its params."""

        def get_params(self, deep=True):
            """Expose a native early-stopping parameter, as the strip step expects."""
            params = super().get_params(deep=deep)
            params["early_stopping_rounds"] = 10
            return params

        def fit(self, X, y, **kwargs):
            """Accept only the fit kwargs LightGBM knows; mlframe passes trainer-only keys."""
            return super().fit(X, y, **{k: v for k, v in kwargs.items() if k in {"eval_set", "sample_weight"}})

        def set_params(self, **params):
            """Refuse the early-stopping strip when asked to, otherwise set the parameters normally."""
            if set_params_refuses and "early_stopping_rounds" in params:
                raise ValueError("set_params refused")
            return super().set_params(**params)

    return LGBMRegressorStrip


def _run(model, slice_config: SliceStableESConfig, caplog) -> list[logging.LogRecord]:
    """Run ``train_and_evaluate_model`` on a small regression split and return the WARNING records it emitted."""
    rng = np.random.default_rng(0)

    def _frame(n):
        """A random regression frame with a linear target."""
        X = rng.uniform(0, 1, (n, 4))
        return pd.DataFrame(X, columns=list("abcd")), X.sum(axis=1) + rng.normal(0, 0.1, n)

    (Xtr, ytr), (Xv, yv), (Xte, yte) = _frame(200), _frame(80), _frame(80)
    df = pd.concat([Xtr, Xv, Xte], ignore_index=True)
    target = np.concatenate([ytr, yv, yte])
    data = DataConfig(
        df=df,
        target=target,
        train_df=Xtr,
        val_df=Xv,
        test_df=Xte,
        train_target=ytr,
        val_target=yv,
        test_target=yte,
        train_idx=np.arange(200),
        val_idx=np.arange(200, 280),
        test_idx=np.arange(280, 360),
        sample_weight=None,
    )
    control = TrainingControlConfig(
        verbose=False,
        compute_trainset_metrics=False,
        compute_valset_metrics=True,
        compute_testset_metrics=True,
        just_evaluate=False,
        pre_pipeline=None,
        model_category="lgb",
        slice_stable_es=slice_config,
    )
    with caplog.at_level(logging.WARNING):
        train_and_evaluate_model(
            model,
            data,
            control,
            MetricsConfig(),
            ReportingConfig(print_report=False, show_perf_chart=False, show_fi=False),
            NamingConfig(),
            predictions=PredictionsContainer(test_preds=None),
        )
    return [r for r in caplog.records if r.levelno >= logging.WARNING and _STRIP_MESSAGE in r.getMessage()]


def test_refused_strip_is_logged_at_warning_with_its_cause(caplog):
    """A model that refuses the strip logs one WARNING naming the cause; native ES stays wired alongside slice ES (28-F56)."""
    model = _make_model_class(set_params_refuses=True)(n_estimators=5, verbose=-1)
    records = _run(model, SliceStableESConfig(enabled=True, k=2, source="random", min_rows_per_shard=1), caplog)

    assert len(records) == 1
    assert "set_params refused" in records[0].getMessage()
    assert "native ES stays wired" in records[0].getMessage()


def test_accepted_strip_logs_nothing(caplog):
    """Control: a model that accepts the strip logs no strip warning (28-F56)."""
    model = _make_model_class(set_params_refuses=False)(n_estimators=5, verbose=-1)
    assert _run(model, SliceStableESConfig(enabled=True, k=2, source="random", min_rows_per_shard=1), caplog) == []


def test_diagnostic_only_never_attempts_the_strip(caplog):
    """Control: in diagnostic-only mode the native ES path is left intact, so the refusing model is never asked to strip (28-F56)."""
    model = _make_model_class(set_params_refuses=True)(n_estimators=5, verbose=-1)
    slice_config = SliceStableESConfig(enabled=False, diagnostic_only=True, k=2, source="random", min_rows_per_shard=1)
    assert _run(model, slice_config, caplog) == []
