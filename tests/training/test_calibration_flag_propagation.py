"""Regression tests for behavior-config flag propagation.

These tests defend against the class of bug where a flag declared on
``TrainingBehaviorConfig`` reaches ``configure_training_params`` but then
fails to alter the actual model it is supposed to control. Weaker
"config-passthrough" tests (e.g. that the config value reaches the
configuration function) do NOT catch this — the flag must demonstrably
change what the downstream model is built/fit with.

Level 2 tests (targeted):
    Flipping ``prefer_calibrated_classifiers`` must change
    - CatBoostClassifier.eval_metric  (CB_CALIB_CLASSIF vs CB_CLASSIF)
    - XGBClassifier.eval_metric       (XGB_CALIB_CLASSIF vs XGB_GENERAL_CLASSIF)
    - LGBM fit_params['eval_metric']  (injected vs absent)

Level 3 test (matrix invariant):
    Any future flag that changes a tree-model eval_metric will trip the
    parametric sweep below, which asserts that the classification
    eval_metrics differ between the two values of the flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.trainer import (
    _configure_xgboost_params,
    _configure_lightgbm_params,
    configure_training_params,
)
from mlframe.training.helpers import get_training_configs
from mlframe.metrics import ICE


def _identity(x):
    return x


@pytest.fixture(scope="module")
def real_configs():
    cpu = get_training_configs(iterations=10, has_gpu=False)
    return cpu, cpu


def _classifier_eval_metric(model):
    """Extract eval_metric from a fitted-or-unfitted sklearn-style classifier."""
    params = model.get_params()
    return params.get("eval_metric")


# ---------------------------------------------------------------------------
# Level 2 — targeted: prefer_calibrated_classifiers flips base model config
# ---------------------------------------------------------------------------

def test_xgb_classifier_eval_metric_differs_with_flag(real_configs):
    cpu, gpu = real_configs
    kwargs = dict(
        configs=gpu, cpu_configs=cpu,
        use_regression=False,
        prefer_cpu_for_xgboost=True,
        use_flaml_zeroshot=False,
        xgboost_verbose=False,
        metamodel_func=_identity,
    )
    out_true = _configure_xgboost_params(prefer_calibrated_classifiers=True, **kwargs)
    out_false = _configure_xgboost_params(prefer_calibrated_classifiers=False, **kwargs)

    em_true = _classifier_eval_metric(out_true["model"])
    em_false = _classifier_eval_metric(out_false["model"])

    assert em_true is not em_false, (
        "prefer_calibrated_classifiers=True must change XGBClassifier.eval_metric; "
        "both values returned the same object"
    )


def test_lgb_classifier_fit_params_eval_metric_only_when_calibrated(real_configs):
    cpu, gpu = real_configs
    kwargs = dict(
        configs=gpu, cpu_configs=cpu,
        use_regression=False,
        prefer_cpu_for_lightgbm=True,
        use_flaml_zeroshot=False,
        metamodel_func=_identity,
    )
    out_true = _configure_lightgbm_params(prefer_calibrated_classifiers=True, **kwargs)
    out_false = _configure_lightgbm_params(prefer_calibrated_classifiers=False, **kwargs)

    assert "eval_metric" in out_true["fit_params"], (
        "prefer_calibrated_classifiers=True must inject eval_metric into LGBM fit_params"
    )
    assert "eval_metric" not in out_false["fit_params"], (
        "prefer_calibrated_classifiers=False must NOT inject eval_metric into LGBM fit_params"
    )


def test_lgb_regressor_ignores_flag(real_configs):
    """Sanity: flag affects only classification; regression path is unchanged."""
    cpu, gpu = real_configs
    kwargs = dict(
        configs=gpu, cpu_configs=cpu,
        use_regression=True,
        prefer_cpu_for_lightgbm=True,
        use_flaml_zeroshot=False,
        metamodel_func=_identity,
    )
    out_true = _configure_lightgbm_params(prefer_calibrated_classifiers=True, **kwargs)
    out_false = _configure_lightgbm_params(prefer_calibrated_classifiers=False, **kwargs)
    assert out_true["fit_params"] == out_false["fit_params"] == {}


def test_cb_classifier_eval_metric_differs_with_flag():
    """CB has no module-level helper; test via configure_training_params directly."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({"f": rng.standard_normal(n)})
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    common_kwargs = dict(
        df=df, train_df=df.iloc[:40], val_df=df.iloc[40:50], test_df=df.iloc[50:],
        target=y, train_target=y.iloc[:40], val_target=y.iloc[40:50], test_target=y.iloc[50:],
        train_idx=np.arange(40), val_idx=np.arange(40, 50), test_idx=np.arange(50, n),
        use_regression=False,
        prefer_gpu_configs=False,
        mlframe_models=["cb"],
        verbose=False,
        config_params={"iterations": 10},
    )
    out_true = configure_training_params(**common_kwargs, prefer_calibrated_classifiers=True)
    out_false = configure_training_params(**common_kwargs, prefer_calibrated_classifiers=False)

    cb_true = out_true[1]["cb"]["model"]
    cb_false = out_false[1]["cb"]["model"]

    em_true = _classifier_eval_metric(cb_true)
    em_false = _classifier_eval_metric(cb_false)

    # Calibrated path uses an ICE(...) instance; uncalibrated uses the string "AUC".
    assert isinstance(em_true, ICE), (
        f"Expected ICE(...) eval_metric when prefer_calibrated_classifiers=True, got {em_true!r}"
    )
    assert em_false == "AUC", (
        f"Expected 'AUC' eval_metric when prefer_calibrated_classifiers=False, got {em_false!r}"
    )


# ---------------------------------------------------------------------------
# Level 3 — matrix-invariant: flipping the flag changes eval_metric for EVERY
# classification model that claims to support calibration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_key", ["cb", "xgb", "lgb"])
def test_calibration_flag_differentiates_classifier(model_key):
    """For each supported classifier, prefer_calibrated_classifiers=True vs
    False must produce a different eval_metric (either on the model or in
    fit_params). This is the invariant that catches silent no-op
    regressions like the 2026-04-15 refactor.
    """
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({"f0": rng.standard_normal(n), "f1": rng.standard_normal(n)})
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    common_kwargs = dict(
        df=df, train_df=df.iloc[:40], val_df=df.iloc[40:50], test_df=df.iloc[50:],
        target=y, train_target=y.iloc[:40], val_target=y.iloc[40:50], test_target=y.iloc[50:],
        train_idx=np.arange(40), val_idx=np.arange(40, 50), test_idx=np.arange(50, n),
        use_regression=False,
        prefer_gpu_configs=False,
        mlframe_models=[model_key],
        verbose=False,
        config_params={"iterations": 10},
    )
    out_true = configure_training_params(**common_kwargs, prefer_calibrated_classifiers=True)
    out_false = configure_training_params(**common_kwargs, prefer_calibrated_classifiers=False)

    entry_true = out_true[1][model_key]
    entry_false = out_false[1][model_key]

    em_true = _classifier_eval_metric(entry_true["model"])
    em_false = _classifier_eval_metric(entry_false["model"])
    fp_em_true = entry_true.get("fit_params", {}).get("eval_metric")
    fp_em_false = entry_false.get("fit_params", {}).get("eval_metric")

    model_differs = em_true != em_false  # works for scalars AND object identity-by-value
    fit_params_differ = fp_em_true != fp_em_false

    assert model_differs or fit_params_differ, (
        f"{model_key}: prefer_calibrated_classifiers is a no-op — neither model.eval_metric "
        f"({em_true!r} vs {em_false!r}) nor fit_params.eval_metric "
        f"({fp_em_true!r} vs {fp_em_false!r}) changes when the flag is flipped"
    )
