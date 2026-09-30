"""``configure_training_params`` must hand out RFECV selectors / estimators that stdlib pickle accepts.

The RFECV scorer used to be a closure local to ``configure_training_params``:

    AttributeError: Can't pickle local object 'configure_training_params.<locals>.fs_and_hpt_integral_calibration_error'

which silently pushed ``save_mlframe_model`` onto its dill fallback.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from mlframe.training._trainer_configure import configure_training_params


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 1500
    df = pd.DataFrame({"f": rng.standard_normal(n)})
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    return df, y


def _configure(df, y, models, calibrated):
    n = len(df)
    return configure_training_params(
        df=df, train_df=df.iloc[:1000], val_df=df.iloc[1000:1250], test_df=df.iloc[1250:],
        target=y, train_target=y.iloc[:1000], val_target=y.iloc[1000:1250], test_target=y.iloc[1250:],
        train_idx=np.arange(1000), val_idx=np.arange(1000, 1250), test_idx=np.arange(1250, n),
        use_regression=False, prefer_gpu_configs=False, mlframe_models=models, verbose=False,
        config_params={"iterations": 10}, prefer_calibrated_classifiers=calibrated,
    )


def test_calibrated_rfecv_scorer_survives_stdlib_pickle_roundtrip(data):
    df, y = data
    out = _configure(df, y, ["lgb"], calibrated=True)
    rfecv = out[3]  # lgb_rfecv
    restored = pickle.loads(pickle.dumps(rfecv))
    assert "fs_and_hpt_integral_calibration_error" in repr(restored.scoring)

    rng = np.random.default_rng(1)
    yt = (rng.random(800) > 0.5).astype(int)
    p = np.clip(rng.random(800), 1e-3, 1 - 1e-3)
    proba = np.column_stack([1 - p, p])
    before = rfecv.scoring._score_func(yt, proba)
    after = restored.scoring._score_func(yt, proba)
    assert before == after


def test_xgb_default_eval_metric_survives_stdlib_pickle_roundtrip(data):
    df, y = data
    out = _configure(df, y, ["xgb"], calibrated=False)
    est = out[1]["xgb"]["model"]
    restored = pickle.loads(pickle.dumps(est))
    metric = restored.get_params()["eval_metric"]
    assert metric.__name__ == "neg_ovr_roc_auc_score"
    yt = np.array([0, 1, 0, 1, 1, 0])
    proba = np.array([[.8, .2], [.3, .7], [.6, .4], [.2, .8], [.4, .6], [.7, .3]])
    assert metric(yt, proba[:, 1]) == est.get_params()["eval_metric"](yt, proba[:, 1])
