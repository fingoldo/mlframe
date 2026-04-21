"""Tests for mlframe.scoring — the salvaged Models.py symbols."""

import numpy as np
import pytest

from mlframe.scoring import (
    ProbaScoreProxy,
    log_uniform,
    rmse_loss,
    rmse_score,
    rmsle_loss,
    rmsle_score,
)


def test_rmse_loss_zero_on_match():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse_loss(y, y) == 0.0


def test_rmse_loss_known_value():
    # sqrt(mean([1,1,1])) == 1
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    assert rmse_loss(y_true, y_pred) == pytest.approx(1.0)


def test_rmsle_loss_clips_negative_predictions():
    y_true = np.array([1.0, 2.0])
    y_pred_neg = np.array([-5.0, 2.0])
    y_pred_zero = np.array([0.0, 2.0])
    # negative preds clipped to 0 → same loss as predicting 0
    assert rmsle_loss(y_true, y_pred_neg) == pytest.approx(rmsle_loss(y_true, y_pred_zero))


def test_rmsle_loss_zero_on_match():
    y = np.array([0.5, 1.5, 10.0])
    assert rmsle_loss(y, y) == pytest.approx(0.0)


def test_rmse_and_rmsle_scorers_greater_is_better_false():
    # make_scorer with greater_is_better=False negates the output
    assert rmse_score._sign == -1
    assert rmsle_score._sign == -1


def test_log_uniform_bounds():
    lu = log_uniform(a=-2, b=2, base=10)
    samples = lu.rvs(size=500, random_state=42)
    assert samples.shape == (500,)
    assert np.all(samples >= 10 ** -2 - 1e-9)
    assert np.all(samples <= 10 ** 2 + 1e-9)


def test_log_uniform_scalar():
    lu = log_uniform(a=0, b=1, base=10)
    val = lu.rvs(random_state=1)
    assert np.isscalar(val) or val.shape == ()
    assert 1 <= float(val) <= 10


def test_log_uniform_random_state_reproducible():
    lu = log_uniform(-1, 1, base=10)
    a = lu.rvs(size=10, random_state=7)
    b = lu.rvs(size=10, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_proba_score_proxy_selects_column():
    from sklearn.metrics import roc_auc_score

    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_probs = np.array(
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4]]
    )
    # class 1 column == proba of positive class
    expected = roc_auc_score(y_true, y_probs[:, 1])
    assert ProbaScoreProxy(y_true, y_probs, class_idx=1, proxied_func=roc_auc_score) == expected
