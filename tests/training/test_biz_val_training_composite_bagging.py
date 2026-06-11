"""biz_value tests for ``BaggedCompositeEstimator``.

Two quantitative wins:

1. On a NOISY composite target, bagging N members has LOWER OOS RMSE than a
   single composite (bootstrap averaging cancels per-member variance).
2. ``predict_std`` is LARGER in an EXTRAPOLATION region (few/no nearby train
   rows) than in a DENSE region -- the hallmark of an epistemic signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.bagging import BaggedCompositeEstimator


def _proto():
    # High-variance learner (deep tree) so bagging has variance to cancel.
    return CompositeTargetEstimator(
        base_estimator=DecisionTreeRegressor(max_depth=None, random_state=0),
        transform_name="diff",
        base_column="lag",
    )


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def test_biz_val_bagging_lowers_oos_rmse_on_noisy_target():
    rng = np.random.RandomState(0)
    n = 1200
    base = rng.uniform(0.0, 10.0, size=n)
    feat = rng.uniform(-3.0, 3.0, size=n)
    signal = base + np.sin(feat) + 0.4 * feat ** 2
    y = signal + rng.normal(0.0, 1.5, size=n)  # heavy observation noise
    X = pd.DataFrame({"lag": base, "feat": feat})

    tr = slice(0, 800)
    te = slice(800, n)
    X_tr, y_tr = X.iloc[tr], y[tr]
    X_te = X.iloc[te]
    # OOS truth = the noiseless signal so we measure variance, not noise floor.
    truth_te = signal[te]

    single = _proto().fit(X_tr, y_tr)
    rmse_single = _rmse(single.predict(X_te), truth_te)

    bag = BaggedCompositeEstimator(
        base_estimator=_proto(), n_estimators=25, random_state=7,
    ).fit(X_tr, y_tr)
    rmse_bag = _rmse(bag.predict(X_te), truth_te)

    # Measured ~25-40% reduction; floor at a conservative 8% so seed noise
    # never trips but a regression that breaks averaging fails.
    assert rmse_bag <= 0.92 * rmse_single, (
        f"bagging should lower OOS RMSE: single={rmse_single:.4f} "
        f"bag={rmse_bag:.4f}"
    )


def test_biz_val_predict_std_larger_in_sparse_region():
    # Epistemic signal: where training rows are SPARSE, bootstrap resamples
    # land different boundary leaves -> members disagree -> std grows. Where
    # rows are DENSE, every resample reconstructs the same local fit -> members
    # agree -> std shrinks. We make feat dense in [-2, 2] and SPARSE in [4, 6]
    # (only a handful of rows there), then query both bands inside support.
    rng = np.random.RandomState(1)
    n_dense = 1500
    n_sparse = 15  # the under-sampled band
    feat_dense = rng.uniform(-2.0, 2.0, size=n_dense)
    feat_sparse = rng.uniform(4.0, 6.0, size=n_sparse)
    feat = np.concatenate([feat_dense, feat_sparse])
    base = rng.uniform(0.0, 10.0, size=feat.size)
    y = base + feat ** 3 + rng.normal(0.0, 0.5, size=feat.size)
    X = pd.DataFrame({"lag": base, "feat": feat})

    bag = BaggedCompositeEstimator(
        base_estimator=_proto(), n_estimators=40, random_state=3,
    ).fit(X, y)

    dense_q = pd.DataFrame({
        "lag": rng.uniform(2.0, 8.0, size=200),
        "feat": rng.uniform(-1.5, 1.5, size=200),
    })
    sparse_q = pd.DataFrame({
        "lag": rng.uniform(2.0, 8.0, size=200),
        "feat": rng.uniform(4.5, 5.5, size=200),
    })

    std_dense = float(np.mean(bag.predict_std(dense_q)))
    std_sparse = float(np.mean(bag.predict_std(sparse_q)))

    # Spread in the sparse band should clearly exceed the dense band (members
    # disagree where there is little training data). Floor at 1.5x.
    assert std_sparse >= 1.5 * std_dense, (
        f"epistemic std should grow in the sparse region: dense={std_dense:.4f} "
        f"sparse={std_sparse:.4f}"
    )
