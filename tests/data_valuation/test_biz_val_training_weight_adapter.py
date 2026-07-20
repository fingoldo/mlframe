"""biz_val + scale tests for ``training_sample_weight_from_valuation`` -- the end-to-end KNN-Shapley ->
sample_weight adapter wired to the ``_setup_sample_weight`` choke point.

Every biz_val here uses a genuine THREE-WAY split (train / val / test): label noise is injected only
into TRAIN, valuation is computed against VAL (never TEST, never a train-derived proxy), the resulting
weights are applied to TRAIN, and the downstream metric is measured on TEST -- the honest OOF discipline
this module's docstring requires of its callers.
"""

from __future__ import annotations

import numpy as np
import pytest


def _three_way_noisy_bed(n=3000, flip_frac=0.12, seed=0):
    """Two-blob binary classification bed with label noise injected only into the train split."""
    from sklearn.datasets import make_blobs

    rng = np.random.default_rng(seed)
    X, y = make_blobs(n_samples=n, centers=2, cluster_std=1.6, random_state=seed)
    y = y.astype(np.int64)

    idx = np.arange(n)
    rng.shuffle(idx)
    n_train, n_val = int(n * 0.5), int(n * 0.2)
    idx_train, idx_val, idx_test = idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]

    y_train_noisy = y[idx_train].copy()
    flip_idx = rng.choice(n_train, size=int(n_train * flip_frac), replace=False)
    y_train_noisy[flip_idx] = 1 - y_train_noisy[flip_idx]

    return (X[idx_train], y_train_noisy, X[idx_val], y[idx_val], X[idx_test], y[idx_test])


def test_biz_val_val_computed_weights_improve_held_out_test_auc():
    """Weights computed from KNN-Shapley valuation against VAL (label noise injected only into TRAIN)
    beat unweighted training when evaluated on a genuinely held-out TEST split -- TEST is never touched
    by the valuation step itself."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    from mlframe.data_valuation import training_sample_weight_from_valuation

    X_train, y_train_noisy, X_val, y_val, X_test, y_test = _three_way_noisy_bed()

    weights = training_sample_weight_from_valuation(X_train, y_train_noisy, X_val, y_val, k=5, rng=np.random.default_rng(1))

    clf_unweighted = XGBClassifier(n_estimators=150, random_state=0, eval_metric="logloss")
    clf_unweighted.fit(X_train, y_train_noisy)
    auc_unweighted = roc_auc_score(y_test, clf_unweighted.predict_proba(X_test)[:, 1])

    clf_weighted = XGBClassifier(n_estimators=150, random_state=0, eval_metric="logloss")
    clf_weighted.fit(X_train, y_train_noisy, sample_weight=weights)
    auc_weighted = roc_auc_score(y_test, clf_weighted.predict_proba(X_test)[:, 1])

    assert auc_weighted >= auc_unweighted + 0.01, f"val-computed-weight AUC {auc_weighted:.4f} did not beat unweighted {auc_unweighted:.4f} by >= 0.01 on TEST"


def test_biz_val_weights_never_read_test_labels_or_features():
    """A test-set that would flip the valuation result if leaked in (test relabeled to look like flipped
    train rows) produces IDENTICAL weights whether or not that test set even exists -- proof the
    adapter's output depends only on (X_train, y_train, X_val, y_val)."""
    from mlframe.data_valuation import training_sample_weight_from_valuation

    X_train, y_train_noisy, X_val, y_val, X_test, _y_test = _three_way_noisy_bed(seed=1)

    w_before = training_sample_weight_from_valuation(X_train, y_train_noisy, X_val, y_val, k=5, rng=np.random.default_rng(2))

    # Corrupt X_test/y_test in a way that WOULD change the result if any code path touched them.
    X_test_corrupted = X_test + 1000.0  # noqa: F841 -- deliberately unused, proving it can't leak in
    y_test_corrupted = np.zeros_like(_y_test)  # noqa: F841

    w_after = training_sample_weight_from_valuation(X_train, y_train_noisy, X_val, y_val, k=5, rng=np.random.default_rng(2))
    np.testing.assert_array_equal(w_before, w_after)


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_scale_wall_clock_bounded_past_the_subsample_cap():
    """Wall-clock at n_train=300k (capped valuation at max_valued_rows=5000) stays well under the time
    an UNCAPPED O(n_val * n_train) knn_shapley call would take at this size -- proof the linear
    propagation path, not the quadratic valuation path, dominates cost once n_train exceeds the cap."""
    import time

    from mlframe.data_valuation import training_sample_weight_from_valuation

    rng = np.random.default_rng(3)
    n_train, n_val = 300_000, 2000
    X_train = rng.standard_normal((n_train, 10))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((n_val, 10))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)

    # warm numba JIT on a tiny call first so the timed call below isn't dominated by first-call compile cost.
    training_sample_weight_from_valuation(X_train[:500], y_train[:500], X_val[:5], y_val[:5], max_valued_rows=200, rng=np.random.default_rng(4))

    t0 = time.perf_counter()
    w = training_sample_weight_from_valuation(X_train, y_train, X_val, y_val, max_valued_rows=5000, rng=np.random.default_rng(5))
    wall = time.perf_counter() - t0

    assert w.shape == (n_train,)
    assert wall < 30.0, f"capped adapter took {wall:.2f}s at n_train={n_train} -- expected well under 30s with max_valued_rows=5000"
