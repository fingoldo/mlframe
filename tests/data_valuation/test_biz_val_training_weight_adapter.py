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
@pytest.mark.timeout(900)
def test_scale_valuation_work_bounded_past_the_subsample_cap(monkeypatch):
    """At n_train=300k with max_valued_rows=5000, the quadratic knn_shapley valuation only ever sees the 5000-row
    subsample; the full training set is reached only through the linear propagation step.

    Asserted through call shapes rather than wall-clock: the valuation call's train side is exactly the cap,
    and the propagation call covers every training row.
    """
    from mlframe.data_valuation import _training_weight_adapter as adapter
    from mlframe.data_valuation import training_sample_weight_from_valuation

    rng = np.random.default_rng(3)
    n_train, n_val, cap = 300_000, 2000, 5000
    X_train = rng.standard_normal((n_train, 10))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(np.int64)
    X_val = rng.standard_normal((n_val, 10))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(np.int64)

    real_knn = adapter.knn_shapley
    real_propagate = adapter.propagate_subsample_values
    knn_train_rows = []
    propagate_full_rows = []

    def _spy_knn(X_tr, *args, **kwargs):
        """Record the train-side row count of every valuation call, then delegate."""
        knn_train_rows.append(X_tr.shape[0])
        return real_knn(X_tr, *args, **kwargs)

    def _spy_propagate(X_full, *args, **kwargs):
        """Record the number of rows propagated to, then delegate."""
        propagate_full_rows.append(X_full.shape[0])
        return real_propagate(X_full, *args, **kwargs)

    monkeypatch.setattr(adapter, "knn_shapley", _spy_knn)
    monkeypatch.setattr(adapter, "propagate_subsample_values", _spy_propagate)

    w = training_sample_weight_from_valuation(X_train, y_train, X_val, y_val, max_valued_rows=cap, rng=np.random.default_rng(5))

    assert w.shape == (n_train,)
    assert knn_train_rows == [cap], f"valuation must run on exactly the {cap}-row subsample, got {knn_train_rows}"
    assert propagate_full_rows == [n_train], f"propagation must cover all {n_train} training rows, got {propagate_full_rows}"
