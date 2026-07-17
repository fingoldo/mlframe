"""biz_value test for ``preprocessing.outlier_detector_zoo.make_ensemble_outlier_scores``.

Synthetic combines BOTH outlier shapes the single-detector biz_value test proved IsolationForest and LOF
disagree on: a GLOBAL outlier (far from every cluster -- IsolationForest's strength, isolated in very few tree
splits) and a LOCAL-density outlier (close to one cluster in absolute terms but far outside its tight local
density -- LOF's strength). Neither single detector should rank BOTH planted outliers at the very top; the
rank-averaged ensemble should.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from mlframe.preprocessing.outlier_detector_zoo import make_ensemble_outlier_scores, make_outlier_detector


def _make_mixed_outlier_dataset(seed: int = 0):
    """Helper that make mixed outlier dataset."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(loc=(0.0, 0.0), scale=0.15, size=(300, 2))
    sparse = rng.normal(loc=(15.0, 15.0), scale=3.0, size=(60, 2))
    local_outlier = np.array([[1.2, 1.2]])  # just outside the dense cluster; well within sparse-cluster spread
    global_outlier = np.array([[80.0, -80.0]])  # far from every cluster
    X = np.vstack([dense, sparse, local_outlier, global_outlier])
    labels = np.zeros(X.shape[0], dtype=int)
    local_row = dense.shape[0] + sparse.shape[0]
    global_row = local_row + 1
    labels[[local_row, global_row]] = 1
    return X, labels, local_row, global_row


def _single_detector_auc(method: str, X: np.ndarray, labels: np.ndarray, **kwargs) -> float:
    """Helper that single detector auc."""
    detector = make_outlier_detector(method, random_state=0, **kwargs)
    if method == "lof":
        detector.fit_predict(X)
        anomaly_score = -detector.negative_outlier_factor_
    else:
        detector.fit(X)
        anomaly_score = -detector.decision_function(X)
    return float(roc_auc_score(labels, anomaly_score))


def test_biz_val_ensemble_detects_both_global_and_local_outliers_better_than_either_alone():
    """Ensemble detects both global and local outliers better than either alone."""
    X, labels, local_row, global_row = _make_mixed_outlier_dataset()

    iso_auc = _single_detector_auc("isolation_forest", X, labels, n_estimators=300)
    lof_auc = _single_detector_auc("lof", X, labels, n_neighbors=20, novelty=False)

    ensemble_scores = make_ensemble_outlier_scores(
        X,
        methods=("isolation_forest", "lof"),
        random_state=0,
        detector_kwargs={"isolation_forest": {"n_estimators": 300}, "lof": {"n_neighbors": 20, "novelty": False}},
    )
    ensemble_auc = float(roc_auc_score(labels, ensemble_scores))

    assert ensemble_auc >= max(iso_auc, lof_auc), (
        f"expected ensemble AUC ({ensemble_auc:.4f}) to be at least as good as the better single detector "
        f"(iso={iso_auc:.4f}, lof={lof_auc:.4f}) on a synthetic with both a global and a local outlier"
    )
    assert ensemble_auc >= 0.85, f"expected ensemble to reliably separate both planted outlier types, got AUC={ensemble_auc:.4f}"

    # Rank both planted outliers among the top few most-anomalous rows -- neither single detector is
    # guaranteed to do this for BOTH points, but the ensemble should.
    order = np.argsort(-ensemble_scores)
    top_k = set(order[:5].tolist())
    assert local_row in top_k, "expected the ensemble to rank the local-density outlier near the top"
    assert global_row in top_k, "expected the ensemble to rank the global outlier near the top"


def test_outlier_detector_ensemble_requires_at_least_two_methods():
    """Outlier detector ensemble requires at least two methods."""
    import pytest

    with pytest.raises(ValueError):
        make_ensemble_outlier_scores(np.zeros((5, 2)), methods=("isolation_forest",))


def test_outlier_detector_ensemble_rejects_ecod():
    """Outlier detector ensemble rejects ecod."""
    import pytest

    with pytest.raises(ValueError, match="ecod"):
        make_ensemble_outlier_scores(np.zeros((5, 2)), methods=("isolation_forest", "ecod"))
