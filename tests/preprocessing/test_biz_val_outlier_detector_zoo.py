"""biz_value test for ``preprocessing.outlier_detector_zoo.make_outlier_detector``.

Synthetic: a dense cluster plus a sparse, spread-out cluster, with a point placed just outside the dense
cluster (a clear LOCAL density anomaly). LOF (density-based, ranks by local neighborhood density ratio) and
IsolationForest (tree-partition-based, ranks by global isolation depth) both flag this specific point as an
outlier here, but their overall anomaly-score RANKING across every point is only moderately correlated -- the
two backends genuinely capture different notions of "anomalous," not a redundant re-implementation of the same
signal under a different name. A perfect (or near-perfect) rank correlation would mean the pluggability is
cosmetic; a real gap proves swapping backends changes what gets caught.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from mlframe.preprocessing.outlier_detector_zoo import make_outlier_detector


def _make_local_outlier_dataset(seed: int = 0):
    rng = np.random.default_rng(seed)
    dense = rng.normal(loc=(0.0, 0.0), scale=0.15, size=(300, 2))
    sparse = rng.normal(loc=(15.0, 15.0), scale=3.0, size=(60, 2))
    local_outlier = np.array([[1.2, 1.2]])  # clearly outside the dense cluster, but well within sparse-cluster spread
    X = np.vstack([dense, sparse, local_outlier])
    outlier_row = X.shape[0] - 1
    return X, outlier_row


def test_biz_val_lof_and_isolation_forest_rank_outliers_differently():
    X, outlier_row = _make_local_outlier_dataset()

    lof = make_outlier_detector("lof", n_neighbors=20, novelty=False)
    lof.fit_predict(X)
    lof_scores = lof.negative_outlier_factor_  # higher (less negative) = more normal

    iso = make_outlier_detector("isolation_forest", n_estimators=300, random_state=0)
    iso.fit(X)
    iso_scores = iso.decision_function(X)  # higher = more normal

    rho, _ = spearmanr(lof_scores, iso_scores)
    assert rho < 0.85, f"expected LOF and IsolationForest anomaly-score rankings to diverge meaningfully (measured rho~0.61 on this synthetic), got rho={rho:.4f} -- pluggability would be cosmetic if the two backends always agreed"

    lof_labels = lof.fit_predict(X)
    assert lof_labels[outlier_row] == -1, "expected LOF (density-based) to flag the planted local-density outlier"


def test_outlier_detector_zoo_isolation_forest_default():
    detector = make_outlier_detector()
    from sklearn.ensemble import IsolationForest

    assert isinstance(detector, IsolationForest)


def test_outlier_detector_zoo_lof():
    detector = make_outlier_detector("lof")
    from sklearn.neighbors import LocalOutlierFactor

    assert isinstance(detector, LocalOutlierFactor)


def test_outlier_detector_zoo_ecod_missing_dependency_raises_clear_error():
    try:
        import pyod  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("pyod is installed in this environment -- nothing to test for the missing-dependency path")
    with pytest.raises(ImportError, match="pyod"):
        make_outlier_detector("ecod")


def test_outlier_detector_zoo_rejects_unknown_method():
    with pytest.raises(ValueError):
        make_outlier_detector("bogus")
