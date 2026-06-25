"""M7: base-margin residual classification (CompositeClassificationEstimator).

The composite anchors a GBDT on a cheap base model's log-odds (init_score /
base_margin / baseline) so it learns only the RESIDUAL log-odds. Biz_value: on a
target whose signal is a dominant LINEAR driver + a nonlinear (XOR) residual the
linear base cannot see, the composite beats the base alone and matches a plain
GBDT that must re-derive everything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.training.composite import CompositeClassificationEstimator
from mlframe.training.composite.classification import _sigmoid

lgb = pytest.importorskip("lightgbm")


def _xor_residual_data(seed=1, n=8000):
    rng = np.random.default_rng(seed)
    s = rng.normal(0.0, 1.0, n)            # dominant linear driver
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    xor = np.sign(a * b)                    # nonlinear residual; linear base blind
    logit = 2.5 * s + 1.8 * xor
    p = _sigmoid(logit)
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"s": s, "a": a, "b": b})
    return X, y


class TestM7BizValue:
    def test_residual_over_base_beats_base_and_matches_plain_gbdt(self) -> None:
        X, y = _xor_residual_data()
        tr, te = slice(0, 5000), slice(5000, None)
        base = LogisticRegression(max_iter=1000).fit(X.iloc[tr], y[tr])
        auc_base = roc_auc_score(y[te], base.predict_proba(X.iloc[te])[:, 1])
        plain = lgb.LGBMClassifier(n_estimators=200, verbose=-1).fit(X.iloc[tr], y[tr])
        auc_plain = roc_auc_score(y[te], plain.predict_proba(X.iloc[te])[:, 1])
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=200, verbose=-1),
        ).fit(X.iloc[tr], y[tr])
        auc_comp = roc_auc_score(y[te], est.predict_proba(X.iloc[te])[:, 1])
        assert auc_comp >= auc_base + 0.03, (
            f"composite {auc_comp:.4f} should clearly beat base {auc_base:.4f}"
        )
        assert auc_comp >= auc_plain - 0.01, (
            f"composite {auc_comp:.4f} should match plain GBDT {auc_plain:.4f}"
        )


class TestM7Contract:
    def test_predict_proba_rows_sum_to_one(self) -> None:
        X, y = _xor_residual_data(n=2000)
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=50, verbose=-1),
        ).fit(X, y)
        proba = est.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_precomputed_margin_column_is_stripped(self) -> None:
        X, y = _xor_residual_data(n=3000)
        # A precomputed (oracle-ish) base margin column.
        X = X.copy()
        X["base_logit"] = 2.5 * X["s"].to_numpy()
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=80, verbose=-1),
            base_margin_column="base_logit",
        ).fit(X, y)
        # The inner must have been fit WITHOUT the margin column. The composite's own
        # ``n_features_in_`` follows the sklearn convention (== the full X passed to fit,
        # margin column included -- it is plumbing, not a learned dim), so the contract is
        # asserted on the INNER estimator, which sees X with the margin column dropped.
        assert est.estimator_.n_features_in_ == X.shape[1] - 1
        assert est.n_features_in_ == X.shape[1]
        auc = roc_auc_score(y, est.predict_proba(X)[:, 1])
        assert auc > 0.85

    def test_multiclass_is_supported(self) -> None:
        # Multiclass is now first-class (softmax over (n, K) base + residual
        # margins); a 3-class target fits and predicts in-range.
        X, _ = _xor_residual_data(n=900)
        y3 = np.tile([0, 1, 2], len(X) // 3 + 1)[: len(X)]
        est = CompositeClassificationEstimator(
            base_estimator=lgb.LGBMClassifier(n_estimators=20, verbose=-1),
        ).fit(X, y3)
        proba = est.predict_proba(X)
        assert proba.shape == (len(X), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    def test_single_class_raises(self) -> None:
        X, _ = _xor_residual_data(n=300)
        with pytest.raises(ValueError, match=">= 2 classes"):
            CompositeClassificationEstimator(
                base_estimator=lgb.LGBMClassifier(n_estimators=10, verbose=-1),
            ).fit(X, np.zeros(len(X), dtype=int))

    def test_inner_without_margin_path_rejected(self) -> None:
        X, y = _xor_residual_data(n=600)
        with pytest.raises(NotImplementedError):
            CompositeClassificationEstimator(
                base_estimator=LogisticRegression(max_iter=200),
            ).fit(X, y)
