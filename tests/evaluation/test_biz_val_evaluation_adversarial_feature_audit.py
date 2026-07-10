"""biz_value test for ``evaluation.adversarial_validation_feature_audit``.

The win: a feature that is genuinely predictive of the target but happens to have a shifted marginal
distribution between train and test (so it scores high on adversarial-AUC contribution) should be recommended
"keep" by the audit -- dropping it should measurably HURT pseudo-private AUC -- while a pure train/test-split
artifact with zero true relationship to the target should not show the same harm when dropped. This
replicates the source writeup's finding that adversarial-AUC contribution alone is a poor predictor of whether
a feature actually helps or hurts generalization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.evaluation.adversarial_feature_audit import adversarial_validation_feature_audit


def _make_data(seed: int):
    rng = np.random.default_rng(seed)
    n_train, n_test = 2500, 1000

    y_train = rng.integers(0, 2, size=n_train)
    # genuinely predictive of y, but its marginal distribution is shifted between train/test (drift) --
    # this alone makes it a strong adversarial-AUC contributor despite remaining useful for predicting y.
    good_shifted_train = y_train * 2.0 + rng.normal(0, 1.0, size=n_train)
    good_shifted_test = rng.integers(0, 2, size=n_test) * 2.0 + rng.normal(0, 1.0, size=n_test) + 3.0

    # pure train/test-split artifact (monotonically increasing "row order" proxy): perfectly separates
    # train from test but has ZERO true relationship with y.
    id_noise_train = np.arange(n_train, dtype=np.float64)
    id_noise_test = np.arange(n_train, n_train + n_test, dtype=np.float64)

    # a few uninformative filler features so the adversarial classifier has more than 2 columns to rank.
    filler_train = rng.normal(0, 1, size=(n_train, 3))
    filler_test = rng.normal(0, 1, size=(n_test, 3))

    X_train = pd.DataFrame(
        {
            "good_shifted": good_shifted_train,
            "id_noise": id_noise_train,
            "filler_0": filler_train[:, 0],
            "filler_1": filler_train[:, 1],
            "filler_2": filler_train[:, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "good_shifted": good_shifted_test,
            "id_noise": id_noise_test,
            "filler_0": filler_test[:, 0],
            "filler_1": filler_test[:, 1],
            "filler_2": filler_test[:, 2],
        }
    )
    return X_train, y_train, X_test


def test_biz_val_adversarial_feature_audit_keeps_genuinely_predictive_shifted_feature():
    X_train, y_train, X_test = _make_data(seed=0)

    result = adversarial_validation_feature_audit(
        X_train, y_train, X_test, top_k_features=5, seed=0, lgbm_params={"n_estimators": 50, "verbosity": -1}
    )

    assert result["adversarial_auc"] > 0.8, f"the synthetic drift should be strongly adversarial-detectable, got {result['adversarial_auc']:.3f}"

    by_name = {a["name"]: a for a in result["audited_features"]}
    assert "good_shifted" in by_name, "the genuinely predictive shifted feature should rank among the top adversarial contributors"

    good = by_name["good_shifted"]
    assert good["recommendation"] == "keep", f"dropping the genuinely predictive feature should hurt pseudo-private AUC: {good}"
    assert good["private_auc_delta_when_dropped"] < 0

    if "id_noise" in by_name:
        id_noise = by_name["id_noise"]
        # the pure split-artifact feature is not genuinely predictive: dropping it should not hurt as much
        # as dropping the genuinely predictive feature (and should not show a strongly negative delta).
        assert id_noise["private_auc_delta_when_dropped"] >= good["private_auc_delta_when_dropped"]


def test_adversarial_feature_audit_returns_correlation_field():
    X_train, y_train, X_test = _make_data(seed=1)
    result = adversarial_validation_feature_audit(
        X_train, y_train, X_test, top_k_features=5, seed=1, lgbm_params={"n_estimators": 50, "verbosity": -1}
    )
    assert "importance_vs_generalization_correlation" in result
    assert len(result["audited_features"]) > 0
