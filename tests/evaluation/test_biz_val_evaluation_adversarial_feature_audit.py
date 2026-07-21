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
    """Builds seeded synthetic test data; returns ``(X_train, y_train, X_test)``."""
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
    """Adversarial feature audit keeps genuinely predictive shifted feature."""
    X_train, y_train, X_test = _make_data(seed=0)

    result = adversarial_validation_feature_audit(X_train, y_train, X_test, top_k_features=5, seed=0, lgbm_params={"n_estimators": 50, "verbosity": -1})

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
    """Adversarial feature audit returns correlation field."""
    X_train, y_train, X_test = _make_data(seed=1)
    result = adversarial_validation_feature_audit(X_train, y_train, X_test, top_k_features=5, seed=1, lgbm_params={"n_estimators": 50, "verbosity": -1})
    assert "importance_vs_generalization_correlation" in result
    assert len(result["audited_features"]) > 0


def test_biz_val_adversarial_feature_audit_stability_folds_default_omitted_is_bit_identical():
    """stability_folds is opt-in: omitting it must reproduce the exact single-split output."""
    X_train, y_train, X_test = _make_data(seed=0)
    kwargs = dict(top_k_features=5, seed=0, lgbm_params={"n_estimators": 50, "verbosity": -1})

    baseline = adversarial_validation_feature_audit(X_train, y_train, X_test, **kwargs)
    repeat = adversarial_validation_feature_audit(X_train, y_train, X_test, **kwargs)

    assert baseline["adversarial_auc"] == repeat["adversarial_auc"]
    assert "stability_folds" not in baseline
    for a, b in zip(baseline["audited_features"], repeat["audited_features"]):
        assert a["name"] == b["name"]
        assert a["private_auc_delta_when_dropped"] == b["private_auc_delta_when_dropped"]
        assert "stability" not in a


def _make_stability_data(seed: int):
    """A dataset with one robust-verdict feature and one borderline feature whose single-split verdict is
    genuinely a coin flip -- its true effect on private AUC is ~0, so which side of 0 a given random pseudo-
    split lands on is noise, giving it a much higher cross-fold keep/drop flip rate than the robust feature.
    """
    rng = np.random.default_rng(seed)
    n_train, n_test = 2500, 1000

    y_train = rng.integers(0, 2, size=n_train)
    y_test_latent = rng.integers(0, 2, size=n_test)

    # robust feature: strongly and unambiguously predictive, shifted so it's adversarial-flagged. Its
    # ablation effect on private AUC is large and consistently negative across any reshuffle.
    robust_train = y_train * 3.0 + rng.normal(0, 1.0, size=n_train)
    robust_test = y_test_latent * 3.0 + rng.normal(0, 1.0, size=n_test) + 3.0

    # borderline feature: only very weakly (near-noise) related to y, but strongly shifted so it still gets
    # adversarial-flagged. Its true ablation effect hovers around 0, so different pseudo-splits should
    # disagree on whether dropping it helps or hurts -- an unstable keep/drop call.
    borderline_train = y_train * 0.05 + rng.normal(0, 1.0, size=n_train)
    borderline_test = y_test_latent * 0.05 + rng.normal(0, 1.0, size=n_test) + 3.0

    filler_train = rng.normal(0, 1, size=(n_train, 3))
    filler_test = rng.normal(0, 1, size=(n_test, 3))

    X_train = pd.DataFrame(
        {
            "robust": robust_train,
            "borderline": borderline_train,
            "filler_0": filler_train[:, 0],
            "filler_1": filler_train[:, 1],
            "filler_2": filler_train[:, 2],
        }
    )
    X_test = pd.DataFrame(
        {
            "robust": robust_test,
            "borderline": borderline_test,
            "filler_0": filler_test[:, 0],
            "filler_1": filler_test[:, 1],
            "filler_2": filler_test[:, 2],
        }
    )
    return X_train, y_train, X_test


def test_biz_val_adversarial_feature_audit_stability_folds_distinguishes_robust_from_noisy_calls():
    """Adversarial feature audit stability folds distinguishes robust from noisy calls."""
    X_train, y_train, X_test = _make_stability_data(seed=0)

    result = adversarial_validation_feature_audit(
        X_train,
        y_train,
        X_test,
        top_k_features=5,
        seed=0,
        lgbm_params={"n_estimators": 50, "verbosity": -1},
        stability_folds=6,
    )

    assert result["stability_folds"] == 6
    by_name = {a["name"]: a for a in result["audited_features"]}
    assert "robust" in by_name and "borderline" in by_name

    robust = by_name["robust"]["stability"]
    borderline = by_name["borderline"]["stability"]

    # the robust feature's delta stays consistently negative (dropping it clearly hurts) -> unanimous vote,
    # low variance across reshuffles.
    assert robust["stable"] is True, f"robust feature should get a unanimous keep/drop call across folds: {robust}"
    assert robust["keep_frac"] == 1.0

    # the borderline feature's true effect is near zero, so different pseudo-splits flip its sign -> its
    # vote is not unanimous. Quantify "how far from a unanimous call" as distance from 0/1: the robust
    # feature must land exactly on 0 (fully unanimous), the borderline feature must land materially off it.
    robust_distance_from_unanimous = min(robust["keep_frac"], 1.0 - robust["keep_frac"])
    borderline_distance_from_unanimous = min(borderline["keep_frac"], 1.0 - borderline["keep_frac"])
    assert robust_distance_from_unanimous == 0.0
    assert (
        borderline_distance_from_unanimous >= 0.25
    ), f"borderline feature's verdict should flip on a substantial share of folds, not just a fluke: {borderline}"
