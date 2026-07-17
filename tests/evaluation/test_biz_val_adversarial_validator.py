"""biz_value test for ``evaluation.AdversarialValidator``.

The win (via the facade's two composed halves): (1) diagnostic -- when train is a mixture of a majority
regime and a minority regime that test is drawn entirely from, the adversarial classifier should detect
strong train/test separability (AUC well above chance) and correctly flag the shifted feature as the top
importance; (2) fold selection -- validating on the facade's "most test-like" fold should track true test
performance far more closely than a random validation fold, exactly mirroring
``build_test_like_validation_fold``'s own biz_value test but exercised through the unified class API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from mlframe.evaluation import AdversarialValidator


def _make_shifted_scenario(seed: int):
    rng = np.random.default_rng(seed)
    n_train, n_test = 3000, 600

    regime = rng.random(n_train) < 0.2  # True = minority regime B, False = majority regime A
    x_train = rng.normal(0, 1, n_train)
    segment_train = np.where(regime, rng.normal(3, 0.5, n_train), rng.normal(0, 0.5, n_train))
    w_A, w_B = 2.0, -2.0
    y_train = np.where(regime, w_B * x_train, w_A * x_train) + rng.normal(0, 0.3, n_train)
    X_train = pd.DataFrame({"x": x_train, "segment": segment_train})

    x_test = rng.normal(0, 1, n_test)
    segment_test = rng.normal(3, 0.5, n_test)
    y_test = w_B * x_test + rng.normal(0, 0.3, n_test)
    X_test = pd.DataFrame({"x": x_test, "segment": segment_test})

    return X_train, y_train, regime, X_test, y_test


def test_biz_val_adversarial_validator_diagnostic_detects_shift_and_flags_shifted_feature():
    X_train, _, _, X_test, _ = _make_shifted_scenario(seed=0)
    validator = AdversarialValidator(seed=0).fit(X_train, X_test)

    assert validator.auc_ > 0.85, f"expected the adversarial classifier to detect strong train/test separability, got AUC={validator.auc_:.4f}"
    report = validator.report()
    assert report.iloc[0]["feature"] == "segment", (
        f"expected 'segment' (the genuinely shifted column) to be the top-importance feature, got {report.iloc[0]['feature']!r}"
    )


def test_biz_val_adversarial_validator_fold_selection_tracks_true_test_score():
    X_train, y_train, regime, X_test, y_test = _make_shifted_scenario(seed=1)
    validator = AdversarialValidator(seed=1).fit(X_train, X_test)

    val_idx_adv, remainder_idx_adv = validator.select_validation_fold(val_fraction=0.2)

    rng = np.random.default_rng(2)
    perm = rng.permutation(len(y_train))
    n_val = len(val_idx_adv)
    val_idx_random, remainder_idx_random = perm[:n_val], perm[n_val:]

    def _fit_eval(remainder_idx, val_idx):
        model = LinearRegression().fit(X_train.iloc[remainder_idx][["x"]], y_train[remainder_idx])
        val_mae = mean_absolute_error(y_train[val_idx], model.predict(X_train.iloc[val_idx][["x"]]))
        test_mae = mean_absolute_error(y_test, model.predict(X_test[["x"]]))
        return val_mae, test_mae

    val_mae_adv, test_mae_adv = _fit_eval(remainder_idx_adv, val_idx_adv)
    val_mae_random, test_mae_random = _fit_eval(remainder_idx_random, val_idx_random)

    gap_adv = abs(val_mae_adv - test_mae_adv)
    gap_random = abs(val_mae_random - test_mae_random)
    assert gap_adv < gap_random * 0.2, (
        f"expected the facade's selected fold to track true test performance far more closely than random: gap_adv={gap_adv:.4f} gap_random={gap_random:.4f}"
    )
    assert regime[val_idx_adv].mean() > 0.9


def test_adversarial_validator_report_before_fit_raises():
    validator = AdversarialValidator()
    try:
        validator.report()
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def _make_pruning_scenario(seed: int, n_train: int = 2500, n_test: int = 2500, n_drift: int = 3, n_clean: int = 12):
    """Only ``n_drift`` columns are shifted between train/test; the remaining ``n_clean`` columns are iid noise
    on both sides, so the minimal drift-driving subset is exactly known ahead of time."""
    rng = np.random.default_rng(seed)
    drift_cols = [f"drift_{i}" for i in range(n_drift)]
    clean_cols = [f"clean_{i}" for i in range(n_clean)]

    train_data = {c: rng.normal(0, 1, n_train) for c in drift_cols}
    train_data.update({c: rng.normal(0, 1, n_train) for c in clean_cols})
    test_data = {c: rng.normal(4, 1, n_test) for c in drift_cols}  # shifted mean -> genuine drift
    test_data.update({c: rng.normal(0, 1, n_test) for c in clean_cols})  # same distribution as train

    X_train = pd.DataFrame(train_data)[drift_cols + clean_cols]
    X_test = pd.DataFrame(test_data)[drift_cols + clean_cols]
    return X_train, X_test, set(drift_cols)


def test_biz_val_adversarial_validator_prune_drift_features_isolates_known_drift_subset():
    X_train, X_test, known_drift_cols = _make_pruning_scenario(seed=3)
    validator = AdversarialValidator(seed=3).fit(X_train, X_test)

    baseline_auc = validator.auc_
    assert baseline_auc > 0.9, f"sanity check: expected strong baseline separability before pruning, got AUC={baseline_auc:.4f}"

    validator.prune_drift_features(target_auc=0.6, max_iterations=8, features_per_iteration=1)

    pruned = set(validator.pruned_features_)
    precision = len(pruned & known_drift_cols) / len(pruned) if pruned else 0.0
    recall = len(pruned & known_drift_cols) / len(known_drift_cols)

    assert precision >= 0.8, f"expected pruned set to mostly match the known drift columns, precision={precision:.3f} pruned={pruned}"
    assert recall == 1.0, f"expected all known drift columns to be pruned, recall={recall:.3f} pruned={pruned} known={known_drift_cols}"
    assert validator.pruning_final_auc_ < 0.6, f"expected final AUC on the survivors to be near-chance, got {validator.pruning_final_auc_:.4f}"

    # fit()'s own attributes must be untouched by the opt-in pruning call.
    assert validator.auc_ == baseline_auc


def test_adversarial_validator_default_fit_unchanged_when_pruning_not_called():
    """Bit-identical default: calling fit() alone (never invoking prune_drift_features) must not set any
    pruning attributes, proving the new opt-in path has zero effect unless explicitly requested."""
    X_train, X_test, _ = _make_pruning_scenario(seed=4)
    validator = AdversarialValidator(seed=4).fit(X_train, X_test)
    for attr in ("pruned_features_", "remaining_features_", "pruning_history_", "pruning_final_auc_"):
        assert not hasattr(validator, attr), f"unexpected pre-set pruning attribute {attr!r} on a validator that never called prune_drift_features()"
