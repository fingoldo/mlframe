"""biz_value tests for ``mlframe.competition.threshold_range_rescaler``.

Covers the "magic correction" grid search:

* a positive-signal scenario mirroring the source trick ("revolving loan >0.4,
  boost by 0.8") — a genuine systematic miscalibration confined to one
  subgroup+prediction-range combination — the grid search must find a
  correction and improve held-out AUC.
* a companion honest-negative scenario with no genuine subgroup
  miscalibration, proving the grid search does not fabricate a meaningful
  correction out of CV noise (near-no-op, no material held-out improvement) —
  the overfitting-risk the tracker entry explicitly warns about.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.competition.threshold_range_rescaler import ThresholdRangeRescaler


def _make_subgroup_miscalibrated_dataset(n: int, seed: int):
    """Predictions systematically over-shoot for "revolving_loan" subjects once pred > 0.4.

    True labels are generated from a latent score; for the revolving-loan subgroup, the model's
    raw predicted probability is inflated above 0.4 (an over-confident systematic bias), exactly
    mirroring the "correct predictions > 0.4 for revolving loan, boost by 0.8" source trick (here
    the boost needed is a *reduction*, i.e. a multiplier < 1.0, since the model over-predicts).
    """
    rng = np.random.default_rng(seed)
    is_revolving = rng.random(n) < 0.35
    latent = rng.normal(size=n)
    true_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_prob).astype(np.int64)

    pred = true_prob + rng.normal(scale=0.05, size=n)
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)

    # inflate revolving-loan predictions above 0.4 by a fixed multiplicative bias
    inflate_hit = is_revolving & (pred > 0.4)
    pred[inflate_hit] = np.clip(pred[inflate_hit] * 1.6, 1e-6, 1.0 - 1e-6)

    subgroups = {"revolving_loan": is_revolving, "other_loan": ~is_revolving}
    return pred, y, subgroups


def test_biz_val_threshold_range_rescaler_finds_and_applies_subgroup_correction():
    """Grid search finds the correct subgroup/threshold/shrinking-multiplier and improves held-out AUC by >0.01."""
    pred, y, subgroups = _make_subgroup_miscalibrated_dataset(n=6000, seed=0)

    pred_train, pred_test, y_train, y_test, idx_train, idx_test = train_test_split(pred, y, np.arange(len(pred)), test_size=0.4, random_state=0, stratify=y)
    subgroups_train = {name: mask[idx_train] for name, mask in subgroups.items()}
    subgroups_test = {name: mask[idx_test] for name, mask in subgroups.items()}

    baseline_test_auc = roc_auc_score(y_test, pred_test)

    rescaler = ThresholdRangeRescaler(
        thresholds=np.linspace(0.1, 0.9, 17),
        multipliers=np.linspace(0.5, 1.3, 17),
        n_splits=5,
        max_corrections=3,
        random_state=0,
    )
    rescaler.fit(pred_train, y_train, subgroups_train)

    assert len(rescaler.corrections_) >= 1, "expected the grid search to find at least one correction"
    best = rescaler.corrections_[0]
    assert best.subgroup == "revolving_loan", f"expected the correction to target the miscalibrated subgroup, got {best.subgroup}"
    assert 0.2 <= best.threshold <= 0.6, f"expected threshold near the true 0.4 boundary, got {best.threshold}"
    assert best.multiplier < 1.0, f"model over-predicts on the subgroup, expected a shrinking multiplier < 1.0, got {best.multiplier}"

    corrected_test = rescaler.transform(pred_test, subgroups_test)
    corrected_test_auc = roc_auc_score(y_test, corrected_test)

    assert corrected_test_auc > baseline_test_auc, f"expected held-out AUC improvement, baseline={baseline_test_auc:.4f}, corrected={corrected_test_auc:.4f}"
    improvement = corrected_test_auc - baseline_test_auc
    assert improvement > 0.01, f"expected a real (>0.01) held-out AUC gain, got {improvement:.4f}"


def test_biz_val_threshold_range_rescaler_noop_when_no_genuine_subgroup_miscalibration():
    """Honest negative control: no subgroup miscalibration exists, so the CV-selected correction
    must be a near no-op (or none at all), demonstrating the grid search does not fabricate a
    "magic number" fix out of CV noise -- the exact overfitting risk the source idea warns about.
    """
    rng = np.random.default_rng(7)
    n = 6000
    is_revolving = rng.random(n) < 0.35
    latent = rng.normal(size=n)
    true_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_prob).astype(np.int64)
    pred = np.clip(true_prob + rng.normal(scale=0.05, size=n), 1e-6, 1.0 - 1e-6)  # well-calibrated everywhere

    subgroups = {"revolving_loan": is_revolving, "other_loan": ~is_revolving}

    pred_train, pred_test, y_train, y_test, idx_train, idx_test = train_test_split(pred, y, np.arange(len(pred)), test_size=0.4, random_state=1, stratify=y)
    subgroups_train = {name: mask[idx_train] for name, mask in subgroups.items()}
    subgroups_test = {name: mask[idx_test] for name, mask in subgroups.items()}

    baseline_test_auc = roc_auc_score(y_test, pred_test)

    rescaler = ThresholdRangeRescaler(
        thresholds=np.linspace(0.1, 0.9, 17),
        multipliers=np.linspace(0.5, 1.5, 21),
        n_splits=5,
        max_corrections=3,
        min_improvement=1e-4,
        random_state=1,
    )
    rescaler.fit(pred_train, y_train, subgroups_train)

    corrected_test = rescaler.transform(pred_test, subgroups_test)
    corrected_test_auc = roc_auc_score(y_test, corrected_test)

    # either no correction was accepted, or every accepted multiplier is close to a no-op
    if rescaler.corrections_:
        for correction in rescaler.corrections_:
            assert abs(correction.multiplier - 1.0) < 0.15, f"expected near-no-op multiplier on well-calibrated data, got {correction.multiplier}"

    # held-out AUC must not meaningfully move (CV-fit noise should not survive to held-out data)
    assert (
        abs(corrected_test_auc - baseline_test_auc) < 0.01
    ), f"expected negligible held-out AUC change on well-calibrated data, baseline={baseline_test_auc:.4f}, corrected={corrected_test_auc:.4f}"
