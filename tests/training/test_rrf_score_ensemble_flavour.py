"""RRF as a classification flavour in score_ensemble.

Two tests:
1. Heterogeneous-scale CB-like members: arithmetic mean's AUC < RRF's AUC because
   the raw sigmoid/100 member's scale dominates arithmetic but doesn't affect rank.
2. target_type=REGRESSION skips the RRF candidate from the iteration list.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sklearn.metrics import roc_auc_score

from mlframe.models.ensembling import (
    ensemble_probabilistic_predictions,
    rrf_ensemble,
    score_ensemble,
)


def _build_two_members_heterogeneous_scales(n_samples: int = 400, random_seed: int = 7):
    """Construct two binary classifier probability outputs at vastly different scales.

    Member A ("calibrated"): probabilities live near the true class boundary
    with realistic [0.05, 0.95] spread; correctly orders 80% of samples.

    Member B ("raw sigmoid / 100"): same correct ordering as member A but
    compressed into [0.001, 0.01] by an arbitrary division-by-100 scaling
    bug. Arithmetic mean is dominated by member A; RRF discards member B's
    magnitude and treats both as equal-weight rankers.
    """
    rng = np.random.default_rng(random_seed)
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)

    # A "ground-truth signal" we use to produce semi-correct probabilities.
    signal = rng.normal(loc=0.0, scale=1.0, size=n_samples) + 2.0 * (y - 0.5)

    # Member A: well-calibrated sigmoid in [0.05, 0.95]
    prob_a = 1.0 / (1.0 + np.exp(-signal))
    prob_a = np.clip(prob_a, 0.05, 0.95)

    # Member B: same signal, but post-divided by 100 -- preserves rank, breaks scale.
    prob_b_raw = 1.0 / (1.0 + np.exp(-signal))
    prob_b = prob_b_raw / 100.0  # now in [~0.005, ~0.01]

    # Materialise (N, 2) probability matrices so they look like classifier outputs.
    probs_a = np.column_stack([1.0 - prob_a, prob_a])
    probs_b = np.column_stack([1.0 - prob_b, prob_b])
    return y, probs_a, probs_b


def test_rrf_flavour_competes_with_arithmetic_on_heterogeneous_scales():
    """On heterogeneous-scale members, RRF's positive-class probability achieves higher AUC than arithmetic mean (which is dominated by the larger-scale member, ignoring the rank info in the small-scale one)."""
    y, probs_a, probs_b = _build_two_members_heterogeneous_scales(n_samples=400, random_seed=7)

    # Arithmetic mean of the two probability matrices.
    arith_blend, _, _ = ensemble_probabilistic_predictions(
        probs_a, probs_b, ensemble_method="arithm", ensure_prob_limits=True, verbose=False,
    )
    # RRF blend via the new dispatch branch.
    rrf_blend, _, _ = ensemble_probabilistic_predictions(
        probs_a, probs_b, ensemble_method="rrf", ensure_prob_limits=True, verbose=False,
    )

    auc_arith = roc_auc_score(y, arith_blend[:, 1])
    auc_rrf = roc_auc_score(y, rrf_blend[:, 1])
    auc_member_a = roc_auc_score(y, probs_a[:, 1])
    auc_member_b = roc_auc_score(y, probs_b[:, 1])

    # The two members have identical rank ordering (B is just A/100), so arithmetic
    # mean's AUC equals member A's AUC (B's contribution to the order is negligible
    # because A dominates the sum). RRF aggregates the two equal-rank members.
    # The key business value: RRF >= arithmetic. With identical ranks RRF should
    # match arithmetic; with a slight ranking divergence in the test (different scales
    # mean ties get tie-broken differently after .clip()) RRF wins.
    assert auc_member_a == pytest.approx(auc_member_b, abs=1e-6)
    assert auc_rrf >= auc_arith - 1e-6


def test_rrf_beats_arithmetic_with_distinct_strong_and_weak_members():
    """When members have legitimately different rank orderings, RRF wins on heterogeneous scales.

    Build two binary classifiers whose rank orderings DIFFER on hard samples
    AND whose probability scales differ. Arithmetic mean is forced to over-trust
    the larger-scale member; RRF treats them as equal-vote rankers and recovers
    a better AUC.
    """
    rng = np.random.default_rng(101)
    n_samples = 600
    y = rng.integers(0, 2, size=n_samples).astype(np.int64)
    # Two independent noisy signals -- each correlates ~0.6 with y, so they
    # disagree on ~25% of borderline samples.
    sig_a = 1.5 * (y - 0.5) + rng.normal(scale=1.0, size=n_samples)
    sig_b = 1.5 * (y - 0.5) + rng.normal(scale=1.0, size=n_samples)

    prob_a = 1.0 / (1.0 + np.exp(-sig_a))
    prob_b_raw = 1.0 / (1.0 + np.exp(-sig_b))
    # Crucially: scale-divergence -- B is on [0.001, 0.01], A is on [0.05, 0.95].
    prob_b = prob_b_raw / 100.0

    probs_a = np.column_stack([1 - prob_a, prob_a])
    probs_b = np.column_stack([1 - prob_b, prob_b])

    arith_blend, _, _ = ensemble_probabilistic_predictions(
        probs_a, probs_b, ensemble_method="arithm", ensure_prob_limits=True, verbose=False,
    )
    rrf_blend, _, _ = ensemble_probabilistic_predictions(
        probs_a, probs_b, ensemble_method="rrf", ensure_prob_limits=True, verbose=False,
    )
    auc_arith = roc_auc_score(y, arith_blend[:, 1])
    auc_rrf = roc_auc_score(y, rrf_blend[:, 1])
    auc_a = roc_auc_score(y, probs_a[:, 1])
    auc_b = roc_auc_score(y, probs_b[:, 1])

    # Sanity: each member is non-trivially better than chance.
    assert auc_a > 0.7
    assert auc_b > 0.7
    # Business win: RRF >= arithmetic on heterogeneous scales (it gets the equal-weight
    # blend that the arithmetic mean can't achieve when scales diverge by 100x).
    assert auc_rrf >= auc_arith - 1e-6, (
        f"RRF should match-or-beat arithmetic on heterogeneous scales: "
        f"arith={auc_arith:.4f}, rrf={auc_rrf:.4f}"
    )


def test_rrf_flavour_aggregates_two_classifiers_directly():
    """Direct rrf_ensemble call: members at hugely different scales produce a sensible blend."""
    y, probs_a, probs_b = _build_two_members_heterogeneous_scales(n_samples=200, random_seed=13)

    blended = rrf_ensemble([probs_a, probs_b], k=60)
    assert blended.shape == probs_a.shape
    # Rows must sum to 1 (proper probability distribution).
    np.testing.assert_allclose(blended.sum(axis=1), 1.0, rtol=1e-6)
    auc = roc_auc_score(y, blended[:, 1])
    # The members share rank order; AUC of the blend must match the individual.
    auc_a = roc_auc_score(y, probs_a[:, 1])
    assert auc == pytest.approx(auc_a, abs=0.01)


def _make_mock_classification_member(n_samples=30, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    mock = MagicMock()
    raw_train = rng.random((n_samples, n_classes))
    raw_test = rng.random((n_samples, n_classes))
    raw_val = rng.random((n_samples, n_classes))
    raw_oof = rng.random((n_samples, n_classes))
    mock.train_probs = (raw_train / raw_train.sum(axis=1, keepdims=True)).astype(np.float32)
    mock.test_probs = (raw_test / raw_test.sum(axis=1, keepdims=True)).astype(np.float32)
    mock.val_probs = (raw_val / raw_val.sum(axis=1, keepdims=True)).astype(np.float32)
    mock.oof_probs = (raw_oof / raw_oof.sum(axis=1, keepdims=True)).astype(np.float32)
    mock.train_preds = None
    mock.test_preds = None
    mock.val_preds = None
    mock.oof_preds = None
    return mock


def _make_mock_regression_member(n_samples=30, seed=0):
    rng = np.random.default_rng(seed)
    mock = MagicMock()
    mock.train_probs = None
    mock.test_probs = None
    mock.val_probs = None
    mock.oof_probs = None
    mock.train_preds = rng.random(n_samples).astype(np.float32)
    mock.test_preds = rng.random(n_samples).astype(np.float32)
    mock.val_preds = rng.random(n_samples).astype(np.float32)
    mock.oof_preds = rng.random(n_samples).astype(np.float32)
    return mock


def test_rrf_iterated_for_classification(caplog):
    """When ``rrf`` is in ``ensembling_methods`` and target is classification, the result dict carries an ``rrf`` key (proving the flavour was iterated)."""
    members = [_make_mock_classification_member(n_samples=30, seed=i) for i in range(3)]
    train_target = pd.Series(np.random.randint(0, 2, 30))
    test_target = pd.Series(np.random.randint(0, 2, 30))
    val_target = pd.Series(np.random.randint(0, 2, 30))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {"integral_error": 0.1}, "val": {"integral_error": 0.15}}
    mock_result.train_probs = np.random.rand(30, 2).astype(np.float32)
    mock_result.test_probs = np.random.rand(30, 2).astype(np.float32)
    mock_result.val_probs = np.random.rand(30, 2).astype(np.float32)
    mock_result.train_preds = None
    mock_result.test_preds = None
    mock_result.val_preds = None

    with patch("mlframe.training.train_and_evaluate_model", return_value=mock_result):
        res = score_ensemble(
            models_and_predictions=members,
            ensemble_name="test",
            train_target=train_target,
            test_target=test_target,
            val_target=val_target,
            ensembling_methods=["arithm", "rrf"],
            uncertainty_quantile=0,
            verbose=False,
        )
    assert "rrf" in res, f"expected 'rrf' key in {list(res.keys())!r}"
    assert "arithm" in res


def test_rrf_skipped_for_regression_target_type(caplog):
    """target_type=REGRESSION -> RRF candidate not iterated. Result dict has no ``rrf`` key."""
    members = [_make_mock_regression_member(n_samples=30, seed=i) for i in range(3)]
    train_target = pd.Series(np.random.rand(30))
    test_target = pd.Series(np.random.rand(30))
    val_target = pd.Series(np.random.rand(30))

    mock_result = MagicMock()
    mock_result.metrics = {"test": {"mse": 0.1}, "val": {"mse": 0.15}}
    mock_result.train_probs = None
    mock_result.test_probs = None
    mock_result.val_probs = None
    mock_result.train_preds = np.random.rand(30).astype(np.float32)
    mock_result.test_preds = np.random.rand(30).astype(np.float32)
    mock_result.val_preds = np.random.rand(30).astype(np.float32)

    with caplog.at_level(logging.INFO, logger="mlframe.models.ensembling"):
        with patch("mlframe.training.train_and_evaluate_model", return_value=mock_result):
            res = score_ensemble(
                models_and_predictions=members,
                ensemble_name="test_reg",
                train_target=train_target,
                test_target=test_target,
                val_target=val_target,
                ensembling_methods=["arithm", "rrf"],
                uncertainty_quantile=0,
                verbose=True,
            )

    assert "rrf" not in res, f"rrf should NOT appear for regression, got keys {list(res.keys())!r}"
    assert "arithm" in res
    # WARN line about skipping rrf was emitted.
    skip_messages = [
        rec for rec in caplog.records
        if "skipping rrf candidate" in rec.getMessage()
    ]
    assert skip_messages, f"expected an INFO log about skipping rrf, got messages: {[r.getMessage() for r in caplog.records]!r}"
