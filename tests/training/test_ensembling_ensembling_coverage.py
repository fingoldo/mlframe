"""§8.3 Ensembling test coverage gaps -- regression tests for previously uncovered code paths.

Sibling F2 (test_audit_2026_05_16_f2_ensembling.py) already covers:
  * P1 single-member no-op (test_score_ensemble_returns_empty_for_single_member)
  * P1 sample_weight threading (test_compute_member_quality_gate_accepts_sample_weight)

This file covers what F2 did not: the recurrent-augmented predict path and the RRF row_sum==0
degenerate-tie guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# §8.3 P1: _phase_recurrent.py:135 _build_recurrent_member_entry shape
# ---------------------------------------------------------------------------


def test_build_recurrent_member_entry_classification_shape():
    """A fitted recurrent classifier's (N, K) probability output must wrap into a member entry that
    score_ensemble can blend: both *_preds and *_probs slots populated, model_name retained."""
    from mlframe.training.core._phase_recurrent import _build_recurrent_member_entry

    n = 30
    train_probs = np.column_stack([np.linspace(0.9, 0.1, n), np.linspace(0.1, 0.9, n)])
    val_probs = np.column_stack([np.linspace(0.8, 0.2, n), np.linspace(0.2, 0.8, n)])
    test_probs = np.column_stack([np.linspace(0.7, 0.3, n), np.linspace(0.3, 0.7, n)])
    member = _build_recurrent_member_entry(
        recurrent_model_name="recurrent_lstm",
        model=SimpleNamespace(),  # opaque -- only attribute access tested
        train_preds=train_probs,
        val_preds=val_probs,
        test_preds=test_probs,
        is_classification=True,
    )
    assert member.model_name == "recurrent_lstm"
    # Classification: binary-positive column projected onto *_preds; full prob tensor stays on *_probs.
    assert member.val_probs is val_probs
    assert member.val_preds is not None and member.val_preds.shape == (n,)
    # train_preds / test_preds also collapsed to the positive column.
    assert member.train_preds.shape == (n,)
    assert member.test_preds.shape == (n,)


def test_build_recurrent_member_entry_regression_shape():
    """The regression branch leaves probs slots None and routes raw predictions onto *_preds only.
    A regression sentinel against accidental classification-style probability wrapping."""
    from mlframe.training.core._phase_recurrent import _build_recurrent_member_entry

    n = 20
    train_preds = np.linspace(0.0, 10.0, n)
    val_preds = np.linspace(5.0, 15.0, n)
    test_preds = np.linspace(10.0, 20.0, n)
    member = _build_recurrent_member_entry(
        recurrent_model_name="recurrent_rnn",
        model=SimpleNamespace(),
        train_preds=train_preds,
        val_preds=val_preds,
        test_preds=test_preds,
        is_classification=False,
    )
    assert member.val_probs is None and member.train_probs is None and member.test_probs is None
    assert member.val_preds is val_preds
    assert member.train_preds is train_preds and member.test_preds is test_preds


# ---------------------------------------------------------------------------
# §8.3 P2: _rrf_aggregate_probs row_sum==0 degenerate guard (ensembling.py:438 area)
# ---------------------------------------------------------------------------


def test_rrf_aggregate_probs_handles_degenerate_zero_aggregate():
    """If every member produces an all-zero per-row score (the degenerate tie case where the safe
    fallback to 1.0 fires), the RRF aggregator must NOT divide by zero and must return finite
    output. Force the degeneracy by feeding three identical (N, K=2) prob matrices where every
    row has both probabilities equal so the rank tie collapses similarly across members; the
    re-normalisation guard at ensembling.py:608-610 keeps the output finite."""
    from mlframe.models.ensembling import _rrf_aggregate_probs

    # Three identical, perfectly-tied probability tensors.
    n = 5
    p = np.full((n, 2), 0.5)
    stacked = np.stack([p, p, p], axis=0)
    out = _rrf_aggregate_probs(stacked, k=60)
    assert np.all(np.isfinite(out)), f"RRF output must be finite under degenerate tie; got {out}"
    # Row-normalised probs sum to ~1 per row.
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"row sums must be ~1 after re-normalisation; got {row_sums}"


def test_rrf_ensemble_three_identical_zero_preds_no_divide_by_zero():
    """End-to-end via the public ``rrf_ensemble`` wrapper: three identical 1-D zero score vectors
    must not raise and must produce finite output."""
    from mlframe.models.ensembling import rrf_ensemble

    n = 8
    p = np.zeros(n)
    out = rrf_ensemble([p, p, p], k=60)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# §8.3 P2: composite_ensemble per-family calibration is currently NOT implemented
# in the source tree (no calibrator field on CompositeCrossTargetEnsemble); covered
# as a construction smoke instead so we surface the moment the field appears.
# ---------------------------------------------------------------------------


def test_composite_ensemble_has_no_per_family_calibrators_today():
    """Sentinel for the §8.3 P2 ``per-family calibration before averaging`` finding: the feature is
    NOT in the current source tree. This test asserts the documented state so the moment a
    ``per_family_calibrator`` (or similar) attribute is added, the test fails loudly and the
    coverage gap can be filled with a behavioural assertion. Doubles as a structural sentry that
    blocks accidental silent introduction of a calibrator that is never replayed at predict
    time."""
    from mlframe.training.composite.ensemble import CompositeCrossTargetEnsemble
    # Construct via the cheapest factory available -- from_train_metrics needs only model metrics.
    from unittest.mock import MagicMock
    ens = CompositeCrossTargetEnsemble.from_train_metrics(
        component_models=[MagicMock(name="cb"), MagicMock(name="xgb")],
        component_names=["cb", "xgb"],
        component_train_rmse=[1.0, 1.2],
        component_oof_rmse=[1.0, 1.2],
    )
    # The currently-implemented attributes (regression sentry: don't lose any of these silently).
    assert hasattr(ens, "component_models")
    assert hasattr(ens, "weights")
    # The aspirational per-family calibrator slot is NOT present today; if it lands later,
    # convert this assertion into a positive behavioural test.
    assert not hasattr(ens, "per_family_calibrators"), (
        "per_family_calibrators slot appeared -- flip this test into a positive replay assertion"
    )
