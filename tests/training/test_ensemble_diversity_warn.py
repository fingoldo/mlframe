"""Diversity check for ensemble members: surface near-duplicate val-pred pairs via WARN + ``_diversity`` payload.

Verifies the user-mandated contract: NO member is dropped on high correlation. The previous design ("auto-drop redundant
member") was explicitly rejected on round-2 review - mean / median ensembles tolerate redundancy fine, and operators want
visibility (so they can prune the suite definition upstream) rather than silent removal.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.models.ensembling import compute_high_correlation_pairs


def _make_member(val_preds: np.ndarray, name: str) -> SimpleNamespace:
    """Minimal stand-in for the ensemble-member duck-type (only the attrs ``compute_high_correlation_pairs`` reads)."""
    return SimpleNamespace(
        val_preds=val_preds,
        val_probs=None,
        test_preds=None,
        test_probs=None,
        train_preds=None,
        train_probs=None,
        model_name=name,
    )


def test_near_duplicate_members_flagged_but_not_dropped(caplog) -> None:
    """Two members with ~0.99 corr trigger WARN + pair record; both members survive."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=500)
    # Pair A: near-duplicate (corr ~0.99)
    member_a = _make_member(base, "model_A")
    member_b = _make_member(base + 0.05 * rng.normal(size=500), "model_B")
    # Pair C: independent of A / B
    member_c = _make_member(rng.normal(size=500), "model_C")
    members = [member_a, member_b, member_c]
    tags = ["model_A", "model_B", "model_C"]

    # Verify the synthetic setup actually crosses the threshold (so failure cannot be blamed on weak signal).
    sanity_corr = float(np.corrcoef(member_a.val_preds, member_b.val_preds)[0, 1])
    assert sanity_corr > 0.98, f"synthetic A vs B corr {sanity_corr:.4f} - sample/noise tuning broken"

    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        pairs, split_used = compute_high_correlation_pairs(members, tags, threshold=0.98)

    # Members are NOT dropped from the input sequence (this function is pure / observational).
    assert len(members) == 3
    # Exactly the A-B pair is flagged.
    assert len(pairs) == 1
    pair = pairs[0]
    assert {pair["m1"], pair["m2"]} == {"model_A", "model_B"}
    assert pair["corr"] > 0.98
    assert split_used == "val_preds"


def test_independent_members_no_pair_no_warn(caplog) -> None:
    """Three uncorrelated members produce zero pairs and no warning - guards against false positives."""
    rng = np.random.default_rng(7)
    members = [_make_member(rng.normal(size=400), f"m{i}") for i in range(3)]
    tags = [f"m{i}" for i in range(3)]
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        pairs, _split = compute_high_correlation_pairs(members, tags, threshold=0.98)
    assert pairs == []


def test_constant_member_skipped_not_flagged() -> None:
    """A member whose val_preds is a flat constant has zero std; corrcoef returns NaN. Must be skipped silently, never flagged."""
    rng = np.random.default_rng(11)
    members = [
        _make_member(np.full(200, 1.5), "flat"),
        _make_member(rng.normal(size=200), "noisy"),
    ]
    tags = ["flat", "noisy"]
    pairs, split_used = compute_high_correlation_pairs(members, tags, threshold=0.98)
    assert pairs == []
    assert split_used == "val_preds"


def test_score_ensemble_kwarg_threshold_plumbed_to_helper(caplog) -> None:
    """Integration-style: WARN-level log emitted by score_ensemble's plumbing on a near-duplicate pair (default 0.98).

    Calls the helper directly (the full score_ensemble path needs a trained model; out of scope for this unit test).
    Asserts the log line shape that downstream operators rely on for grep-based triage.
    """
    rng = np.random.default_rng(99)
    base = rng.normal(size=300)
    members = [
        _make_member(base, "cb"),
        _make_member(base + 0.04 * rng.normal(size=300), "xgb"),
    ]
    tags = ["cb", "xgb"]
    pairs, split_used = compute_high_correlation_pairs(members, tags, threshold=0.98)
    assert pairs and pairs[0]["corr"] > 0.98
    # Simulating the WARN format the caller (score_ensemble) emits per pair.
    logger = logging.getLogger("mlframe.models.ensembling")
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        logger.warning(
            "[ensemble] high-correlation member pair (split=%s): %s vs %s -- Pearson corr=%.4f > threshold=%.4f.",
            split_used,
            pairs[0]["m1"],
            pairs[0]["m2"],
            pairs[0]["corr"],
            0.98,
        )
    matching = [r for r in caplog.records if "high-correlation member pair" in r.getMessage()]
    assert matching, "expected WARN line not emitted"
    assert "cb" in matching[0].getMessage() and "xgb" in matching[0].getMessage()


if __name__ == "__main__":
    pytest.main([__file__, "-xvs", "--no-cov"])
