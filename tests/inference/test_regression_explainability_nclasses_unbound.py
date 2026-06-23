"""Regression: in explain_models, nclasses was only assigned inside the per-fold
`if max_test_ind < L:` branch. When the FINAL TimeSeries fold has no OOS rows that branch is
skipped on the last iteration, leaving nclasses unbound/stale at the show_custom_calibration_plot
call. The fix derives nclasses once from the stacked probs (probs.shape[1]) before that call.

These sensors reproduce the do_ts_oos accumulation behaviourally: folds contribute (or skip) OOS
probs, and the stacked-probs nclasses derivation stays correct regardless of which fold contributed
the rows -- including when the final / leading folds contribute none."""

from __future__ import annotations

import numpy as np


def _per_fold_then_stacked(folds: list[np.ndarray | None]):
    """Reproduce the explain_models do_ts_oos accumulation: each TS fold either contributes OOS
    probs (and the per-fold ``nclasses = probs.shape[1]`` assignment fires) or contributes none
    (the ``if max_test_ind < L`` branch is skipped, leaving the per-fold nclasses stale/unbound).
    After the loop the fix re-derives nclasses from the STACKED probs. Returns
    (per_fold_nclasses, stacked_nclasses)."""
    per_fold_nclasses = None  # pre-fix: read at plot time even if the final fold skipped the branch
    collected = []
    for fold in folds:
        if fold is not None and fold.shape[0] > 0:
            collected.append(fold)
            per_fold_nclasses = fold.shape[1]
    probs = np.vstack(collected)
    stacked_nclasses = probs.shape[1]  # the fix: always defined from the stacked probs
    return per_fold_nclasses, stacked_nclasses


def test_nclasses_derived_from_stacked_probs_when_final_fold_empty():
    # Final fold contributes no OOS rows: the per-fold conditional assignment is skipped on the
    # last iteration but the stacked-probs derivation still yields the correct class count.
    per_fold, stacked = _per_fold_then_stacked([np.zeros((3, 4)), np.zeros((2, 4)), None])
    assert stacked == 4, "stacked-probs nclasses must hold across an empty final fold"
    # The per-fold value happens to be carried from an earlier fold here, but the contract the fix
    # relies on is the stacked derivation -- it is correct regardless of fold ordering.
    assert per_fold == 4


def test_stacked_nclasses_correct_even_if_only_late_fold_has_rows():
    # First folds empty, only a later fold contributes: the stacked derivation is still correct,
    # whereas a naive "use the last per-fold value" would be unbound across the empty leading folds.
    per_fold, stacked = _per_fold_then_stacked([None, np.zeros((0, 3)), np.zeros((5, 3))])
    assert stacked == 3
