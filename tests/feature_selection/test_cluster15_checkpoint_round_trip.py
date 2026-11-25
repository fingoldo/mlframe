"""Cluster 15 regression: the RFECV checkpoint keeps the no-features score, the per-split scores and the signed importances (audit 2026-11-20,
findings 15-F3, 15-F4 and 15-F5).

A resumed run used to compare against NaN, lose its per-split columns and mix importance scales. The state is now saved and restored by the
same round trip the run uses.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._checkpoint import _maybe_resume_from_checkpoint
from mlframe.feature_selection.wrappers.rfecv._fit_outer_loop import _save_outer_loop_checkpoint


class _Store:
    """In-memory checkpoint: the object exposes the save and load hooks the RFECV uses."""

    _CHECKPOINT_VERSION = 1

    def __init__(self):
        self.saved = None
        self.checkpoint_path = "in-memory"
        self._signed_importances = np.array([0.5, -0.25])

    def _save_checkpoint(self, state):
        """Keep the saved dict as the checkpoint payload."""
        self.saved = dict(state)

    def _load_checkpoint(self):
        """Return the saved payload, shaped as the load hook returns it."""
        return self.saved


def _state():
    """Outer-loop state with a non-NaN no-features score and two per-split rows."""
    return SimpleNamespace(
        nsteps=3,
        evaluated_scores_mean={5: 0.7, 4: 0.72},
        evaluated_scores_std={5: 0.01, 4: 0.02},
        feature_importances={"a": 1.0},
        selected_features_per_nfeatures={5: ["a"]},
        prev_score=0.72,
        prev_nfeatures=4,
        n_noimproving_iters=0,
        best_nfeatures=4,
        best_iter=1,
        best_score=0.72,
        dummy_scores=[0.5],
        Optimizer=None,
        nofeatures_score=0.5,
        per_fold_scores={4: [0.71, 0.73], 5: [0.69, 0.71]},
    )


def test_nofeatures_score_survives_a_round_trip():
    """The no-features baseline is restored, not NaN, so the dummy early-stop comparison works on resume (15-F3)."""
    store = _Store()
    _save_outer_loop_checkpoint(store, "sig", _state(), verbose=False)
    fresh = {"nofeatures_score": float("nan"), "per_fold_scores": {}, "nsteps": 0, "evaluated_scores_mean": {}, "evaluated_scores_std": {},
             "feature_importances": {}, "selected_features_per_nfeatures": {}, "prev_score": 0.0, "prev_nfeatures": 0,
             "n_noimproving_iters": 0, "best_nfeatures": 0, "best_iter": 0, "best_score": -np.inf, "dummy_scores": [], "Optimizer": None}
    restored = _maybe_resume_from_checkpoint(store, signature="sig", verbose=False, state=fresh)
    assert restored["nofeatures_score"] == 0.5


def test_per_split_scores_survive_a_round_trip():
    """The per-split score lists come back for every evaluated N (15-F4)."""
    store = _Store()
    _save_outer_loop_checkpoint(store, "sig", _state(), verbose=False)
    fresh = {"nofeatures_score": float("nan"), "per_fold_scores": {}, "nsteps": 0, "evaluated_scores_mean": {}, "evaluated_scores_std": {},
             "feature_importances": {}, "selected_features_per_nfeatures": {}, "prev_score": 0.0, "prev_nfeatures": 0,
             "n_noimproving_iters": 0, "best_nfeatures": 0, "best_iter": 0, "best_score": -np.inf, "dummy_scores": [], "Optimizer": None}
    restored = _maybe_resume_from_checkpoint(store, signature="sig", verbose=False, state=fresh)
    assert restored["per_fold_scores"] == {4: [0.71, 0.73], 5: [0.69, 0.71]}


def test_signed_importances_survive_a_round_trip():
    """The signed coefficients are restored onto the estimator, so dispatched linear aggregation sees the same scale (15-F5)."""
    store = _Store()
    _save_outer_loop_checkpoint(store, "sig", _state(), verbose=False)
    fresh = {"nofeatures_score": float("nan"), "per_fold_scores": {}, "nsteps": 0, "evaluated_scores_mean": {}, "evaluated_scores_std": {},
             "feature_importances": {}, "selected_features_per_nfeatures": {}, "prev_score": 0.0, "prev_nfeatures": 0,
             "n_noimproving_iters": 0, "best_nfeatures": 0, "best_iter": 0, "best_score": -np.inf, "dummy_scores": [], "Optimizer": None}
    _maybe_resume_from_checkpoint(store, signature="sig", verbose=False, state=fresh)
    assert np.allclose(store._signed_importances, [0.5, -0.25])
