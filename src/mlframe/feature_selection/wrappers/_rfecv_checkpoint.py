"""Checkpoint resume for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s pre-while setup. Restores mutable outer-
loop state iff the checkpoint signature matches the current
``(X.shape, y.shape, columns_key)``. Mismatch silently starts fresh so
users can keep the same ``checkpoint_path`` across experiments.

Re-imported at the parent's module bottom so historical
``from ._rfecv_fit import _maybe_resume_from_checkpoint`` keeps
resolving transparently.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def _maybe_resume_from_checkpoint(self, *, signature, verbose, state: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate the passed ``state`` dict in place to reflect any matching
    checkpoint, then return it.

    ``state`` carries the mutable outer-loop bindings (nsteps,
    evaluated_scores_*, feature_importances, selected_features_per_nfeatures,
    prev_score, prev_nfeatures, n_noimproving_iters, best_nfeatures,
    best_iter, best_score, dummy_scores, Optimizer). Caller unpacks the
    updated dict after the call so the parent function's locals stay in
    sync with what the checkpoint restored.
    """
    if self.checkpoint_path is None:
        return state
    _state = self._load_checkpoint()
    if _state is not None and _state.get("signature") == signature:
        state["nsteps"] = int(_state.get("nsteps", 0))
        state["evaluated_scores_mean"] = dict(_state.get("evaluated_scores_mean", {}))
        state["evaluated_scores_std"] = dict(_state.get("evaluated_scores_std", {}))
        state["feature_importances"] = dict(_state.get("feature_importances", {}))
        state["selected_features_per_nfeatures"] = dict(_state.get("selected_features_per_nfeatures", {}))
        state["prev_score"] = _state.get("prev_score", state["prev_score"])
        state["prev_nfeatures"] = _state.get("prev_nfeatures", state["prev_nfeatures"])
        state["n_noimproving_iters"] = int(_state.get("n_noimproving_iters", 0))
        state["best_nfeatures"] = int(_state.get("best_nfeatures", 0))
        state["best_iter"] = int(_state.get("best_iter", 0))
        state["best_score"] = _state.get("best_score", -np.inf)
        state["dummy_scores"] = list(_state.get("dummy_scores", []))
        # MBHOptimizer carries its own evaluation history; restore if present and current search method is MBH.
        _saved_optimizer = _state.get("optimizer")
        if _saved_optimizer is not None and state.get("Optimizer") is not None:
            state["Optimizer"] = _saved_optimizer
        if verbose:
            logger.info(
                "RFECV: resumed from checkpoint at %s (nsteps=%d, best_score=%s).",
                self.checkpoint_path, state["nsteps"],
                f"{state['best_score']:.4f}" if np.isfinite(state["best_score"]) else str(state["best_score"]),
            )
    elif _state is not None:
        if verbose:
            logger.info(
                "RFECV: checkpoint at %s does not match current "
                "(X, y, columns) signature; starting from scratch.",
                self.checkpoint_path,
            )
    return state
