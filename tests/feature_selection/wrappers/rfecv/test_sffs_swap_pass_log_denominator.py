"""FS_WRAPPERS-9 (2026-08-05 audit): the SFFS swap-pass summary log must report the number of pairs
actually attempted (``min(len(swap_out), len(swap_in))``, since the loop runs ``zip(swap_out, swap_in)``),
not ``len(swap_out)`` -- which overstates the denominator whenever fewer drop-candidates exist than
kept-candidates.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np


def test_swap_pass_log_denominator_matches_attempted_pairs(caplog, monkeypatch):
    """Fewer swap-in candidates than swap-out candidates: the logged denominator must be the smaller count."""
    import sklearn.model_selection as sk_ms
    import mlframe.feature_selection.wrappers.rfecv._sffs as sffs_mod

    monkeypatch.setattr(sk_ms, "cross_val_score", lambda *a, **k: np.array([0.0]))

    # K=3 kept-candidates worth of FI history, but only 1 feature outside best_set to swap in.
    self_obj = SimpleNamespace(swap_top_k=3, _fit_sample_weight_=None)
    best_set = ["k0", "k1", "k2", "k3", "k4"]
    original_features = [*best_set, "extra0"]  # only 1 feature not in best_set
    selected_features_per_nfeatures = {5: list(best_set)}
    feature_importances = {0: {f: 0.1 for f in best_set}}

    with caplog.at_level(logging.INFO, logger="mlframe.feature_selection.wrappers.rfecv._sffs"):
        sffs_mod._sffs_swap_pass(
            self=self_obj,
            X=None,
            y=None,
            estimator=None,
            cv=None,
            scoring=None,
            best_nfeatures=5,
            best_score_ref=1.0,
            selected_features_per_nfeatures=selected_features_per_nfeatures,
            feature_importances=feature_importances,
            original_features=original_features,
            evaluated_scores_mean={},
            evaluated_scores_std={},
            verbose=1,
            ndigits=4,
        )

    summary = [r.getMessage() for r in caplog.records if "swap pass" in r.getMessage()]
    assert len(summary) == 1, f"expected exactly one swap-pass summary log, got {summary!r}"
    assert "0/1 paired swaps" in summary[0], f"expected denominator 1 (min(3 swap_out, 1 swap_in)), got: {summary[0]!r}"
