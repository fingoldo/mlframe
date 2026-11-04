"""Faith-Shap surrogate losses sit on the same scale as the real proxy losses they are merged with."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_faith_interactions import faith_interaction_top_n
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_heuristics import _Evaluator


def test_surrogate_loss_tracks_the_real_loss_of_the_same_subset():
    """With a well-fitting surrogate the loss it reports for a subset is within a small fraction of the real loss, not shifted by the intercept."""
    rng = np.random.default_rng(0)
    n, f = 600, 6
    phi = rng.normal(0, 0.2, size=(n, f))
    phi[:, 0] += 1.0
    phi[:, 1] += 0.6
    base = np.zeros(n)
    y = phi[:, 0] + phi[:, 1] + 0.5 * phi[:, 2] + rng.normal(0, 0.1, n) + 3.0
    cands, info = faith_interaction_top_n(
        phi, base, y, classification=False, metric="rmse", candidate_pairs=[(0, 1)], min_card=1, max_card=4, top_n=5, n_coalitions=512, rng=np.random.default_rng(1)
    )
    assert abs(info["a0"]) > 0.5, "the bed must have a sizeable intercept for the shift to show"
    ev = _Evaluator(phi, base, y, "rmse")
    for loss, key in cands:
        real = float(ev.loss(list(key)))
        assert abs(loss - real) < 0.35 * abs(real) + 0.2, f"surrogate loss {loss:.3f} vs real {real:.3f} for {key}"
