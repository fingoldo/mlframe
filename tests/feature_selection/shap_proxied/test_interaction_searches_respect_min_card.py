"""The interaction searches only return subsets inside the min_card..max_card window."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interaction_proxy import interaction_proxy_top_n
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import interaction_top_n


def _bed(n=120, p=6):
    """Main-effect SHAP, a symmetric interaction tensor, base and target for a tiny regression."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=n)
    phi = rng.normal(0, 0.1, size=(n, p))
    phi[:, 0] += 0.5 * y
    phi[:, 1] += 0.3 * y
    Phi = rng.normal(0, 0.02, size=(n, p, p))
    Phi = (Phi + Phi.transpose(0, 2, 1)) / 2
    return phi, Phi, np.zeros(n), y


def test_interaction_proxy_search_honours_min_card():
    """With min_card=3 no size-1 or size-2 subset comes back; before, the greedy pass cached every size from 1 and the final list was not filtered."""
    phi, Phi, base, y = _bed()
    out = interaction_proxy_top_n(phi, Phi, base, y, classification=False, metric="rmse", min_card=3, max_card=5, top_n=30)
    assert out and all(3 <= len(k) <= 5 for _, k in out)


def test_interaction_top_n_honours_min_card_and_max_card():
    """The exhaustive-plus-greedy search filters the same way, including the upper bound."""
    _phi, Phi, base, y = _bed()
    out = interaction_top_n(Phi, base, y, classification=False, metric="rmse", min_card=2, max_card=3, top_n=30)
    assert out and all(2 <= len(k) <= 3 for _, k in out)
