"""Beam search returns subsets of size min_card..max_card, however large min_card is."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_heuristics import beam_search


def _bed():
    """Eight SHAP-like columns, the first four carrying the target."""
    rng = np.random.default_rng(0)
    n, f = 400, 8
    y = rng.normal(size=n)
    phi = rng.normal(0, 0.1, size=(n, f))
    for j, w in enumerate((0.4, 0.3, 0.2, 0.1)):
        phi[:, j] += w * y
    return phi, np.zeros(n), y


def test_min_four_max_six_reaches_size_six_and_drops_smaller_subsets():
    """Used to stop at size 3 (max_card - min_card + 1) and return only subsets below min_card."""
    phi, base, y = _bed()
    out = beam_search(phi, base, y, classification=False, metric="rmse", beam_width=4, min_card=4, max_card=6, top_n=20)
    sizes = {len(k) for _, k in out}
    assert out and sizes <= {4, 5, 6} and 6 in sizes


def test_min_equals_max_returns_that_size():
    """With min_card == max_card the loop never ran, so only single features came back."""
    phi, base, y = _bed()
    out = beam_search(phi, base, y, classification=False, metric="rmse", beam_width=4, min_card=3, max_card=3, top_n=5)
    assert out and all(len(k) == 3 for _, k in out)


def test_default_min_card_is_unchanged():
    """With min_card=1 every size from 1 to max_card is still searched."""
    phi, base, y = _bed()
    out = beam_search(phi, base, y, classification=False, metric="rmse", beam_width=4, max_card=3, top_n=50)
    assert {1, 2, 3} <= {len(k) for _, k in out}
